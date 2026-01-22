import json
import jsonlines
import sys
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import re
import tempfile
from torchvision import transforms


import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy
import random
from functools import partial

# from one_stage_prompt_manager import OneStagePromptManager

# import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)
from transformers import AutoImageProcessor, AutoModel
import json
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms


from scipy.spatial.transform import Rotation as R
# from utils1 import convert_weights_cuda_cpu


# from semseg.rednet import RedNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

# from utils1.semantic_utils import color_label

# GPT
gpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../GPT"))
sys.path.append(gpt_dir)

from api import gpt_infer_back_track 
from one_stage_prompt_manager import OneStagePromptManager
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
#RAM
import argparse
import numpy as np
import random

import torch
sys.path.append("recognize-anything/ram")
sys.path.append("recognize-anything/ram")
from waypoint_predictor.TRM_net import BinaryDistPredictor_TRM
sys.path.insert(0, os.path.abspath(".")) 
from PIL import Image
from models import ram_plus
from inference import inference_ram  





class BaseVLNCETrainer(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0


    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.envs)
        print('Model version: 4o')



    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''

        
        self.waypoint_predictor = BinaryDistPredictor_TRM()
        self.waypoint_predictor.load_state_dict(
            torch.load(
                'waypoint_predictor/checkpoints/final-camera-ready',
                map_location = torch.device('cpu'),
            )['predictor']['state_dict'],strict=False
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False
        logger.info(f"Finish load WP, start Zero shot nav")
        logger.info("Finished setting up policy.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)
    
    



    def make_equiv_action(self, a_t, candidates_dict, traj=None):
        """
        Executes actions based on waypoint_id found via a_t as an index.

        Parameters:
            - a_t: integer index for the selected action (output of parse_json_action)
            - candidates_dict: dictionary of candidates with angle/distance keyed by waypoint_id
            - waypoint_ids: ordered list of waypoint IDs for direct index access
            - traj: optional, to update the trajectory path with selected actions

        Returns:
            - env_actions: List of dictionaries specifying action and action_args.
        """
        env_actions = []

        if a_t[0] == -1:  # assuming index -1 corresponds to 'stop'
            env_actions.append({'action': {'action': 0, 'action_args': {}}})
            if traj is not None:
                traj[0]['path'].append("stop")
        else:
            # Use a_t to get the correct waypoint_id and retrieve angle and distance
            angle = candidates_dict[a_t[0]]['angle']
            distance = candidates_dict[a_t[0]]['distance']
            env_actions.append({
                'action': {
                    'action': 4,  # assuming 4 denotes the "move" action
                    'action_args': {
                        'angle': angle,
                        'distance': distance
                    }
                }
            })

            # Update the trajectory path if `traj` is provided
            if traj is not None:
                traj[0]['path'].append(candidates_dict[a_t[0]]['waypoint'])

        return env_actions
    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()
        config.defrost()

        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print("skipping -- evaluation exists.")
                return

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        )

        dataset_length = sum(envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)
        obs_transforms = get_active_obs_transforms(config)
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.envs = envs
        self._build_prompt_manager()  
        self.policy.eval()
        self.waypoint_predictor.eval()
        
        observations = envs.reset()
        instruction_data = observations[0]["instruction"]
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        # prev_actions = torch.zeros(
        #     envs.num_envs, 1, device=self.device, dtype=torch.long
        # )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm(total=episodes_to_eval) if config.use_pbar else None
        if 'infos' not in locals():
            infos = {0: {'steps_taken': 0}}  
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        traj = [{
                'path': [],
                'details': {},
                'a_t': {},
            }]
        start_time = time.time()
        DINO_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')

        episodes_to_eval = 100
        currid = 0
        step=0
        img_id = 0
        trajectory = []
        history = []
        stop=False
        num_epi = 0
        ended = np.array([False] * envs.num_envs)
        just_ended = np.array([False] * envs.num_envs)
        ram_model = ram_plus(pretrained="ram_plus_swin_large_14m.pth",
                            image_size=384,
                            vit='swin_l').cuda()
        ram_model.eval()
        self.prompt_manager.history = ['']
        self.prompt_manager.panoramic_history = ['']
        self.prompt_manager.nodes_list = [[]]
        self.prompt_manager.node_imgs = [[]]
        self.prompt_manager.graph = [{}]
        self.prompt_manager.trajectory = [[]]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."]]
        self.prompt_manager.last_action = ""
        self.prompt_manager.backtrack=False
        self.episode_api_time = 0
        self.episode_api_time_list = []
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            if currid != current_episodes[0].episode_id:
                trajectory = [str(0)]
                history = []
                img_id = 0
                step=0
                num_epi+=1
                stop=False
            else:
                step+=1
            currid = current_episodes[0].episode_id
            
            positions = []; headings = []
            for ob_i in range(len(current_episodes)):
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info_1", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            with torch.no_grad():
                start_time = time.time()
                # h_t = torch.zeros(
                #     envs.num_envs, 768,
                #     device=self.device,
                # )
                # language_features = torch.zeros(
                #     envs.num_envs, 80, 768,
                #     device=self.device,
                # )
                not_done_masks = torch.zeros(
                envs.num_envs, 1, dtype=torch.uint8, device=self.device
                )
                
                selected_depth_images,selected_rgb_images, batch_angles, batch_distances,global_img = self.policy.net(
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,DINO_processor=DINO_processor,
                    masks = not_done_masks,
                )
            def preprocess_nav_images(rgb_images, image_size=384):
                rgb_images = rgb_images.permute(0, 3, 1, 2).float() / 255.0  #  [4, 3, 224, 224]

                transform = Compose([
                    Resize((image_size, image_size)),  
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                ])
                # process each img
                processed_images = torch.stack([transform(img) for img in rgb_images])
                return processed_images  # [4, 3, 384, 384]
            
            ram_image = preprocess_nav_images(selected_rgb_images).cuda()
            global_image = preprocess_nav_images(global_img).cuda()

            
            res_list = []
            for img in torch.unbind(ram_image, dim=0):  
                res_single = inference_ram(img.unsqueeze(0), ram_model)  
                res_list.append(res_single[0])
            

            if step == 0:
                cand_inputs = self.prompt_manager.make_action_prompt_backtrackv2(agent_state_i['position'], agent_state_i['ori'],batch_angles, batch_distances, selected_rgb_images,res_list=res_list,fuse_close_node=False,t=step,env_action=None)
            else:
                cand_inputs = self.prompt_manager.make_action_prompt_backtrackv2(agent_state_i['position'], agent_state_i['ori'],batch_angles, batch_distances, selected_rgb_images,res_list=res_list,fuse_close_node=False,t=step,env_action=env_actions[0])

            nav_input = self.prompt_manager.make_r2r_json_prompts(instruction_data['text'], cand_inputs=cand_inputs, t=step, res_list=res_list)

            last_part = config.TENSORBOARD_DIR.split('/')[-1]
            environment_prompts = nav_input["prompts"][0]
            print('-------------------- Environment Prompts --------------------')
            print(environment_prompts)

            nav_output, tokens = gpt_infer_back_track(nav_input["task_description"], environment_prompts, selected_rgb_images, model="gpt-4o-2024-08-06", max_tokens=600, response_format={"type": "json_object"},step=step,\
                num_epi=num_epi,semantic_map=None,exp_name=last_part,img_id=cand_inputs['cand_index'][0])


            if nav_output == None:
                print("none")
            json_output = json.loads(nav_output)
            a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], step)
            self.prompt_manager.parse_json_planning(json_output)
            print('-------------------- Output --------------------')
            print(nav_output)

            traj[0]['a_t'][step] = a_t[0]
            if step >=2:
                a_t_stop = (a_t[0] == -1)
            else:
                a_t_stop = 0
            cpu_a_t = []
            if a_t_stop or ended:
                cpu_a_t.append(-1)
                just_ended = True
            else:
                cpu_a_t.append(a_t[0])

            env_actions = self.make_equiv_action(cpu_a_t, self.prompt_manager.candidates_dict, traj)
            outputs = envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            for j, ob in enumerate(observations):
                if env_actions[j]['action']['action'] == 0:
                    continue
                else:
                    envs.call_at(j, 
                        'change_current_path',
                        {'new_path': ob.pop('positions'),
                        'collisions': ob.pop('collisions')}
                    )

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8, device=self.device)

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                # if a_t_stop == False:
                if not dones[i]:
                    self.prompt_manager.make_history(a_t, nav_input, step)
                    # self.prompt_manager.make_history_v2(a_t, nav_input, step,unique_objects)
                    # self.prompt_manager.curr_action = 
                    continue
                # else:
                traj = [{
                'path': [],
                'details': {},
                'a_t': {},
                }]

                # Initialization the tracking state
                ended = np.array([False] * envs.num_envs)
                just_ended = np.array([False] * envs.num_envs)

                self.prompt_manager.history = ['']
                self.prompt_manager.panoramic_history = ['']
                self.prompt_manager.nodes_list = [[]]
                self.prompt_manager.node_imgs = [[]]
                self.prompt_manager.graph = [{}]
                self.prompt_manager.trajectory = [[]]
                self.prompt_manager.viewpoint_coords_dict = {}
                self.prompt_manager.planning = [["Navigation has just started, with no planning yet."]]
                self.prompt_manager.last_action = ""
                self.episode_api_time = 0
                info = infos[i]
                metric = {}
                metric['steps_taken'] = info['steps_taken']
                ep_id = str(envs.current_episodes()[i].episode_id)
                gt_path = np.array(self.gt_data[ep_id]['locations']).astype(np.float)
                if 'current_path' in envs.current_episodes()[i].info.keys():
                    positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(np.float)
                    collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                    assert collisions_.shape[0] == positions_.shape[0] - 1
                else:
                    positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(np.float)
                distance = np.array(info['position']['distance']).astype(np.float)
                metric['distance_to_goal'] = distance[-1]
                metric['success'] = 1. if distance[-1] <= 3. and env_actions[i]['action']['action'] == 0 else 0.
                print("success",metric['success'])
                metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                metric['collisions'] = collisions_.mean()
                gt_length = distance[0]
                metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])
                print(metric['spl'])

                act_con_path = positions_
                gt_con_path = np.array(gt_path).astype(np.float)
                dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                metric['ndtw'] = nDTW
                stats_episodes[current_episodes[i].episode_id] = metric

                observations[i] = envs.reset_at(i)[0]
                # if 'CMA' in self.config.MODEL.policy_name:
                #     rnn_states[i] *= 0.
                # elif 'VLNBERT' in self.config.MODEL.policy_name:
                #     h_t[i] *= 0.

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=episodes_to_eval,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "spl": stats_episodes[
                                current_episodes[i].episode_id
                            ]["spl"]
                        },
                        tb_writer=writer,
                    )

                    del stats_episodes[current_episodes[i].episode_id][
                        "top_down_map_vlnce"
                    ]
                    del stats_episodes[current_episodes[i].episode_id][
                        "collisions"
                    ]
                    rgb_frames[i] = []
            instruction_data = observations[0]["instruction"]
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            # if 'VLNBERT' in self.config.MODEL.policy_name:
            #     rnn_states = h_t

            # headings = torch.tensor(headings)
            # (
            #     envs,
            #     rnn_states,
            #     not_done_masks,
            #     headings,  # prev_actions
            #     batch,
            #     rgb_frames,
            # ) = self._pause_envs(
            #     envs_to_pause,
            #     envs,
            #     rnn_states,
            #     not_done_masks,
            #     headings,
            #     batch,
            #     rgb_frames,
            # )
            # headings = headings.tolist()
            # if 'VLNBERT' in self.config.MODEL.policy_name:
            #     h_t = rnn_states

        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)



    def collect_val_traj(self):
        from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            if "{role}" in self.config.IL.RECOLLECT_TRAINER.gt_file:
                gt_data = {}
                for role in RxRVLNCEDatasetV1.annotation_roles:
                    if (
                        ALL_ROLES_MASK not in self.config.TASK_CONFIG.DATASET.ROLES
                        and role not in self.config.TASK_CONFIG.DATASET.ROLES
                    ):
                        continue

                    with gzip.open(
                        self.config.IL.RECOLLECT_TRAINER.gt_file.format(
                            split=split, role=role
                        ),
                        "rt",
                    ) as f:
                        gt_data.update(json.load(f))
            else:
                with gzip.open(
                    self.config.IL.RECOLLECT_TRAINER.gt_path.format(
                        split=split)
                ) as f:
                    gt_data = json.load(f)
        else:
            with gzip.open(
                self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                    split=split)
            ) as f:
                gt_data = json.load(f)

        self.gt_data = gt_data

        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]

        return trajectories




    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        # split = self.config.TASK_CONFIG.DATASET.SPLIT
        # self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        # self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        #
        # if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
        self.traj = self.collect_val_traj()
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            if os.path.isfile(self.config.EVAL_CKPT_PATH_DIR):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(
                    self.config.EVAL_CKPT_PATH_DIR
                )
                if proposed_index is not None:
                    ckpt_idx = proposed_index
                else:
                    ckpt_idx = 0
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                
                self._eval_checkpoint(
                    checkpoint_path=None,
                    writer=writer,
                    checkpoint_index=prev_ckpt_ind,
                )


