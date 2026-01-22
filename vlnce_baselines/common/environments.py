from typing import Any, Dict, Optional, Tuple, List, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import math

@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []
        self.plan_frames = []
    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
    def get_agent_info_1(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "ori":agent_state.rotation,
            "stop": self._env.task.is_stop_called,
        }
    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)
        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))
        return obs

    def observations_by_angles(self, angle_list: List[float]):
        r'''for getting observations from desired angles
        requires rad, positive represents anticlockwise'''
        obs = []
        sim = self._env.sim
        init_state = sim.get_agent_state()
        prev_angle = 0
        left_action = HabitatSimActions.TURN_LEFT
        init_amount = sim.get_agent(0).agent_config.action_space[left_action].actuation.amount # turn left
        for angle in angle_list:
            sim.get_agent(0).agent_config.action_space[left_action].actuation.amount = (angle-prev_angle)*180/np.pi
            obs.append(sim.step(left_action))
            prev_angle = angle
        sim.set_agent_state(init_state.position, init_state.rotation)
        sim.get_agent(0).agent_config.action_space[left_action].actuation.amount = init_amount
        return obs

    def current_dist_to_goal(self):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance

    def change_current_path(self, new_path: Any, collisions: Any):
        '''just for recording current path in high to low'''
        if 'current_path' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['current_path'] = [np.array(self._env.current_episode.start_position)]
        self._env.current_episode.info['current_path'] += new_path
        if 'collisions' not in self._env.current_episode.info.keys():
            self._env.current_episode.info['collisions'] = []
        self._env.current_episode.info['collisions'] += collisions
        
    
    def get_pos_ori(self):
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
        return (pos, ori)

    def step_video(self, action, vis_info, *args, **kwargs):
        act = action['act']

        if act == 4: # high to low
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to front node
            if action['back_path'] is None:
                self.teleport(action['front_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)

            # 2. forward to ghost node
            self.single_step_control(action['ghost_pos'], action['tryout'], vis_info,action['action_args'])
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)

        elif act == 0:   # stop
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to stop node
            if action['back_path'] is None:
                self.teleport(action['stop_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)

            # 2. stop
            observations = self._env.step(act)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                    )
                )
                self.get_plan_frame(vis_info)

        else:
            raise NotImplementedError                

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        if self.video_option and done:
            # if 0 < info["spl"] <= 0.6:  #TODO backtrack
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                # metrics={"SPL": round(info["spl"], 3)},
                tb_writer=None,
                fps=8,
            )
            # for pano visualization
            # metrics={
            #             # "sr": round(info["success"], 3), 
            #             "spl": round(info["spl"], 3),
            #             # "ndtw": round(info["ndtw"], 3),
            #             # "sdtw": round(info["sdtw"], 3),
            #         }
            # metric_strs = []
            # for k, v in metrics.items():
            #     metric_strs.append(f"{k}{v:.2f}")
            # episode_id=self._env.current_episode.episode_id
            # scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
            # tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
            # tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
            # tmp_fn = os.path.join(self.video_dir, tmp_name)
            # tmp = np.concatenate(self.plan_frames, axis=0)
            # cv2.imwrite(tmp_fn, tmp)
            # self.plan_frames = []

        return observations, reward, done, info

    def wrap_act(self, act, vis_info):
        ''' wrap action, get obs if video_option '''
        observations = None
        if self.video_option:
            observations = self._env.step(act)
            info = self.get_info(observations)
            self.video_frames.append(
                navigator_video_frame(
                    observations,
                    info,
                    vis_info,
                )
            )
        else:
            self._env.sim.step_without_obs(act)
            self._env._task.measurements.update_measures(
                episode=self._env.current_episode, action=act, task=self._env.task 
            )
        return observations

    def turn(self, ang, vis_info):    
        ''' angle: 0 ~ 360 degree '''
        act_l = HabitatSimActions.TURN_LEFT
        act_r = HabitatSimActions.TURN_RIGHT
        uni_l = self._env.sim.get_agent(0).agent_config.action_space[act_l].actuation.amount
        ang_degree = math.degrees(ang)
        ang_degree = round(ang_degree / uni_l) * uni_l
        observations = None

        if 180 < ang_degree <= 360:
            ang_degree -= 360
        if ang_degree >=0:
            turns = [act_l] * ( ang_degree // uni_l)
        else:
            turns = [act_r] * (-ang_degree // uni_l)
        
        for turn in turns:
            observations = self.wrap_act(turn, vis_info)
        return observations

    def single_step_control(self, pos, tryout, vis_info,action_args):
        act_f = HabitatSimActions.MOVE_FORWARD
        uni_f = self._env.sim.get_agent(0).agent_config.action_space[act_f].actuation.amount
        agent_state = self._env.sim.get_agent_state()
        ang, dis = calculate_vp_rel_pos(agent_state.position, pos, heading_from_quaternion(agent_state.rotation))
        self.turn(ang, vis_info)
        # self.turn(np.pi*2-action_args['angle'], vis_info)

        ksteps = int(dis // uni_f)
        if not tryout:
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
        else:
            cnt = 0 
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
                if self._env.sim.previous_step_collided:
                    break
                else:
                    cnt += 1
            # left forward step
            ksteps = ksteps - cnt
            if ksteps > 0:
                try_ang = random.choice([math.radians(90), math.radians(270)]) # left or right randomly
                self.turn(try_ang, vis_info)
                if try_ang == math.radians(90):     # from left to right
                    turn_seqs = [
                        (0, 270),   # 90, turn_left=30, turn_right=330
                        (330, 300), # 60
                        (330, 330), # 30
                        (300, 30),  # -30
                        (330, 60),  # -60
                        (330, 90),  # -90
                    ]
                elif try_ang == math.radians(270):  # from right to left
                    turn_seqs = [
                        (0, 90),   # -90
                        (30, 60),  # -60
                        (30, 30),  # -30
                        (60, 330), # 30
                        (30, 300), # 60
                        (30, 270), # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    # do head_turns
                    self.turn(math.radians(turn_seq[0]), vis_info)
                    prev_position = self._env.sim.get_agent_state().position
                    self.wrap_act(act_f, vis_info)
                    post_posiiton = self._env.sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        # do tail_turns
                        self.turn(math.radians(turn_seq[1]), vis_info)
                        # do left forward actions
                        for _ in range(ksteps):
                            self.wrap_act(act_f, vis_info)
                            if self._env.sim.previous_step_collided:
                                break
                        break

    def teleport(self, pos):
        self._env.sim.set_agent_state(pos, quat_from_heading(0))

    def get_plan_frame(self, vis_info):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        info = self.get_info(observations)

        frame = planner_video_frame(observations, info, vis_info)
        frame = cv2.copyMakeBorder(frame, 6,6,5,5, cv2.BORDER_CONSTANT, value=(255,255,255))
        self.plan_frames.append(frame)

    def multi_step_control(self, path, tryout, vis_info):
        for vp, vp_pos in path: #path[::-1]:
            self.single_step_control(vp_pos, tryout, vis_info)

    def reset(self):
        observations = self._env.reset()
        if self.video_option:
            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, 
                    info,
                )
            ]
        return observations
        
@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }


import torch
import cv2
from copy import deepcopy
from habitat_extensions import maps
from habitat.utils.visualizations import maps as habitat_maps
obs_trans_to_eq = baseline_registry.get_obs_transformer("CubeMap2Equirect")
UUIDS_EQ = ['rgbback', 'rgbdown', 'rgbfront', 'rgbright', 'rgbleft', 'rgbup']
CUBE2EQ = obs_trans_to_eq(UUIDS_EQ, (224,448))

def navigator_video_frame(
    observations,
    info,
    vis_info=None,
    map_k="top_down_map_vlnce",
):
    # rgb = {k: v for k, v in observations.items() if k.startswith("rgb")}
    # rgb["rgb_0"] = rgb["rgb"]
    # del rgb["rgb"]
    # rgb = [
    #     f[1]
    #     for f in sorted(rgb.items(), key=lambda f: int(f[0].split("_")[1]))
    # ]

    # rgb = [
    #     add_id_on_img(rgb[i][:, 80 : (rgb[i].shape[1] - 80), :], str(i))
    #     for i in range(len(rgb))
    # ][::-1]
    # rgb = np.concatenate(rgb[6:] + rgb[:6], axis=1).astype(np.uint8)
    # new_height = int((frame_width / rgb.shape[1]) * rgb.shape[0])
    # rgb = cv2.resize(
    #     rgb,
    #     (frame_width, new_height),
    #     interpolation=cv2.INTER_CUBIC,
    # )
    cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
    cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
    eq = CUBE2EQ(cube)
    rgb = eq['rgbback'][0].numpy().copy()

    top_down_map = colorize_draw_agent_and_fit_to_height(
        info[map_k], 
        rgb.shape[0], 
        vis_info,
    )
    frame = np.concatenate([rgb, top_down_map], axis=1)
    frame = append_text_to_image(frame, observations["instruction"]["text"])

    return frame

def colorize_draw_agent_and_fit_to_height(
    info: Dict[str, Any], 
    output_height: int,
    vis_info: Dict,
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    # def _ang_dis_to_coord(
    #     ang, dis, current_position, current_heading
    # ):
    #     phi = (heading_from_quaternion(current_heading) + ang) % (2 * np.pi)
    #     x = current_position[0] - dis * np.sin(phi)
    #     z = current_position[-1] - dis * np.cos(phi)
    #     return [x, z]

    top_down_map = deepcopy(info["map"])

    if vis_info is not None:
        if 'nodes' in vis_info:
            for p in vis_info['nodes']:
                maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.NODE)
        if 'ghosts' in vis_info:
            for p in vis_info['ghosts']:
                maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.GHOST)
        # if 'teacher_ghost' in vis_info and vis_info['teacher_ghost'] is not None:
        #     maps.draw_waypoint(top_down_map, vis_info['teacher_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.TEACHER_GHOST)
        if 'predict_ghost' in vis_info:
            maps.draw_waypoint(top_down_map, vis_info['predict_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.PREDICT_GHOST)
        
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["fog_of_war_mask"]
    )
    map_agent_pos = info["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map

import textwrap
def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final

from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from numpy import ndarray
from habitat.utils.visualizations.utils import images_to_video

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[ndarray],
    episode_id: Union[str, int],
    scene_id: str,
    checkpoint_idx: int,
    # metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
):
    """Generate video according to specified information. Using a custom
    verion instead of Habitat's that passes FPS to video maker.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    """
    if len(images) < 1:
        return

    # metric_strs = []
    # for k, v in metrics.items():
    #     metric_strs.append(f"{k}{v:.2f}")

    video_name = f"{scene_id}-{episode_id}"
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )

def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    # xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    # to (0, 2pi)
    while heading < 0:
        heading += 2*np.pi
    heading = heading % (2*np.pi)

    return heading, xz_dist

import quaternion

def heading_from_quaternion(quat: quaternion.quaternion) -> float:
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    heading_vector = quaternion_rotate_vector(
        quat.inverse(), np.array([0, 0, -1])
    )
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)

from scipy.spatial.transform import Rotation as R
def quat_from_heading(heading, elevation=0):
    array_h = np.array([0, heading, 0])
    array_e = np.array([0, elevation, 0])
    rotvec_h = R.from_rotvec(array_h)
    rotvec_e = R.from_rotvec(array_e)
    quat = (rotvec_h * rotvec_e).as_quat()
    return quat

def planner_video_frame(
    observations,
    info,
    vis_info=None,
    map_k="top_down_map_vlnce",
):
    cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
    cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
    eq = CUBE2EQ(cube)
    rgb = eq['rgbback'][0].numpy().copy()

    top_down_map = colorize_draw_agent_and_fit_to_height(
        info[map_k], 
        rgb.shape[0], 
        vis_info,
    )
    frame = np.concatenate([rgb, top_down_map], axis=1)
    frame = cv2.copyMakeBorder(frame, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,0,0))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # frame = append_text_to_image(frame, observations["instruction"]["text"])

    return frame