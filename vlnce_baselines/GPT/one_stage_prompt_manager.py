import re
import math
import uuid
import numpy as np
import logging
from functools import partial
from habitat.utils.geometry_utils import quaternion_rotate_vector
import numpy as np
logging.basicConfig(level=logging.INFO)

class OneStagePromptManager(object):
    def __init__(self, envs):
        self.envs = envs  
        self.history = ['']
        self.panoramic_history = ['']
        self.nodes_list = [[]]
        self.node_imgs = [[]]
        self.graph = [{}]
        self.trajectory = [[]]
        self.planning = [["Navigation has just started, with no planning yet."]]


    def get_action_concept_backtrack(self, rel_heading):

        deg = math.degrees(rel_heading)  # rad to deg

        if 0 <= deg < 30 or deg >= 330:
            return 'go forward'
        elif 30 <= deg < 90:
            return 'turn slight left'
        elif 90 <= deg < 150:
            return 'turn sharp left'
        elif 150 <= deg <= 210:
            return 'turn around'
        elif 210 < deg < 270:
            return 'turn sharp right'
        elif 270 <= deg < 330:
            return 'turn slight right'
        else:
            
            assert False, f"Unexpected angle deg={deg:.2f}, out of covered ranges!"


    def make_action_prompt_backtrackv2(self, agent_state, agent_ori,candidate_angles, candidate_distances, selected_rgb_images,res_list,fuse_close_node=True,t=0,env_action=None):
        if self.backtrack == True:
            last_backtrack = True
            self.backtrack = False
        else:
            last_backtrack = False
        nodes_list, graph, trajectory, node_imgs = self.nodes_list, self.graph, self.trajectory, self.node_imgs
        # node_imgs[0] = []
        if not hasattr(self, 'viewpoint_coords_dict'):
            self.viewpoint_coords_dict = {}
        last_distance = 0
        if env_action != None:
            last_distance = env_action['action']['action_args']['distance']
            last_angle = env_action['action']['action_args']['angle']
        batch_view_lens, batch_cand_vpids = [], []
        batch_cand_index = []
        batch_action_prompts = []

        cand_vpids, cand_index, action_prompts = [], [], []
        candidates_dict = {}

        viewpoint = str(uuid.uuid4())
        # self.viewpoint_coords_dict[viewpoint] = (current_x, current_y, current_z)

        if len(candidate_angles) != len(candidate_distances):
            raise ValueError("candidate_angles and candidate_distances must be of the same length.")
        # print(viewpoint)
        for idx, (angle, distance) in enumerate(zip(candidate_angles[0], candidate_distances[0])):
            waypoint_id = str(uuid.uuid4())
            candidates_dict[idx] = {
                'angle': angle,
                'distance': distance,
                'waypoint': (waypoint_id),#'waypoint': (waypoint_id, waypoint_coords),
                'rgb_image': None
            }
            print(candidates_dict[idx]['waypoint'])
            # self.viewpoint_coords_dict[waypoint_id] = waypoint_coords
        if last_backtrack != True and last_distance!=0: # last distance =0 for first step
            waypoint_id = str(uuid.uuid4())
            candidates_dict[idx+1] = {
                'angle': np.pi,
                'distance': last_distance,
                'waypoint': (waypoint_id),#'waypoint': (waypoint_id, waypoint_coords),
                'rgb_image': None
            }
            print(candidates_dict[idx+1]['waypoint'])
            # self.viewpoint_coords_dict[waypoint_id] = waypoint_coords
            
        if viewpoint not in nodes_list[0]:
            nodes_list[0].append(viewpoint) # update self.nodes_list
            node_imgs[0].append(None)

        trajectory[0].append(viewpoint)

        for j, cc in candidates_dict.items():
            waypoint_id = cc['waypoint']
            cand_vpids.append(waypoint_id)
            cand_index.append(j)
            direction = self.get_action_concept_backtrack(cc['angle'])

            nodes_list[0].append(waypoint_id)
            node_imgs[0].append(None)
            node_index = nodes_list[0].index(waypoint_id)
            cand_index[j] = node_index

            if last_backtrack != True and j == len(candidates_dict.items())-1 and t!=0:
                action_text =f"Move back to last position in an opposite direction"
            else:
                action_text = direction + f" to Place {node_index} which is corresponding to Image {node_index}, and this image contains objects such as {res_list[j]}."
            action_prompts.append(action_text)
        
        batch_cand_index.append(cand_index)
        batch_cand_vpids.append(cand_vpids)
        batch_action_prompts.append(action_prompts)

        if viewpoint not in graph[0].keys() and viewpoint in nodes_list[0]:
            graph[0][viewpoint] = cand_vpids

        self.candidates_dict = candidates_dict

        return {#
            'cand_vpids': batch_cand_vpids,
            'cand_index': batch_cand_index,
            'action_prompts': batch_action_prompts,
        }
    def make_action_options(self, cand_inputs, t,res_list):

        
        action_options_batch = []  # complete action options
        only_options_batch = []  # only option labels
        batch_action_prompts = cand_inputs["action_prompts"]
        batch_size = len(batch_action_prompts)

        for i in range(batch_size):
            action_prompts = batch_action_prompts[i]
            if t >= 2:
                action_prompts = ['stop'] + action_prompts

            full_action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))]
            only_options = [chr(j + 65) for j in range(len(action_prompts))]
            action_options_batch.append(full_action_options)
            only_options_batch.append(only_options)

        return action_options_batch, only_options_batch
    def make_action_options_backtrack(self, cand_inputs, t,res_list):
            
            action_options_batch = []  # complete action options
            only_options_batch = []  # only option labels
            batch_action_prompts = cand_inputs["action_prompts"]
            batch_size = len(batch_action_prompts)

            for i in range(batch_size):
                action_prompts = batch_action_prompts[i]
                if t >= 2:
                    action_prompts = ['stop'] + action_prompts

                full_action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))] # ABCD with description
                only_options = [chr(j + 65) for j in range(len(action_prompts))]
                action_options_batch.append(full_action_options)
                only_options_batch.append(only_options)

            return action_options_batch, only_options_batch
        #return action_options, only_options

    def make_history(self, a_t, nav_input, t):
        # if t < 2:
        a_t[0] +=1
        batch_size = len(a_t)
        for i in range(batch_size):
            if  nav_input["only_actions"][i][0] == 'stop':
                aaa = 1
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            self.last_action = last_action
            if t == 0:
                self.history[i] += f"""step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""
            # if last_action== "Move back to last position in an opposite direction":
            if last_action== "Move back to last position in an opposite direction":
                self.backtrack = True
    def make_history_v2(self, a_t, nav_input, t,unique_objects):
        # if t < 2:
        a_t[0] +=1
        batch_size = len(a_t)
        for i in range(batch_size):
            if  nav_input["only_actions"][i][0] == 'stop':
                aaa = 1
            nav_input["only_actions"][i] = ['stop'] + nav_input["only_actions"][i]
            last_action = nav_input["only_actions"][i][a_t[i]]
            self.curr_action = last_action
            if t == 0:
                self.history[i] += f"""The robot starts at Place 0. From this position, it can see the following objects: {', '.join(unique_objects)}. Step {str(t)}: {last_action}"""
            else:
                self.history[i] += f""", step {str(t)}: {last_action}"""
    


    def make_r2r_json_prompts(self, instruction, cand_inputs, t, res_list,make_video=False):


        background = """You are an embodied robot that navigates in the real world."""  
        background_supp = """You need to explore between some places marked with IDs and ultimately find the destination to stop.""" \
                        + """ At each step, a series of images corresponding to the places you have observed will be provided to you."""

        instr_des = """'Instruction' is a global, step-by-step detailed guidance that describes the correct navigation path. However, some steps may have already been executed. Your goal is to determine which parts of the 'Instruction' have **already been completed** and which remain **to be executed**."""

        history = """'History' represents the places you have already explored along with their corresponding images. It includes both correct movements according to the 'Instruction' and some past mistaken explorations.  
        **You must use 'History' to verify whether an instruction step has already been completed.** Seeing an object in an image does **not** mean you have not yet passed it; instead, confirm whether the object was previously observed **from a past location** before assuming that step remains incomplete."""

        option = """'Action options' are the set of available actions at this step. Each action corresponds to a place, an image, and detected objects. After moving for a while, a 'stop' option will appear. Additionally, you have an option to return to the previous location if you believe you have made a mistake or need to explore an alternative path."""

        pre_planning = """'Previous Planning' records prior multi-step navigation strategies. **Your goal is to refine and update this plan rather than discard it unless a critical mistake has occurred.**"""

        requirement = """For each provided image, **analyze it in conjunction with 'Instruction' and 'History'** to determine:  
        1. What **parts of the instruction have already been executed**?  
        2. What **steps remain to be executed**?  
        3. Whether your **current position is still aligned with the instruction** or if you have deviated.  
        Your reasoning must be based on **actual past movements, not just object visibility in images**."""

        return_policy = """If you **detect a navigation error**, or if your current path does not align well with the 'Instruction', you should consider returning to the last position using the 'Move back to last position in an opposite direction' action.  
        """

        thought = """Your response must be in JSON format with three fields:  
        1. 'Thought': Explain your reasoning by integrating 'Instruction', 'History', 'Previous Planning', and 'Action options'. Clearly state **which steps are completed and which remain**, and ensure your next move continues executing the instruction correctly.  
        2. 'New Planning': Update your multi-step path planning based on 'Previous Planning' and your reasoning in 'Thought'. Do not modify completed steps; only refine future steps.  
        3. 'Action': Choose a single capital letter corresponding to an action from 'Action options'. Example: `"Action": "A"`."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{history}\n{pre_planning}\n{option}\n{return_policy}\n{requirement}\n{thought}"""
        
        init_history = 'The navigation has just begun, with no history.'

        batch_size = 1
        action_options_batch, only_options_batch = self.make_action_options_backtrack(cand_inputs, t=t, res_list=res_list)
        prompt_batch = []
        for i in range(batch_size):
            instruction = instruction

            if t == 0:
                prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""
            else:
                prompt = f"""Instruction: {instruction}\nHistory: {self.history[i]}\nPrevious Planning:\n{self.planning[i][-1]}\nAction options (step {str(t)}): {action_options_batch[i]}"""

            prompt_batch.append(prompt)

        nav_input = {
            "task_description": task_description,
            "prompts": prompt_batch,
            "only_options": only_options_batch, # abcd
            "action_options": action_options_batch, # ABCD + description
            "only_actions": cand_inputs["action_prompts"] #description not include 'stop'
        }

        return nav_input


    def parse_planning(self, nav_output):
        """
        Only supports parsing outputs in the style of GPT-4v.
        Please modify the parsers if the output style is inconsistent.
        """
        batch_size = len(nav_output)
        keyword1 = '\nNew Planning:'
        keyword2 = '\nAction:'
        for i in range(batch_size):
            output = nav_output[i].strip()
            start_index = output.find(keyword1) + len(keyword1)
            end_index = output.find(keyword2)

            if output.find(keyword1) < 0 or start_index < 0 or end_index < 0 or start_index >= end_index:
                planning = "No plans currently."
            else:
                planning = output[start_index:end_index].strip()

            planning = planning.replace('new', 'previous').replace('New', 'Previous')

            self.planning[i].append(planning)

        return planning

    def parse_json_planning(self, json_output):
        try:
            planning = json_output["New Planning"]
        except:
            planning = "No plans currently."

        self.planning[0].append(planning)
        return planning


    def parse_json_action(self, json_output, only_options_batch, t):
        try:
            output = str(json_output["Action"])
            if output in only_options_batch[0]:
                output_index = only_options_batch[0].index(output)
            else:
                output_index = 0

        except:
            output_index = 0

        if t >= 2:
            output_index -= 1  # align extra stop

        output_index_batch = [output_index]
        return output_index_batch
