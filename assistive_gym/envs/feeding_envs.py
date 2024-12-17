from .feeding import FeedingEnv
from .feeding_mesh import FeedingMeshEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human, human_mesh
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.human_mesh import HumanMesh
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

import numpy as np

robot_arm = 'right'
human_controllable_joint_indices = human.head_joints
class FeedingPR2Env(FeedingEnv):
    def __init__(self):
        super(FeedingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingBaxterEnv(FeedingEnv):
    def __init__(self):
        super(FeedingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingSawyerEnv(FeedingEnv):
    def __init__(self):
        super(FeedingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingJacoEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingStretchEnv(FeedingEnv):
    def __init__(self):
        super(FeedingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingPandaEnv(FeedingEnv):
    def __init__(self):
        super(FeedingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class FeedingPR2HumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingPR2Human-v1', lambda config: FeedingPR2HumanEnv())

class FeedingBaxterHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingBaxterHuman-v1', lambda config: FeedingBaxterHumanEnv())

class FeedingSawyerHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingSawyerHuman-v1', lambda config: FeedingSawyerHumanEnv())

class FeedingSawyerHumanEnv2(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingSawyerHumanEnv2, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
    
    def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        # --- Scooping, Feeding, Drinking ---
        if self.task in ['feeding', 'drinking']:
            # Penalty when robot's body applies force onto a person
            reward_force_nontarget = -total_force_on_human
        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)


        return self.C_v*reward_velocity + self.C_f*reward_force_nontarget + self.C_hf*reward_high_target_forces + self.C_fd*reward_food_hit_human + self.C_fdv*reward_food_velocities

    def step(self, action, args):

        '''
        modifications:
        1.split human reward and robot reward.
            Human preferences:
                we need food to be sent fast but when delivering, smoothly. we don want spit, don want force.

                i.  food to be delivered fast(penalize for every step). reward_distance_mouth_target
                ii. food not spitted on body. food_hit_human_reward
                iii.food entering mouth slowly. food_mouth_velocities
                iv. getting food. reward_food
                v.  no or low force on human. human_preferences

            Robot preferences:
                we need to send food to human, not spitting it.

                i.  food not spitted on body.
                ii. food entering mouth slowly.
                iii.getting food.
                iv. no or low force on human.
                v.  penalize action.
            System preference:

        2.
        '''

        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.spoon_force_on_human, food_hit_human_reward=food_hit_human_reward, food_mouth_velocities=food_mouth_velocities)

        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()

        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - spoon_pos) # Penalize robot for distance between the spoon and human mouth.
        reward_action = -np.linalg.norm(action) # Penalize actions


        human_dist_reward = args.dist_weight* self.config('distance_weight')*reward_distance_mouth_target
        human_action_reward = args.act_weight* self.config('action_weight')*reward_action
        human_food_reward = args.food_weight* self.config('food_reward_weight')*reward_food
        human_pref_reward = args.pref_weight* preferences_score
        robot_dist_reward = (2.0-args.dist_weight)* self.config('distance_weight')*reward_distance_mouth_target
        robot_action_reward = (2.0-args.act_weight)* self.config('action_weight')*reward_action
        robot_food_reward = (2.0-args.food_weight)* self.config('food_reward_weight')*reward_food
        robot_pref_reward = (2.0-args.pref_weight)* preferences_score
        # to args, split vars: reward_distance_mouth_target, preferences_score
        reward_human = human_dist_reward + human_action_reward + human_food_reward + human_pref_reward
        reward_robot = robot_dist_reward + robot_action_reward + robot_food_reward + robot_pref_reward
        reward = 0.5 * (reward_human + reward_robot)
        # reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score
        
        # print(self.config('distance_weight')*reward_distance_mouth_target, self.config('action_weight')*reward_action, self.config('food_reward_weight')*reward_food, preferences_score)

        if self.gui and reward_food != 0:
            print('Task success:', self.task_success, 'Food reward:', reward_food)

        info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward_robot, 'human': reward_human, '__all__':reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}, {'human_dist_reward': human_dist_reward, 'human_action_reward': human_action_reward, 'human_food_reward': human_food_reward, 'human_pref_reward' : human_pref_reward,
            'robot_dist_reward': robot_dist_reward, 'robot_action_reward': robot_action_reward, 'robot_food_reward': robot_food_reward, 'robot_pref_reward' : robot_pref_reward}

    def get_food_rewards(self):
        # Check all food particles to see if they have left the spoon or entered the person's mouth
        # Give the robot a reward or penalty depending on food particle status
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        foods_active_to_remove = []
        for f in self.foods:
            food_pos, food_orient = f.get_base_pos_orient()
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.03:
                # Food is close to the person's mouth. Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(f.get_velocity(f.base))
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                foods_active_to_remove.append(f)
                f.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                continue
            elif len(f.get_closest_points(self.tool, distance=0.1)[-1]) == 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
        for f in self.foods_active:
            if len(f.get_contact_points(self.human)[-1]) > 0:
                # Record that this food particle just hit the person, so that we can penalize the robot
                food_hit_human_reward -= 1
                foods_active_to_remove.append(f)
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        self.foods_active = [f for f in self.foods_active if f not in foods_active_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward


register_env('assistive_gym:FeedingSawyerHuman-v2', lambda config: FeedingSawyerHumanEnv2())

class FeedingJacoHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingJacoHuman-v1', lambda config: FeedingJacoHumanEnv())

class FeedingStretchHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingStretchHuman-v1', lambda config: FeedingStretchHumanEnv())

class FeedingPandaHumanEnv(FeedingEnv, MultiAgentEnv):
    def __init__(self):
        super(FeedingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:FeedingPandaHuman-v1', lambda config: FeedingPandaHumanEnv())

class FeedingPR2MeshEnv(FeedingMeshEnv):
    def __init__(self):
        super(FeedingPR2MeshEnv, self).__init__(robot=PR2(robot_arm), human=HumanMesh())

class FeedingBaxterMeshEnv(FeedingMeshEnv):
    def __init__(self):
        super(FeedingBaxterMeshEnv, self).__init__(robot=Baxter(robot_arm), human=HumanMesh())

class FeedingSawyerMeshEnv(FeedingMeshEnv):
    def __init__(self):
        super(FeedingSawyerMeshEnv, self).__init__(robot=Sawyer(robot_arm), human=HumanMesh())

class FeedingJacoMeshEnv(FeedingMeshEnv):
    def __init__(self):
        super(FeedingJacoMeshEnv, self).__init__(robot=Jaco(robot_arm), human=HumanMesh())

class FeedingStretchMeshEnv(FeedingMeshEnv):
    def __init__(self):
        super(FeedingStretchMeshEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=HumanMesh())

class FeedingPandaMeshEnv(FeedingMeshEnv):
    def __init__(self):
        super(FeedingPandaMeshEnv, self).__init__(robot=Panda(robot_arm), human=HumanMesh())

