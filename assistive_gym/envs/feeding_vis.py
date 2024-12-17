import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

class FeedingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(FeedingEnv, self).__init__(robot=robot, human=human, task='feeding', obs_robot_len=(18 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(19 + len(human.controllable_joint_indices)))
        self.given_pref = False

    # def human_preferences(self, estimated_weights, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        
    #     # Slow end effector velocities
    #     reward_velocity = -end_effector_velocity

    #     # < 10 N force at target
    #     reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

    #     # --- Scooping, Feeding, Drinking ---
    #     if self.task in ['feeding', 'drinking']:
    #         # Penalty when robot's body applies force onto a person
    #         reward_force_nontarget = -total_force_on_human
    #     # Penalty when robot spills food on the person
    #     reward_food_hit_human = food_hit_human_reward
    #     # Human prefers food entering mouth at low velocities
    #     reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)


    #     return self.C_v*reward_velocity, self.C_f*reward_force_nontarget, self.C_hf*reward_high_target_forces, self.C_fd*reward_food_hit_human, self.C_fdv*reward_food_velocities,
    #             estimated_weights.C_v*reward_velocity, estimated_weights.C_f*reward_force_nontarget, estimated_weights.C_hf*reward_high_target_forces, estimated_weights.C_fd*reward_food_hit_human, estimated_weights.C_fdv*reward_food_velocities

    #改动：此处不再需要传入estimated weights
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


        return reward_velocity, reward_force_nontarget, reward_high_target_forces, reward_food_hit_human, reward_food_velocities

    #新代码
    #改动：此处不再需要传入更多参数
    def step(self, action, args, success_rate):

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
        #改动：这里不需要再传入estimated_weights
        reward_velocity, reward_force_nontarget, reward_high_target_forces, reward_food_hit_human, reward_food_velocities = \
            self.human_preferences(end_effector_velocity=end_effector_velocity, 
                                   total_force_on_human=self.total_force_on_human, 
                                   tool_force_at_target=self.spoon_force_on_human, 
                                   food_hit_human_reward=food_hit_human_reward, 
                                   food_mouth_velocities=food_mouth_velocities)
        
        #改动：这里使用传入的weight
        prefv_r, preff_r, prefh_r, prefhit_r, preffv_r = args.r_velocity*reward_velocity, args.r_force*reward_force_nontarget, args.r_h_force*reward_high_target_forces, args.r_hit*reward_food_hit_human, args.r_food_v*reward_food_velocities
        prefv, preff, prefh, prefhit, preffv = self.C_v*reward_velocity, self.C_f*reward_force_nontarget, self.C_hf*reward_high_target_forces, self.C_fd*reward_food_hit_human, self.C_fdv*reward_food_velocities
        preferences_score = prefv + preff + prefh + prefhit + preffv
        
        #social 专用 preferences_score_r
        preferences_score_r = prefv_r + preff_r + prefh_r + prefhit_r + preffv_r
        #新代码，这样做是为了确保机器人在学习到成功的策略基础上，才综合学习preference相关内容
        preferences_score_r = np.random.choice([preferences_score_r, 0], p=[success_rate, 1 - success_rate])

        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()

        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - spoon_pos) # Penalize robot for distance between the spoon and human mouth.
        reward_action = -np.linalg.norm(action) # Penalize actions

        human_dist_reward =  self.config('distance_weight')*reward_distance_mouth_target
        human_action_reward = self.config('action_weight')*reward_action
        human_food_reward = self.config('food_reward_weight')*reward_food
        human_pref_reward = preferences_score
        robot_dist_reward =  self.config('distance_weight')*reward_distance_mouth_target
        robot_action_reward = self.config('action_weight')*reward_action
        robot_food_reward =  self.config('food_reward_weight')*reward_food
        if args.algo == 'PPO':
            if args.social:
                if args.dempref:
                    robot_pref_reward = preferences_score_r # baseline改这里
                else:
                    robot_pref_reward = 0
            else:
                if args.given_pref:
                    robot_pref_reward = preferences_score
                else:
                    robot_pref_reward = 0
        else:
            robot_pref_reward = 0
        
        # to args, split vars: reward_distance_mouth_target, preferences_score
        reward_human = human_dist_reward + human_action_reward + human_food_reward + human_pref_reward
        reward_robot = robot_dist_reward + robot_action_reward + robot_food_reward + robot_pref_reward
        reward = 0.5 * (reward_human + reward_robot)
    
        if self.gui and reward_food != 0:
            print('Task success:', self.task_success, 'Food reward:', reward_food)

        # info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'particles' : self.task_success, 'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            #新代码，这里多返回了一些pref_array
            # Co-optimization with both human and robot controllable
            return obs, {'robot': reward_robot, 'human': reward_human, '__all__':reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}, {'human_dist_reward': human_dist_reward, 'human_action_reward': human_action_reward, 'human_food_reward': human_food_reward, 'human_pref_reward' : human_pref_reward,
            'robot_dist_reward': robot_dist_reward, 'robot_action_reward': robot_action_reward, 'robot_food_reward': robot_food_reward, 'robot_pref_reward' : robot_pref_reward, 'vel': prefv, 'force': preff, 'h_force' : prefh, 'hit' : prefhit, 'food_v' : preffv}
            #改动：这里的返回不在传pref_entries

    # def step(self, action):
    #     if self.human.controllable:
    #         action = np.concatenate([action['robot'], action['human']])
    #     self.take_step(action)

    #     obs = self._get_obs()

    #     reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()

    #     # Get human preferences
    #     end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.right_end_effector))
    #     preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.spoon_force_on_human, food_hit_human_reward=food_hit_human_reward, food_mouth_velocities=food_mouth_velocities)

    #     spoon_pos, spoon_orient = self.tool.get_base_pos_orient()

    #     reward_distance_mouth_target = -np.linalg.norm(self.target_pos - spoon_pos) # Penalize robot for distance between the spoon and human mouth.
    #     reward_action = -np.linalg.norm(action) # Penalize actions

    #     reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score
    #     # print(self.config('distance_weight')*reward_distance_mouth_target, self.config('action_weight')*reward_action, self.config('food_reward_weight')*reward_food, preferences_score)

    #     if self.gui and reward_food != 0:
    #         print('Task success:', self.task_success, 'Food reward:', reward_food)

    #     info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
    #     done = self.iteration >= 200

    #     if not self.human.controllable:
    #         return obs, reward, done, info
    #     else:
    #         # Co-optimization with both human and robot controllable
    #         return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}

    def get_total_force(self):
        robot_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        spoon_force_on_human = np.sum(self.tool.get_contact_points(self.human)[-1])
        return robot_force_on_human, spoon_force_on_human

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

    def _get_obs(self, agent=None):
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        spoon_pos_real, spoon_orient_real = self.robot.convert_to_realworld(spoon_pos, spoon_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        head_pos_real, head_orient_real = self.robot.convert_to_realworld(head_pos, head_orient)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.robot_force_on_human, self.spoon_force_on_human = self.get_total_force()
        self.total_force_on_human = self.robot_force_on_human + self.spoon_force_on_human
        robot_obs = np.concatenate([spoon_pos_real, spoon_orient_real, spoon_pos_real - target_pos_real, robot_joint_angles, head_pos_real, head_orient_real, [self.spoon_force_on_human]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            spoon_pos_human, spoon_orient_human = self.human.convert_to_realworld(spoon_pos, spoon_orient)
            head_pos_human, head_orient_human = self.human.convert_to_realworld(head_pos, head_orient)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([spoon_pos_human, spoon_orient_human, spoon_pos_human - target_pos_human, human_joint_angles, head_pos_human, head_orient_human, [self.robot_force_on_human, self.spoon_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(FeedingEnv, self).reset()
        self.build_assistive_env('wheelchair')
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # Update robot and human motor gains
        self.robot.motor_gains = self.human.motor_gains = 0.025

        joints_positions = [(self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        joints_positions += [(self.human.j_head_x, self.np_random.uniform(-30, 30)), (self.human.j_head_y, self.np_random.uniform(-30, 30)), (self.human.j_head_z, self.np_random.uniform(-30, 30))]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)

        # Create a table
        self.table = Furniture()
        self.table.init('table', self.directory, self.id, self.np_random)

        self.generate_target()

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=True, mesh_scale=[0.08]*3)

        target_ee_pos = np.array([-0.15, -0.65, 1.15]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient), (self.target_pos, None)], [(self.target_pos, target_ee_orient)], arm='right', tools=[self.tool], collision_objects=[self.human, self.table, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.right_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        # Place a bowl on a table
        self.bowl = Furniture()
        self.bowl.init('bowl', self.directory, self.id, self.np_random)

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=10, physicsClientId=self.id)

        # Generate food
        spoon_pos, spoon_orient = self.tool.get_base_pos_orient()
        food_radius = 0.005
        food_mass = 0.001
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(np.array([i*2*food_radius-0.005, j*2*food_radius, k*2*food_radius+0.01]) + spoon_pos)
        self.foods = self.create_spheres(radius=food_radius, mass=food_mass, batch_positions=batch_positions, visual=False, collision=True)
        colors = [[60./256., 186./256., 84./256., 1], [244./256., 194./256., 13./256., 1],
                  [219./256., 50./256., 54./256., 1], [72./256., 133./256., 237./256., 1]]
        for i, f in enumerate(self.foods):
            p.changeVisualShape(f.body, -1, rgbaColor=colors[i%len(colors)], physicsClientId=self.id)
        self.total_food_count = len(self.foods)
        self.foods_active = [f for f in self.foods]

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Drop food in the spoon
        for _ in range(25):
            p.stepSimulation(physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.human.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.update_targets()

    def update_targets(self):
        head_pos, head_orient = self.human.get_pos_orient(self.human.head)
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])
