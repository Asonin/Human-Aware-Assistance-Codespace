import numpy as np
import pybullet as p

from .env import AssistiveEnv

class ScratchItchEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(ScratchItchEnv, self).__init__(robot=robot, human=human, task='scratch_itch', obs_robot_len=(23 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(24 + len(human.controllable_joint_indices)))

    # def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
    #     # Slow end effector velocities
    #     reward_velocity = -end_effector_velocity

    #     # < 10 N force at target
    #     reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

    #     # --- Scratching, Wiping ---
    #     # Any force away from target is low
    #     reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

    #     # --- Scooping, Feeding, Drinking ---
    #     if self.task in ['feeding', 'drinking']:
    #         # Penalty when robot's body applies force onto a person
    #         reward_force_nontarget = -total_force_on_human
    #     # Penalty when robot spills food on the person
    #     reward_food_hit_human = food_hit_human_reward
    #     # Human prefers food entering mouth at low velocities
    #     reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)

    #     # --- Dressing ---
    #     # Penalty when cloth applies force onto a person
    #     reward_dressing_force = -np.sum(np.linalg.norm(dressing_forces, axis=-1))


        # return self.C_v*reward_velocity, self.C_f*reward_force_nontarget, self.C_hf*reward_high_target_forces, self.C_fd*reward_food_hit_human, self.C_fdv*reward_food_velocities

#新代码，这里只返回项，出去之后再处理
    def human_preferences(self, estimated_weights, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        # --- Scratching, Wiping ---
        # Any force away from target is low
        reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)

        return reward_velocity, reward_force_nontarget, reward_high_target_forces, reward_food_hit_human, reward_food_velocities


    def step(self, action, args, estimated_weights, success_rate):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()
        # print(np.array_str(obs, precision=3, suppress_small=True))

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        
        #新代码，单独计算pref
        reward_velocity, reward_force_nontarget, reward_high_target_forces, reward_food_hit_human, reward_food_velocities = \
            self.human_preferences(estimated_weights=estimated_weights, # 传递这个参数
                                   end_effector_velocity=end_effector_velocity, 
                                   total_force_on_human=self.total_force_on_human, 
                                   tool_force_at_target=self.tool_force_at_target)
        
        # prefv, preff, prefh, prefhit, preffv = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_at_target)
        # preferences_score = prefv + preff + prefh + prefhit + preffv
        
        # 这个矩阵是为了给robot学习，所以要进行归一化
        normed_gt = estimated_weights.get_normalized_gt_weights()
        # print(normed_gt)
        v,f,h,hit,fv = normed_gt['C_v']*reward_velocity, normed_gt['C_f']*reward_force_nontarget, normed_gt['C_hf']*reward_high_target_forces, normed_gt['C_fd']*reward_food_hit_human, normed_gt['C_fdv']*reward_food_velocities
        pref_array = [v,f,h,hit,fv] 
        pref_array = np.array(pref_array).reshape(1, 5)

        #这里要用放缩后的值去计算robot的pref reward
        denormed_estimate_w = estimated_weights.get_denormed_weights()
        prefv_r, preff_r, prefh_r, prefhit_r, preffv_r = denormed_estimate_w['C_v']*reward_velocity, denormed_estimate_w['C_f']*reward_force_nontarget, denormed_estimate_w['C_hf']*reward_high_target_forces, denormed_estimate_w['C_fd']*reward_food_hit_human, denormed_estimate_w['C_fdv']*reward_food_velocities
        prefv, preff, prefh, prefhit, preffv = self.C_v*reward_velocity, self.C_f*reward_force_nontarget, self.C_hf*reward_high_target_forces, self.C_fd*reward_food_hit_human, self.C_fdv*reward_food_velocities
        preferences_score = prefv + preff + prefh + prefhit + preffv
        
        #social 专用 preferences_score_r
        preferences_score_r = prefv_r + preff_r + prefh_r + prefhit_r + preffv_r
        #新代码，这样做是为了确保机器人在学习到成功的策略基础上，才综合学习preference相关内容
        preferences_score_r = np.random.choice([preferences_score_r, 0], p=[success_rate, 1 - success_rate])


        tool_pos = self.tool.get_pos_orient(1)[0]
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos) # Penalize distances away from target
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_force_scratch = 0.0 # Reward force near the target
        if self.target_contact_pos is not None and np.linalg.norm(self.target_contact_pos - self.prev_target_contact_pos) > 0.01 and self.tool_force_at_target < 10:
            # Encourage the robot to move around near the target to simulate scratching
            reward_force_scratch = 5
            self.prev_target_contact_pos = self.target_contact_pos
            self.task_success += 1

        human_dist_reward =  self.config('distance_weight')*reward_distance
        human_action_reward = self.config('action_weight')*reward_action
        human_food_reward = self.config('scratch_reward_weight')*reward_force_scratch # 记得food reward就是scratch reward 
        human_pref_reward = preferences_score
        robot_dist_reward =  self.config('distance_weight')*reward_distance
        robot_action_reward = self.config('action_weight')*reward_action
        robot_food_reward =  self.config('scratch_reward_weight')*reward_force_scratch #
        # robot_pref_reward =  0
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
        


        reward_human = human_dist_reward + human_action_reward + human_food_reward + human_pref_reward
        reward_robot = robot_dist_reward + robot_action_reward + robot_food_reward + robot_pref_reward
        reward = 0.5 * (reward_human + reward_robot)


        # reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('scratch_reward_weight')*reward_force_scratch + preferences_score

        if self.gui and self.tool_force_at_target > 0:
            print('Task success:', self.task_success, 'Tool force at target:', self.tool_force_at_target, reward_force_scratch)

        # info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        info = {'particles' : self.task_success, 'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        
        
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        # else:
        #     # Co-optimization with both human and robot controllable
        #     return obs, {'robot': reward, 'human': reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}
        else:
            # Co-optimization with both human and robot controllable
            #新代码
            return obs, {'robot': reward_robot, 'human': reward_human, '__all__':reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}, {'human_dist_reward': human_dist_reward, 'human_action_reward': human_action_reward, 'human_food_reward': human_food_reward, 'human_pref_reward' : human_pref_reward,
            'robot_dist_reward': robot_dist_reward, 'robot_action_reward': robot_action_reward, 'robot_food_reward': robot_food_reward, 'robot_pref_reward' : robot_pref_reward, 'vel': prefv, 'force': preff, 'h_force' : prefh, 'hit' : prefhit, 'food_v' : preffv}, pref_array



    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_at_target = 0
        target_contact_pos = None
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            # Enforce that contact is close to the target location
            if linkA in [0, 1] and np.linalg.norm(posB - self.target_pos) < 0.025:
                tool_force_at_target += force
                target_contact_pos = posB
        return total_force_on_human, tool_force, tool_force_at_target, None if target_contact_pos is None else np.array(target_contact_pos)

    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        target_pos_real, _ = self.robot.convert_to_realworld(self.target_pos)
        self.total_force_on_human, self.tool_force, self.tool_force_at_target, self.target_contact_pos = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, tool_pos_real - target_pos_real, target_pos_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            target_pos_human, _ = self.human.convert_to_realworld(self.target_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, tool_pos_human - target_pos_human, target_pos_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_at_target]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(ScratchItchEnv, self).reset()
        self.build_assistive_env('wheelchair')
        self.prev_target_contact_pos = np.zeros(3)
        if self.robot.wheelchair_mounted:
            wheelchair_pos, wheelchair_orient = self.furniture.get_base_pos_orient()
            self.robot.set_base_pos_orient(wheelchair_pos + np.array(self.robot.toc_base_pos_offset[self.task]), [0, 0, -np.pi/2.0])

        # self.robot.print_joint_info()

        # Set joint angles for human joints (in degrees)
        joints_positions = [(self.human.j_right_shoulder_x, 30), (self.human.j_right_elbow, -90), (self.human.j_left_elbow, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee, 80)]
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None if self.human.controllable else 1, reactive_gain=0.01)

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[0.001]*3)

        target_ee_pos = np.array([-0.6, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.generate_target()

        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_target(self):
        # Randomly select either upper arm or forearm for the target limb to scratch
        if self.human.gender == 'male':
            self.limb, length, radius = [[self.human.right_shoulder, 0.279, 0.043], [self.human.right_elbow, 0.257, 0.033]][self.np_random.randint(2)]
        else:
            self.limb, length, radius = [[self.human.right_shoulder, 0.264, 0.0355], [self.human.right_elbow, 0.234, 0.027]][self.np_random.randint(2)]
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

        self.update_targets()

    def update_targets(self):
        arm_pos, arm_orient = self.human.get_pos_orient(self.limb)
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        self.target.set_base_pos_orient(self.target_pos, [0, 0, 0, 1])

