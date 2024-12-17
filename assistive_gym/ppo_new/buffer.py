import numpy as np
import scipy.signal
import torch

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x using NumPy.

    Args:
        x: An array containing samples of the scalar to produce statistics for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)
    std = np.std(x)

    if with_min_and_max:
        min_val = np.min(x)
        max_val = np.max(x)
        return mean, std, min_val, max_val
    return mean, std

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PPOBufferDempref:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, pref_dim, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.pref_buf = np.zeros(combined_shape(size, pref_dim), dtype=np.float32) # 这计算出来的是按照我们的estimate算出来的pref_reward
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, pref, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     
        self.pref_buf[self.ptr] = pref
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size   
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, pref_entries=self.pref_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}




class PPOBuffer_Social:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, training_size, training_obsdim, stride, gamma=0.99, lam=0.95, social_space=4, input_dim=7,robot_start=0,human_start=0,human_len=0,robot_len=0):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.social_space = social_space
        self.input_dim = input_dim
        self.robot_start = robot_start
        self.human_start = human_start
        self.human_len = human_len
        self.robot_len = robot_len
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.training_obs_buf = np.zeros(combined_shape(size, training_obsdim), dtype=np.float32)
        self.training_obsdim = training_obsdim
        self.training_size = training_size
        self.stride = stride
        self.training_buf = np.zeros((1, 2, training_size, input_dim), dtype=np.float32) # this is for model input

        self.social_pred_buf = np.zeros(combined_shape(size, social_space), dtype=np.float32)
        
    def store(self, obs, act, rew, val, logp, training_obs, pred_buf):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.training_obs_buf[self.ptr] = training_obs
        self.social_pred_buf[self.ptr] = pred_buf
        self.ptr += 1


    def reset_training_buf(self):
        self.training_buf = np.zeros((1, 2, self.training_size, self.input_dim), dtype=np.float32) # this is for model input

    def organize_data(self): # keeping enough data but not too much
        np.random.shuffle(self.training_buf)
        self.training_buf = self.training_buf[:10000]
        

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        for i in range(self.path_start_idx, self.ptr, self.stride):
            if i+self.training_size>self.ptr:
                break
            poses_human = self.training_obs_buf[i:i+self.training_size,self.human_start:self.human_start+self.human_len]
            poses_robot = self.obs_buf[i:i+self.training_size,self.robot_start:self.robot_start+self.robot_len]
            if self.human_len < self.robot_len:
                poses_human = np.pad(poses_human, pad_width=((0, 0), (0, self.robot_len-self.human_len)), mode='constant', constant_values=(0,))
            else:
                poses_robot = np.pad(poses_robot, pad_width=((0, 0), (0, self.human_len-self.robot_len)), mode='constant', constant_values=(0,))
            
            self.training_buf = np.append(self.training_buf, [np.concatenate((np.expand_dims(poses_human, axis=0),np.expand_dims(poses_robot, axis=0)), axis=0)],axis=0)

        self.path_start_idx = self.ptr

        


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, social=self.social_pred_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class PPOBuffer_Social_Dempref:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, pref_dim, obs_dim, act_dim, size, training_size, training_obsdim, stride, gamma=0.99, lam=0.95, social_space=4, input_dim=7,robot_start=0,human_start=0,human_len=0,robot_len=0):
        self.pref_buf = np.zeros(combined_shape(size, pref_dim), dtype=np.float32) # 这计算出来的是按照我们的estimate算出来的pref_reward
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.social_space = social_space
        self.input_dim = input_dim
        self.robot_start = robot_start
        self.human_start = human_start
        self.human_len = human_len
        self.robot_len = robot_len
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.training_obs_buf = np.zeros(combined_shape(size, training_obsdim), dtype=np.float32)
        self.training_obsdim = training_obsdim
        self.training_size = training_size
        self.stride = stride
        self.training_buf = np.zeros((1, 2, training_size, input_dim), dtype=np.float32) # this is for model input

        self.social_pred_buf = np.zeros(combined_shape(size, social_space), dtype=np.float32)
        
    def store(self, pref, obs, act, rew, val, logp, training_obs, pred_buf):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     
        self.pref_buf[self.ptr] = pref
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.training_obs_buf[self.ptr] = training_obs
        self.social_pred_buf[self.ptr] = pred_buf
        self.ptr += 1


    def reset_training_buf(self):
        self.training_buf = np.zeros((1, 2, self.training_size, self.input_dim), dtype=np.float32) # this is for model input

    def organize_data(self): 
        np.random.shuffle(self.training_buf)
        self.training_buf = self.training_buf[:10000]
        

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        for i in range(self.path_start_idx, self.ptr, self.stride):
            if i+self.training_size>self.ptr:
                break
            poses_human = self.training_obs_buf[i:i+self.training_size,self.human_start:self.human_start+self.human_len]
            poses_robot = self.obs_buf[i:i+self.training_size,self.robot_start:self.robot_start+self.robot_len]
            if self.human_len < self.robot_len:
                poses_human = np.pad(poses_human, pad_width=((0, 0), (0, self.robot_len-self.human_len)), mode='constant', constant_values=(0,))
            else:
                poses_robot = np.pad(poses_robot, pad_width=((0, 0), (0, self.human_len-self.robot_len)), mode='constant', constant_values=(0,))
            
            self.training_buf = np.append(self.training_buf, [np.concatenate((np.expand_dims(poses_human, axis=0),np.expand_dims(poses_robot, axis=0)), axis=0)],axis=0)

        self.path_start_idx = self.ptr

        


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, social=self.social_pred_buf, 
                    pref_entries=self.pref_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


