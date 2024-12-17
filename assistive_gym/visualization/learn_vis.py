import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from assistive_gym.ppo_new import ppo, buffer, sac
import json

import os, sys, time, copy, glob, importlib
import re
from numpngw import write_png, write_apng
from PIL import Image, ImageDraw, ImageFont
import random

#reading ini
import configparser

#for comments and code usage, please refer to training scripts

def print_metrics(success_ratio, time_consumption, particles_eaten):
    print(f"Success Ratio: {success_ratio:.2f}")
    print(f"Time Consumption: {time_consumption:.2f}")
    print(f"Particles Eaten: {particles_eaten:.2f}")

def make_env(args, env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(args.seed)
    # setting preferences
    env.given_pref = args.given_pref
    env.C_v = args.velocity_weight
    env.C_f = args.force_nontarget_weight
    env.C_hf = args.high_forces_weight
    env.C_fd = args.food_hit_weight
    env.C_fdv = args.food_velocities_weight

    return env

def vis(args):
    # creating co-op assistive gym env
    env_name = args.env
    coop = ('Human' in args.env)
    env = make_env(env_name, coop)
    env.seed(args.seed)
    obs = env.reset()

    #Setting camera angle here
    env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)

    obsdim_robot = env.observation_space_robot
    obsdim_human = env.observation_space_human
    actdim_robot = env.action_space_robot
    actdim_human = env.action_space_human
    pref_dim = args.pref_dim

    robot_prefix = '/robot_'
    human_prefix = '/human_'
    
    if args.algo == "PPO":
        if args.social:
            if args.dempref:
                agent_robot = ppo.PPO_Social_Dempref(pref_dim, obsdim_robot, actdim_robot, args, load_policy = args.load_policy_path+robot_prefix+ str(args.load_epoch)+'_social' if args.load_policy_path is not None else None)
            else:
                agent_robot = ppo.PPO_Social(obsdim_robot, actdim_robot, args, load_policy = args.load_policy_path+robot_prefix+ str(args.load_epoch)+'_social' if args.load_policy_path is not None else None)
        else:
            if args.dempref:
                agent_robot = ppo.PPO_Dempref(pref_dim, obsdim_robot, actdim_robot, load_policy = args.load_policy_path+robot_prefix+str(args.load_epoch) if args.load_policy_path is not None else None)
            else:
                agent_robot = ppo.PPO(obsdim_robot, actdim_robot, load_policy = args.load_policy_path+robot_prefix+ str(args.load_epoch) if args.load_policy_path is not None else None)
        agent_human = ppo.PPO(obsdim_human, actdim_human, load_policy = args.load_policy_path+human_prefix+ str(args.load_epoch) if args.load_policy_path is not None else None)
    

    episode_rewards = []
    human_episode_rewards = []
    robot_episode_rewards = []
    eval_episode_rewards = []
    eval_human_rewards = []
    eval_robot_rewards = []
    eval_success = []
    ep_reward = 0
    ep_human_ret, ep_robot_ret = 0, 0
    total_len = 0
    total_particles = 0

    obsdim_human = env.observation_space_human.shape[0]
    obsdim_robot = env.observation_space_robot.shape[0]
    ep_len = 0
    human_obs_buf = np.zeros((201, obsdim_human), dtype=np.float32)
    robot_obs_buf = np.zeros((201, obsdim_robot), dtype=np.float32)
    human_obs_buf[ep_len] = obs['human']
    robot_obs_buf[ep_len] = obs['robot']
    ep_len += 1

    
    frames = []
    while len(eval_episode_rewards) < 20:
        with torch.no_grad():
            if args.algo == "PPO":
                if args.social and ep_len >= 10:
                    poses_human = human_obs_buf[ep_len-10:ep_len,args.human_start:args.human_start+args.human_len]
                    poses_robot = robot_obs_buf[ep_len-10:ep_len,args.robot_start:args.robot_start+args.robot_len]
                            
                    if args.human_len < args.robot_len:
                        poses_human = np.pad(poses_human, pad_width=((0, 0), (0, args.robot_len-args.human_len)), mode='constant', constant_values=(0,))
                    else:
                        poses_robot = np.pad(poses_robot, pad_width=((0, 0), (0, args.human_len-args.robot_len)), mode='constant', constant_values=(0,))
            
                    social_pred_human = agent_robot.social_inference(poses_human, poses_robot) # this should return (10,4)
                        
                    if args.dynamic_future:
                        if ep_len > 100:
                            social_decay = 0.4
                        elif ep_len > 50:
                            social_decay = 0.7
                        else:
                            social_decay = 0.9
                    else:
                        social_decay = 0.1 * args.fix_future
                    social_pred_human_np = social_pred_human[int(social_decay*10)].cpu().numpy()
                            
                    thorough_obs = np.concatenate((obs['robot'], social_pred_human_np))
                    a_robot, v_robot, logp_robot = agent_robot.ac.step(torch.as_tensor(thorough_obs, dtype=torch.float32))
                elif args.social and ep_len < 10:
                    social_pred_human = obs['human'][args.human_start:args.human_start+args.human_len]
                    social_pred_human_flat = social_pred_human.flatten()

                    thorough_obs = np.concatenate((obs['robot'], social_pred_human_flat))
                    a_robot, v_robot, logp_robot = agent_robot.ac.step(torch.as_tensor(thorough_obs, dtype=torch.float32))
                else:
                    a_robot, v_robot, logp_robot = agent_robot.ac.step(torch.as_tensor(obs['robot'], dtype=torch.float32))
            else: # other algos
                pass
            a_human, v_human, logp_human = agent_human.ac.step(torch.as_tensor(obs['human'], dtype=torch.float32))
            action = {'robot': a_robot, 'human': a_human}
            obs, r, d, info, s_r = env.step(action, args, success_rate)
            
            if args.algo == "PPO":
                human_obs_buf[ep_len] = obs['human']
                robot_obs_buf[ep_len] = obs['robot']
            else:
                pass
            ep_len += 1

            img, depth = env.get_camera_image_depth()
            frames.append(img)
            ep_reward += r['__all__']
            ep_human_ret += r['human']
            ep_robot_ret += r['robot']

            if d['__all__'] or info['task_success'] == 1:
                total_len += ep_len
                total_particles += info['particles']
                ep_len = 0
                eval_episode_rewards.append(ep_reward)
                eval_human_rewards.append(ep_human_ret)
                eval_robot_rewards.append(ep_robot_ret)
                eval_success.append(info['robot']['task_success'])
                ep_reward = 0 
                ep_human_ret, ep_robot_ret = 0, 0
                obs = env.reset()
                if args.algo == "PPO":
                    human_obs_buf[ep_len] = obs['human']
                    robot_obs_buf[ep_len] = obs['robot']
                    ep_len+=1
                else:
                    ep_len+=1
                if info['robot']['task_success'] == 0:
                    if args.social:
                        if args.dempref:
                            save_name = os.path.join(args.vis_save_dir, 'dempref'+args.exp_name+str(args.load_epoch)+f'{str(len(eval_episode_rewards))}_fail.png')
                        else:
                            save_name = os.path.join(args.vis_save_dir, args.exp_name+str(args.load_epoch)+f'{str(len(eval_episode_rewards))}_fail.png')
                    else:
                        save_name = os.path.join(args.vis_save_dir, 'baseline'+args.exp_name+str(args.load_epoch)+f'{str(len(eval_episode_rewards))}_fail.png')
                else:
                    if args.social:
                        if args.dempref:
                            save_name = os.path.join(args.vis_save_dir, 'dempref'+args.exp_name+str(args.load_epoch)+f'{str(len(eval_episode_rewards))}.png')
                        else:
                            save_name = os.path.join(args.vis_save_dir, args.exp_name+str(args.load_epoch)+f'{str(len(eval_episode_rewards))}.png')
                    else:
                        save_name = os.path.join(args.vis_save_dir, 'baseline'+args.exp_name+str(args.load_epoch)+f'{str(len(eval_episode_rewards))}.png')
                if False:
                    write_apng(save_name, frames, delay=100)
                frames = []
    
    count_ones = sum(eval_success)
    success_ratio = count_ones / 20.0
    time_consumption = total_len / 20.0
    particles_eaten = total_particles / 20.0
    print_metrics(success_ratio, time_consumption, particles_eaten)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FeedingSawyerHuman-v1') # modify this for new reward settings
    parser.add_argument('--exp-name', type=str, default='PPO') # PPO TD3 MADDPG
    parser.add_argument('--algo', type=str, default='') # modify for setting algorithm


    parser.add_argument('--setting-name', type=int, default=1) # pref_setting_number
    parser.add_argument('--velocity-weight', type=float, default=0.25)
    parser.add_argument('--force-nontarget-weight', type=float, default=0.01)
    parser.add_argument('--high-forces-weight', type=float, default=0.05)
    parser.add_argument('--food-hit-weight', type=float, default=1.0)
    parser.add_argument('--food-velocities-weight', type=float, default=1.0)
    parser.add_argument('--r-velocity', type=float, default=0.25)
    parser.add_argument('--r-force', type=float, default=0.01)
    parser.add_argument('--r-h-force', type=float, default=0.05)
    parser.add_argument('--r-hit', type=float, default=1.0)
    parser.add_argument('--r-food-v', type=float, default=1.0)
    parser.add_argument('--previous-suc-rate', type=float, default=1.0)
    
    # for other tasks
    parser.add_argument('--dressing-force-weight', type=float, default=1.0)
    parser.add_argument('--high-pressures-weightt', type=float, default=1.0)

    #dempref args
    parser.add_argument("--dempref", action="store_true", help="Set the flag to True")
    parser.add_argument('--pref-dim', type=int, default=5) # for dempref
    parser.add_argument("--continuous-sampling", action="store_true", help="Set the flag to True")
    parser.add_argument('--n-samples-summ', type=int, default=50000) 
    parser.add_argument('--merge-alpha', type=float, default=0.9)
    
    parser.add_argument("--given-pref", action="store_true", help="Set the flag to True")

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--max-ep-len', type=int, default=200)
    parser.add_argument('--PPO-epochs', type=int, default=600) # ppo epochs
    parser.add_argument('--steps-per-epoch', type=int, default=4000)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--log-interval', type=int, default=2)
    parser.add_argument('--eval-interval', type=int, default=2)
    parser.add_argument('--vis-save-dir', default='./saco_vis/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default=None,
                        help='Path name to saved policy checkpoint ("/data/add_disk0/jason/trained_models/")')
    parser.add_argument('--load-epoch', type=int, default=0) # modify for PPO policy versions
    parser.add_argument("--late", action="store_true", help="Set the flag to True") # loading late policy
    
    #anticipation args
    parser.add_argument("--social", action="store_true", help="Set the flag to True")
    parser.add_argument("--baseline", action="store_true", help="Set the flag to True")
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--d-inner-g', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.000005)
    parser.add_argument('--k-levels', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20) # train anticipation module for x epochs
    parser.add_argument('--batch-size', type=int, default=1536)
    parser.add_argument('--lr-decay', type=float, default=0.99)
    parser.add_argument('--lr-decay-step', type=float, default=40)
    parser.add_argument('--lambda_recon', type=float, default=1)
    parser.add_argument('--train-freq', type=int, default=5)
    parser.add_argument("--dynamic-future", action="store_true", help="Set the flag to True")
    parser.add_argument('--fix-future', type=int, default=4)
    parser.add_argument('--social-space', type=int, default=4) # should be human joint dimension
    parser.add_argument('--input-dim', type=int, default=7) # should be longest dimension of either agents
    parser.add_argument('--robot-start', type=int, default=10) # start of robot joint angle, for sawyer it is 10
    parser.add_argument('--human-start', type=int, default=10) # similar to above, for sawyer it is 10
    parser.add_argument('--human-len', type=int, default=4) # similar to below, for feeding sawyer it is 4
    parser.add_argument('--robot-len', type=int, default=7) # len of robot joints, for sawyer it is 7
    

    args = parser.parse_args()
    args.algo = args.exp_name
    args.dynamic_future = True


    config_file = './assistive_gym/pref.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    setting = config[str(args.setting_name)]
    args.velocity_weight = setting['velocity_weight']
    args.force_nontarget_weight = setting['force_nontarget_weight']
    args.high_forces_weight = setting['high_forces_weight']
    args.food_hit_weight = setting['food_hit_weight']
    args.food_velocities_weight = setting['food_velocities_weight']
    
    if args.given_pref:
        args.exp_name += f"given_pref_"
    if args.dempref:
        args.exp_name += f"conditioned_dempref_"
    if args.continuous_sampling:
        args.exp_name += f"continuous_sampling_merge_alpha={args.merge_alpha}_"
    args.exp_name += f"setting{args.setting_name}"
    args.exp_name += "_seed="+str(args.seed)

    if args.social:
        args.exp_name += "_social_"+str(args.train_freq) + "update"
        args.exp_name += "_decay"+str(args.lr_decay)
        args.exp_name += "_lr"+str(args.lr)
    pattern = r'Feeding(.*?)Human-v1'
    text = args.env
    if "Drinking" in text:
        pattern = r'Drinking(.*?)Human-v1'
    elif "ScratchItch" in text:
        pattern = r'ScratchItch(.*?)Human-v1'
    elif "BedBathing" in text:
        pattern = r'BedBathing(.*?)Human-v1'
    elif "ArmManipulation" in text:
        pattern = r'ArmManipulation(.*?)Human-v1'
    elif "Dressing" in text:
        pattern = r'Dressing(.*?)Human-v1'

    result = re.search(pattern, text) 
    if "Feeding" in text:
        project_name = f'SACO-{result.group(1)}'
    else: # for other tasks
        project_name = f'SACO-{result.group(0)}'

    if args.social: # for social PPO
        args.vis_save_dir += 'social' + '/'
        if not os.path.exists(args.vis_save_dir):
            os.makedirs(args.vis_save_dir)
    if "Feeding" in text:
        pass
    else:
        args.vis_save_dir += result.group(0) + '/'
        if not os.path.exists(args.vis_save_dir):
            os.makedirs(args.vis_save_dir)
    args.vis_save_dir += result.group(1) + '/'
    if not os.path.exists(args.vis_save_dir):
        os.makedirs(args.vis_save_dir)

    if args.dempref:
        args.vis_save_dir += "Dempref_"
    
    if args.social:
        social = "_social"
    else:
        social = ''

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vis(args)