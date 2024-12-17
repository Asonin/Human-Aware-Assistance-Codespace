import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from assistive_gym.ppo_new import ppo, buffer
import json

import os, sys, time, copy, glob, importlib
import re

# tracing experiments
import wandb
import random

#reading ini
import configparser


def make_env(args, env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        # environments are modified, if you with to further modify, please go to assistive_gym.envs
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(args.seed)
    # setting preferences from args
    env.given_pref = args.given_pref
    env.C_v = args.velocity_weight
    env.C_f = args.force_nontarget_weight
    env.C_hf = args.high_forces_weight
    env.C_fd = args.food_hit_weight
    env.C_fdv = args.food_velocities_weight

    return env

def eval(args, agent_robot, agent_human, epoch, estimated_weights, success_rate):
            episode_rewards = []
            human_episode_rewards = []
            robot_episode_rewards = []
            split_rewards_period = {'eval_human_dist_reward': 0, 'eval_human_action_reward': 0, 'eval_human_food_reward': 0, 'eval_human_pref_reward' : 0,
                'eval_robot_dist_reward': 0, 'eval_robot_action_reward': 0, 'eval_robot_food_reward': 0, 'eval_robot_pref_reward' : 0,
                'eval_vel': 0, 'eval_force': 0, 'eval_h_force' : 0, 'eval_hit' : 0, 'eval_food_v' : 0
            }
            

            env_name = args.env
            coop = ('Human' in args.env)
            eval_envs = make_env(args, env_name, coop)
            eval_envs.seed(args.seed + 100)
            
            eval_episode_rewards = []
            eval_episode_time = []
            eval_episode_particles = []
            eval_human_rewards = []
            eval_robot_rewards = []
            eval_success = []
            obs = eval_envs.reset()

            ep_reward = 0
            ep_human_ret, ep_robot_ret, ep_len = 0, 0, 0
            split_rewards = {'human_dist_reward': 0, 'human_action_reward': 0, 'human_food_reward': 0, 'human_pref_reward' : 0,
                'robot_dist_reward': 0, 'robot_action_reward': 0, 'robot_food_reward': 0, 'robot_pref_reward' : 0,
                'vel': 0, 'force': 0, 'h_force' : 0, 'hit' : 0, 'food_v' : 0
            }

            if args.algo == "PPO":
                obsdim_human = eval_envs.observation_space_human.shape[0]
                obsdim_robot = eval_envs.observation_space_robot.shape[0]
                
                human_obs_buf = np.zeros((201, obsdim_human), dtype=np.float32)
                robot_obs_buf = np.zeros((201, obsdim_robot), dtype=np.float32)
                human_obs_buf[ep_len] = obs['human']
                robot_obs_buf[ep_len] = obs['robot']
                ep_len += 1
            #20 evals
            while len(eval_episode_rewards) < 20:
                with torch.no_grad():
                    if args.algo == "PPO":
                        # Anticipation module works here
                        if args.social and ep_len >= 10:
                            poses_human = human_obs_buf[ep_len-10:ep_len,args.human_start:args.human_start+args.human_len]
                            poses_robot = robot_obs_buf[ep_len-10:ep_len,args.robot_start:args.robot_start+args.robot_len]
                            
                            if args.human_len < args.robot_len:
                                poses_human = np.pad(poses_human, pad_width=((0, 0), (0, args.robot_len-args.human_len)), mode='constant', constant_values=(0,))
                            else:
                                poses_robot = np.pad(poses_robot, pad_width=((0, 0), (0, args.human_len-args.robot_len)), mode='constant', constant_values=(0,))
            
                            social_pred_human = agent_robot.social_inference(poses_human, poses_robot) # this should return (10,4)
                            #Dynamic future mechanism works
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
                        a_human, v_human, logp_human = agent_human.ac.step(torch.as_tensor(obs['human'], dtype=torch.float32))

                    action = {'robot': a_robot, 'human': a_human}
                    obs, r, d, info, s_r, pref_entries = eval_envs.step(action, args, estimated_weights, success_rate)
                    
                    if args.algo == "PPO":
                        human_obs_buf[ep_len] = obs['human']
                        robot_obs_buf[ep_len] = obs['robot']
                    else:
                        pass
                    ep_len += 1
                    ep_reward += r['__all__']
                    ep_human_ret += r['human']
                    ep_robot_ret += r['robot']
                    for k in s_r:
                        split_rewards[k] += s_r[k]

                    if d['__all__'] or info['human']['task_success'] == 1: # could end episode earlier
                        #Recording rewards
                        eval_episode_rewards.append(ep_reward)
                        eval_episode_time.append(ep_len)
                        eval_episode_particles.append(info['human']['particles'])
                        eval_human_rewards.append(ep_human_ret)
                        eval_robot_rewards.append(ep_robot_ret)
                        # recored split period
                        for k in split_rewards:
                            split_rewards_period["eval_"+k] += split_rewards[k]
                        eval_success.append(info['robot']['task_success'])
                        
                        ep_len = 0
                        ep_reward = 0 
                        ep_human_ret, ep_robot_ret = 0, 0
                        for k in s_r:
                            split_rewards[k] = 0
                        obs = eval_envs.reset()

                        if args.algo == "PPO":
                            human_obs_buf[ep_len] = obs['human']
                            robot_obs_buf[ep_len] = obs['robot']
                            ep_len+=1
                        else:
                            pass

            eval_envs.close()
            avg_reward = np.array(eval_episode_rewards)
            max_value = np.max(avg_reward)
            min_value = np.min(avg_reward)
            mean_value = np.mean(avg_reward)

            avg_time = np.array(eval_episode_time)
            max_time = np.max(avg_time)
            min_time = np.min(avg_time)
            mean_time = np.mean(avg_time)

            avg_par = np.array(eval_episode_particles)
            max_par = np.max(avg_par)
            min_par = np.min(avg_par)
            mean_par = np.mean(avg_par)


            avg_human_reward = np.array(eval_human_rewards)
            max_human_value = np.max(avg_human_reward)
            min_human_value = np.min(avg_human_reward)
            mean_human_value = np.mean(avg_human_reward)

            avg_robot_reward = np.array(eval_robot_rewards)
            max_robot_value = np.max(avg_robot_reward)
            min_robot_value = np.min(avg_robot_reward)
            mean_robot_value = np.mean(avg_robot_reward)

            wandb.log({
                    "eval_max_value": max_value,
                    "eval_min_value": min_value,
                    "eval_mean_value": mean_value,

                    "eval_max_time": max_time,
                    "eval_min_time": min_time,
                    "eval_mean_time": mean_time,

                    "eval_max_par": max_par,
                    "eval_min_par": min_par,
                    "eval_mean_par": mean_par,

                    "eval_max_human_value": max_human_value,
                    "eval_min_human_value": min_human_value,
                    "eval_mean_human_value": mean_human_value,

                    "eval_max_robot_value": max_robot_value,
                    "eval_min_robot_value": min_robot_value,
                    "eval_mean_robot_value": mean_robot_value,
                    "eval_success_rate": np.sum(eval_success)/20.0
                })

            for k in split_rewards_period:
                split_rewards_period[k] /= 20.0
            wandb.log(split_rewards_period)

            split_rewards_period = {'human_dist_reward_1000': 0, 'human_action_reward_1000': 0, 'human_food_reward_1000': 0, 'human_pref_reward_1000' : 0,
                'robot_dist_reward_1000': 0, 'robot_action_reward_1000': 0, 'robot_food_reward_1000': 0, 'robot_pref_reward_1000' : 0
            }
            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_human_rewards)))
            sys.stdout.flush()

            return mean_human_value, np.sum(eval_success)/20.0


def train_ppo(args):
    # creating co-op assistive gym env
    env_name = args.env
    coop = ('Human' in args.env)
    env = make_env(args, env_name, coop)
    env.seed(args.seed)
    obs = env.reset()

    obsdim_robot = env.observation_space_robot
    obsdim_human = env.observation_space_human
    actdim_robot = env.action_space_robot
    actdim_human = env.action_space_human
    pref_dim = args.pref_dim

    #Initialize Utility module
    estimated_weights = ppo.EstimatedWeights(env.C_v,env.C_f,env.C_hf,env.C_fd,env.C_fdv)
    if args.late:
        robot_prefix = '/late_robot_'
        human_prefix = '/late_human_'
    else:
        robot_prefix = '/robot_'
        human_prefix = '/human_'
    if args.algo == "PPO":
        #Anticipation on
        if args.social:
            #Utility on
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
    

    if args.algo == "PPO":
        if args.social:
            training_size = 20
            stride = 2
            if args.dempref:
                buf_robot = buffer.PPOBuffer_Social_Dempref(pref_dim, obsdim_robot.shape, actdim_robot.shape, args.steps_per_epoch, training_size, obsdim_human.shape, stride, args.gamma, args.lam, args.social_space, args.input_dim, args.robot_start, args.human_start , args.human_len, args.robot_len)
            else:
                buf_robot = buffer.PPOBuffer_Social(obsdim_robot.shape, actdim_robot.shape, args.steps_per_epoch, training_size, obsdim_human.shape, stride, args.gamma, args.lam, args.social_space, args.input_dim, args.robot_start, args.human_start , args.human_len, args.robot_len)
        else:
            if args.dempref:
                buf_robot = buffer.PPOBufferDempref(pref_dim, obsdim_robot.shape, actdim_robot.shape, args.steps_per_epoch, args.gamma, args.lam)
            else:
                buf_robot = buffer.PPOBuffer(obsdim_robot.shape, actdim_robot.shape, args.steps_per_epoch, args.gamma, args.lam)
        buf_human = buffer.PPOBuffer(obsdim_human.shape, actdim_human.shape, args.steps_per_epoch, args.gamma, args.lam)


    # For TD3 and MADDPG, please refer to my implementation in folders and modify code similarly.
    # record rewards
    episode_rewards = []
    human_episode_rewards = []
    robot_episode_rewards = []
    split_rewards_period = {'human_dist_reward_1000': 0, 'human_action_reward_1000': 0, 'human_food_reward_1000': 0, 'human_pref_reward_1000' : 0,
                'robot_dist_reward_1000': 0, 'robot_action_reward_1000': 0, 'robot_food_reward_1000': 0, 'robot_pref_reward_1000' : 0,
                'vel_1000': 0, 'force_1000': 0, 'h_force_1000' : 0, 'hit_1000' : 0, 'food_v_1000' : 0
            }
    
    total_num_steps = 0
    max_eval_mean_rewards = -1000 # a very low value
    success_rate = 0.0

    # Prepare for interaction with environment
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    ep_human_ret, ep_robot_ret = 0, 0
    epi_cnt = 0
    split_rewards = {'human_dist_reward': 0, 'human_action_reward': 0, 'human_food_reward': 0, 'human_pref_reward' : 0,
                'robot_dist_reward': 0, 'robot_action_reward': 0, 'robot_food_reward': 0, 'robot_pref_reward' : 0,
                'vel': 0, 'force': 0, 'h_force' : 0, 'hit' : 0, 'food_v' : 0
            }

    # We will resume the experiment from the checkpoint time.
    if args.load_policy_path is not None:
        start_epoch = args.load_epoch+1
    else:
        start_epoch = 0
    total_num_steps = args.load_epoch*args.steps_per_epoch
    for epoch in range(start_epoch, args.PPO_epochs):
        if args.load_policy_path is not None and epoch == start_epoch:
            mean_value, success_rate = eval(args, agent_robot, agent_human, epoch, estimated_weights, success_rate)
            max_eval_mean_rewards = mean_value
        for t in range(args.steps_per_epoch): 
            total_num_steps += 1
            # Same thing as in Eval
            if args.social and ep_len >= 10:
                poses_human = buf_robot.training_obs_buf[buf_robot.ptr-10:buf_robot.ptr,args.human_start:args.human_start+args.human_len] # 这个indices需要改，适应人的关节位置
                poses_robot = buf_robot.obs_buf[buf_robot.ptr-10:buf_robot.ptr,args.robot_start:args.robot_start+args.robot_len] # 同上
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
            a_human, v_human, logp_human = agent_human.ac.step(torch.as_tensor(obs['human'], dtype=torch.float32))
            
            action = {'robot': a_robot, 'human': a_human}
            
            next_obs, r, d, info, s_r, pref_entries = env.step(action, args, estimated_weights, success_rate)

            ep_ret += r['__all__']
            ep_human_ret += r['human']
            ep_robot_ret += r['robot']
            for k in s_r:
                split_rewards[k] += s_r[k]
            ep_len += 1

            timeout = ep_len == args.max_ep_len
            terminal = d['__all__'] or timeout or info['robot']['task_success']
            epoch_ended = t==args.steps_per_epoch-1

            if args.algo == "PPO":
                # preparing data for social inference
                if args.social and ep_len >= 11:
                    social_pred_human = social_pred_human[int(social_decay*10)].cpu().numpy()
                else:
                    social_pred_human = next_obs['human'][args.human_start:args.human_start+args.human_len]
                if terminal:
                    social_pred_human = next_obs['human'][args.human_start:args.human_start+args.human_len]
                if args.social:
                    if args.dempref:
                        buf_robot.store(pref_entries, obs['robot'], a_robot, r['robot'], v_robot, logp_robot, obs['human'],social_pred_human)
                    else:   
                        buf_robot.store(obs['robot'], a_robot, r['robot'], v_robot, logp_robot, obs['human'],social_pred_human)
                else:
                    if args.dempref:
                        buf_robot.store(pref_entries, obs['robot'], a_robot, r['robot'], v_robot, logp_robot)
                    else:
                        buf_robot.store(obs['robot'], a_robot, r['robot'], v_robot, logp_robot)
                buf_human.store(obs['human'], a_human, r['human'], v_human, logp_human)

            obs = next_obs

            if terminal or epoch_ended:
                if args.algo == "PPO":
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # If trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        if args.social:
                            social_pred_human = obs['human'][args.human_start:args.human_start+args.human_len]
                            social_pred_human_flat = social_pred_human.flatten()
                            thorough_obs = np.concatenate((obs['robot'], social_pred_human_flat))
                            a_robot, v_robot, logp_robot = agent_robot.ac.step(torch.as_tensor(thorough_obs, dtype=torch.float32))
                        else:
                            a_robot, v_robot, logp_robot = agent_robot.ac.step(torch.as_tensor(obs['robot'], dtype=torch.float32))
                        a_human, v_human, logp_human = agent_human.ac.step(torch.as_tensor(obs['human'], dtype=torch.float32))
                    else:
                        v_robot = 0
                        v_human = 0
                    buf_robot.finish_path(v_robot)
                    buf_human.finish_path(v_human)
                else: # Other algos do not to do anything so far.
                    pass

                if terminal:
                    epi_cnt += 1
                    episode_rewards.append(ep_ret)
                    human_episode_rewards.append(ep_human_ret)
                    robot_episode_rewards.append(ep_robot_ret)
                    for k in split_rewards:
                        split_rewards_period[k+"_1000"] += split_rewards[k]

                    wandb.log({"reward": ep_ret})
                    wandb.log({"reward_human": ep_human_ret})
                    wandb.log({"reward_robot": ep_robot_ret})
                    wandb.log(split_rewards)
                    wandb.log({"total_steps": total_num_steps})
                    
                
                if epi_cnt % 5 == 0:
                    epi_cnt = 0
                    wandb.log({
                    "mean_reward_over1000":np.mean(episode_rewards[-5:]),
                    "max_reward_over1000" :np.max(episode_rewards[-5:]),
                    "min_reward_over1000" :np.min(episode_rewards[-5:]) 
                            })
                    wandb.log({
                    "mean_human_reward_over1000":np.mean(human_episode_rewards[-5:]),
                    "max_human_reward_over1000" :np.max(human_episode_rewards[-5:]),
                    "min_human_reward_over1000" :np.min(human_episode_rewards[-5:]) 
                            })
                    wandb.log({
                    "mean_robot_reward_over1000":np.mean(robot_episode_rewards[-5:]),
                    "max_robot_reward_over1000" :np.max(robot_episode_rewards[-5:]),
                    "min_robot_reward_over1000" :np.min(robot_episode_rewards[-5:]) 
                            })

                    for k in split_rewards_period:
                        split_rewards_period[k] /= 5.0
                    wandb.log(split_rewards_period)
                    for k in split_rewards_period:
                        split_rewards_period[k] = 0

                obs, ep_ret, ep_len = env.reset(), 0, 0
                ep_human_ret, ep_robot_ret = 0, 0
                split_rewards = {'human_dist_reward': 0, 'human_action_reward': 0, 'human_food_reward': 0, 'human_pref_reward' : 0,
                    'robot_dist_reward': 0, 'robot_action_reward': 0, 'robot_food_reward': 0, 'robot_pref_reward' : 0,
                    'vel': 0, 'force': 0, 'h_force' : 0, 'hit' : 0, 'food_v' : 0
                }
                

        if args.algo == "PPO":
            if args.social:
                if args.dempref and epoch % args.train_freq == 0:
                    mean_w, var_w = agent_robot.update(buf_robot, epoch, 1)
                else:
                    agent_robot.update(buf_robot, epoch, 1)
                if epoch % args.train_freq == 0: 
                    buf_robot.reset_training_buf()
                else:
                    buf_robot.organize_data()
            else:
                if args.dempref:
                    mean_w, var_w = agent_robot.update(buf_robot)
                else:
                    agent_robot.update(buf_robot)
            agent_human.update(buf_human)
            if args.dempref: 
                estimated_weights.update(mean_w.flatten())
                denormed_new_w = estimated_weights.get_denormed_weights()
                wandb.log({
                    "estimated_velocity_weight": denormed_new_w['C_v'],
                    "estimated_force_nontarget_weight":denormed_new_w['C_f'],
                    "estimated_high_forces_weight":denormed_new_w['C_hf'],
                    "estimated_food_hit_weight":denormed_new_w['C_fd'],
                    "food_velocities_weight":denormed_new_w['C_fdv']
                    })

        if epoch % args.log_interval == 0:
            end = time.time()
            print("Robot/Human updates {}, num timesteps {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(epoch, total_num_steps,
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards)))
            sys.stdout.flush()
        
        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and epoch % args.eval_interval == 0):
            if args.algo == "PPO":
                mean_value, success_rate = eval(args, agent_robot, agent_human, epoch, estimated_weights, success_rate)
            
            if mean_value > max_eval_mean_rewards:
                max_eval_mean_rewards = mean_value
                if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
                    os.makedirs(os.path.join(args.save_dir, args.exp_name))

                if args.algo == "PPO":
                    if args.social:
                        agent_robot.ac.save_weights(os.path.join(args.save_dir, args.exp_name+"/robot_"+str(epoch)+'_social'+'.pt'))
                        torch.save(agent_robot.ch_model.state_dict(), os.path.join(args.save_dir, args.exp_name+"/robot_"+str(epoch)+'_social'+'_pred.pth'))
                    else:
                        agent_robot.ac.save_weights(os.path.join(args.save_dir, args.exp_name+"/robot_"+str(epoch)+'.pt'))
                    agent_human.ac.save_weights(os.path.join(args.save_dir, args.exp_name+"/human_"+str(epoch)+'.pt'))
            elif epoch % 5 == 0:
                if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
                    os.makedirs(os.path.join(args.save_dir, args.exp_name))
                if args.algo == "PPO":
                    if args.social:
                        agent_robot.ac.save_weights(os.path.join(args.save_dir, args.exp_name+"/late_robot_"+str(epoch)+'_social'+'.pt'))
                        torch.save(agent_robot.ch_model.state_dict(), os.path.join(args.save_dir, args.exp_name+"/late_robot_"+str(epoch)+'_social'+'_pred.pth'))
                    else:
                        agent_robot.ac.save_weights(os.path.join(args.save_dir, args.exp_name+"/late_robot_"+str(epoch)+'.pt'))
                    agent_human.ac.save_weights(os.path.join(args.save_dir, args.exp_name+"/late_human_"+str(epoch)+'.pt'))
                
                print(f"recording late updates late_xxx_{epoch}.pt s")


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
    # for other tasks
    parser.add_argument('--dressing-force-weight', type=float, default=1.0)
    parser.add_argument('--high-pressures-weightt', type=float, default=1.0)

    #Utility module settings
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
    parser.add_argument('--save-dir', default='/data/add_disk0/jason/trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default=None,
                        help='Path name to saved policy checkpoint ("/data/add_disk0/jason/trained_models/")')
    parser.add_argument('--load-epoch', type=int, default=0) # modify for PPO policy versions
    parser.add_argument("--late", action="store_true", help="Set the flag to True") # loading late policy
    
    #social arguments
    parser.add_argument("--social", action="store_true", help="Set the flag to True")
    parser.add_argument("--baseline", action="store_true", help="Set the flag to True")
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--d-inner-g', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.000005)
    parser.add_argument('--k-levels', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20) # train social module for x epochs
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
    args.exp_name += "_trunc_ep"
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
        args.save_dir += 'social' + '/'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    if "Feeding" in text:
        pass
    else:
        args.save_dir += result.group(0) + '/'
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    args.save_dir += result.group(1) + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.dempref:
        args.save_dir += "Dempref_"
    
    if args.social:
        social = "_social"
    else:
        social = ''

    if args.load_policy_path is not None:
        wandb.init(
            project=project_name, 
            name=text+"_"+args.exp_name+social,
            config=vars(args),
            id='no4cv7l7', 
            resume='must')
    else:
        wandb.init(
        project=project_name,
        name=text+"_"+args.exp_name+social,
        config=vars(args)
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ppo(args)