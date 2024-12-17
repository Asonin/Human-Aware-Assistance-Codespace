import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import gym
import time
import torch.nn as nn
from torch.distributions.normal import Normal

from assistive_gym.ppo_new.Anticipation.lib.model.models import Transformer, Discriminator
from assistive_gym.ppo_new.Anticipation.lib.utils.utils import *
from assistive_gym.ppo_new import sampling
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import os
import yaml
from pprint import pprint
from easydict import EasyDict
from tqdm import tqdm
from assistive_gym.ppo_new.Anticipation.lib.utils.dataset import MPMotion, MPMotion_Inference
import torch_dct as dct
import wandb
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def compute_loss_pi(data, ac, clip_ratio):
            obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
            act = act.to(device) 
            if 'social' in data.keys():
                social_pred = data['social']
                thorough_obs = torch.cat((obs, social_pred), dim=1).to(device)
                pi, logp = ac.pi(thorough_obs, act)
            else:
                obs = obs.to(device)
                pi, logp = ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old.to(device))
            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv.to(device)  
            loss_pi = -(torch.min(ratio * adv.to(device), clip_adv)).mean()  

            approx_kl = (logp_old.to(device) - logp).mean().item()  
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

def compute_loss_v(data, ac):

            obs, ret = data['obs'], data['ret']
            ret = ret.to(device) 
            if 'social' in data.keys():
                social_pred = data['social']
                thorough_obs = torch.cat((obs, social_pred), dim=1).to(device)  
                return ((ac.v(thorough_obs) - ret.to(device))**2).mean()  
            else:
                obs = obs.to(device)  
                return ((ac.v(obs) - ret.to(device))**2).mean()  


class EstimatedWeights:
    def __init__(self, C_v_gt, C_f_gt, C_hf_gt, C_fd_gt, C_fdv_gt):
        self.C_v_gt = C_v_gt
        self.C_f_gt = C_f_gt
        self.C_hf_gt = C_hf_gt
        self.C_fd_gt = C_fd_gt
        self.C_fdv_gt = C_fdv_gt

        self.gt_weights = np.array([self.C_v_gt, self.C_f_gt, self.C_hf_gt, self.C_fd_gt, self.C_fdv_gt])

        min_weight = self.gt_weights.min()
        max_weight = self.gt_weights.max()
        initial_weights = np.random.uniform(min_weight, max_weight, size=self.gt_weights.shape)

        self.C_v, self.C_f, self.C_hf, self.C_fd, self.C_fdv = initial_weights
        self.normalize_weights()

        self.gt_normalized_weights = self.normalize_gt_weights()

    def normalize_weights(self):
        """Normalize initial weights"""
        weights = np.array([self.C_v, self.C_f, self.C_hf, self.C_fd, self.C_fdv])
        log_weights = np.log(weights + 1e-8) 

        min_log = np.log(self.gt_weights.min() + 1e-8)
        max_log = np.log(self.gt_weights.max() + 1e-8)

        low, high = 0.1, 1.0

        self.C_v, self.C_f, self.C_hf, self.C_fd, self.C_fdv = low + (log_weights - min_log) * (high - low) / (max_log - min_log)


    def normalize_gt_weights(self):
        log_gt_weights = np.log(self.gt_weights + 1e-8) 

        min_log = log_gt_weights.min()
        max_log = log_gt_weights.max()

        low, high = 0.1, 1.0

        return low + (log_gt_weights - min_log) * (high - low) / (max_log - min_log)


    def denormalize_weights(self):
        normalized_weights = np.array([self.C_v, self.C_f, self.C_hf, self.C_fd, self.C_fdv])

        log_min = np.log(self.gt_weights.min() + 1e-8)
        log_max = np.log(self.gt_weights.max() + 1e-8)

        low, high = 0.1, 1.0

        original_log_weights = log_min + (normalized_weights - low) * (log_max - log_min) / (high - low)

        return np.exp(original_log_weights)

    def update(self, normalized_values):
        self.C_v, self.C_f, self.C_hf, self.C_fd, self.C_fdv = abs(normalized_values)

    def get_normalized_weights(self):
        return {
            'C_v': self.C_v,
            'C_f': self.C_f,
            'C_hf': self.C_hf,
            'C_fd': self.C_fd,
            'C_fdv': self.C_fdv
        }

    def get_denormed_weights(self):
        de = self.denormalize_weights()
        return {
            'C_v': de[0],
            'C_f': de[1],
            'C_hf': de[2],
            'C_fd': de[3],
            'C_fdv': de[4]
        }
    
    def get_normalized_gt_weights(self):
        return{
            'C_v':  self.gt_normalized_weights[0],
            'C_f':  self.gt_normalized_weights[1],
            'C_hf':  self.gt_normalized_weights[2],
            'C_fd':  self.gt_normalized_weights[3],
            'C_fdv':  self.gt_normalized_weights[4]
        }




class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        act = act.to(device)  
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, social_space=None, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        if social_space is not None:
            social_dim = social_space
        else:
            social_dim=0
        self.pi = MLPGaussianActor(obs_dim+social_dim, action_space.shape[0], hidden_sizes, activation)

        self.v  = MLPCritic(obs_dim+social_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            obs = obs.to(device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))
        print("loaded ckpt")

class PPO_Social(object):

    def __init__(self,
    state_dim,
	action_dim,
    args,
    load_policy=None,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    clip_ratio=0.2,
    target_kl=0.01,
    train_v_iters=80
    ):  
        self.args = args

        # This should also include a social_dim. The social dim is added in ac logic
        self.ac = MLPActorCritic(state_dim, action_dim, social_space=self.args.social_space).to(device)
        self.device = 'cuda'
        self.ch_model = Transformer(input_dim = self.args.input_dim, d_word_vec=args.d_model, d_model=args.d_model, d_inner=args.d_inner_g,
            n_layers=3, n_head=8, d_k=64, d_v=64, k_levels=args.k_levels, share_d=True, dropout=args.dropout, device=self.device).to(self.device)
        
        if load_policy is not None:
            self.ac.load_weights(load_policy+'.pt')
            checkpoint = torch.load(load_policy+'_pred.pth', map_location=lambda storage, loc: storage)
            self.ch_model.load_state_dict(checkpoint, strict=True)
            print("ckpt loaded")
            
        

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.train_pi_iters = train_pi_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_v_iters = train_v_iters

        
        params = [
            {"params": self.ch_model.parameters(), "lr": args.lr}
        ]
        self.ch_optimizer = Adam(params)
        self.ch_scheduler = StepLR(self.ch_optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

        self.epoch = args.epochs


    def social_inference(self, padded_human, poses_robot):
        
        test_data = np.array([np.concatenate((np.expand_dims(padded_human, axis=0),np.expand_dims(poses_robot, axis=0)), axis=0)])
        test_dataset = MPMotion_Inference(test_data, concat_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            self.ch_model.eval()
            for data in test_dataloader:
                input_seq = data
                input_seq=torch.tensor(input_seq,dtype=torch.float32).to(self.device) 
                use=[input_seq.shape[1]]
                input_=input_seq.view(-1,10,input_seq.shape[-1])
                input_ = dct.dct(input_)

                rec_ = self.ch_model.forward(input_[:,1:10,:]-input_[:,:9,:],dct.idct(input_[:,-1:,:]),input_seq,use)
                rec = dct.idct(rec_[-1])
                results = input_seq[0][:,-1:,:]
                for i in range(1,11):
                    results = torch.cat([results,input_seq[0][:,-1:,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
                results = results[:,1:,:] # this is target pose

        return results[0][:,:self.args.social_space] 


    def update(self, buf, e, update_flag):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, self.ac, self.clip_ratio)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, self.ac).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, self.ac, self.clip_ratio)
            kl = np.sum(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()


        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, self.ac)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']


        if e % self.args.train_freq != 0 : # only do training for 5 epochs
            return

        # train social module
        training_data = MPMotion(buf.training_buf, concat_last=True)
        dataloader = torch.utils.data.DataLoader(training_data, batch_size=self.args.batch_size, shuffle=True)
        
        min_erorr = 100000
        torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(self.epoch):    
            print('Training epoch', epoch)
            self.ch_model.train()

            if self.ch_model.training:
                print("Model is in training mode")
            else:
                print("Model is in evaluation mode")
        
            losses_all = []
            losses_recon = []
            losses_sum = AverageMeter()
            for k in range(self.args.k_levels + 1):
                losses_all.append(AverageMeter())
                losses_recon.append(AverageMeter())

            for j, data in tqdm(enumerate(dataloader)):

                use = None
                input_seq, output_seq = data 
                B = input_seq.shape[0] # batch size
                input_seq = torch.tensor(input_seq,dtype=torch.float32).to(self.device) 
                output_seq = torch.tensor(output_seq,dtype=torch.float32).to(self.device) 
                input_ = input_seq.view(-1,10,input_seq.shape[-1]) 
                output_ = output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1, input_seq.shape[-1])
                
                input_dct = dct.dct(input_)
                rec_ = self.ch_model.forward(input_dct[:,1:10,:]-input_dct[:,:9,:], dct.idct(input_dct[:,-1:,:]), input_seq, use) 

                loss_sum = torch.tensor(0).to(self.device)

                for k in range(1, self.args.k_levels + 1):
                    bc = (k==self.args.k_levels)
                    gail = False
                    rec = dct.idct(rec_[k])                         
                    if bc:
                        loss_l2 = torch.mean((rec[:,:,:]-(output_[:,1:11,:]-output_[:,:10,:]))**2) 
                        loss_recon = loss_l2
                        losses_recon[k].update(loss_recon.item(), B)
                    else:
                        loss_recon = torch.tensor(0).to(self.device)

                    
                    loss_all = self.args.lambda_recon * loss_recon 
                    losses_all[k].update(loss_all.item(), B)
                    loss_sum = loss_sum + loss_all

                self.ch_optimizer.zero_grad()
                loss_sum.backward()
                self.ch_optimizer.step()
                self.ch_scheduler.step()
                losses_sum.update(loss_sum.item(), B)

            stats = {}
            for k in range(self.args.k_levels + 1):
                prefix = 'train/level%d/' % k
                stats[prefix + 'loss_recon'] = losses_recon[k].avg
                stats[prefix + 'loss_all'] = losses_all[k].avg
            stats['train/loss_sum'] = losses_sum.avg
            stats['epoch'] = epoch


class PPO_Social_Dempref(object):

    def __init__(self,
    pref_dim,
    state_dim,
	action_dim,
    args,
    load_policy=None,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    clip_ratio=0.2,
    target_kl=0.01,
    train_v_iters=80
    ):  
        self.args = args

        # This should also include a social_dim. The social dim is added in ac logic
        self.ac = MLPActorCritic(state_dim, action_dim, social_space=self.args.social_space).to(device)
        self.device = 'cuda'
        self.ch_model = Transformer(input_dim = self.args.input_dim, d_word_vec=args.d_model, d_model=args.d_model, d_inner=args.d_inner_g,
            n_layers=3, n_head=8, d_k=64, d_v=64, k_levels=args.k_levels, share_d=True, dropout=args.dropout, device=self.device).to(self.device)
        
        if load_policy is not None:
            self.ac.load_weights(load_policy+'.pt')
            checkpoint = torch.load(load_policy+'_pred.pth', map_location=lambda storage, loc: storage)
            self.ch_model.load_state_dict(checkpoint, strict=True)
            print("ckpt loaded")
            
        

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.train_pi_iters = train_pi_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_v_iters = train_v_iters

        
        params = [
            {"params": self.ch_model.parameters(), "lr": args.lr}
        ]
        self.ch_optimizer = Adam(params)
        self.ch_scheduler = StepLR(self.ch_optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

        self.epoch = args.epochs

        self.n_samples_summ = args.n_samples_summ #default 50000
        if args.continuous_sampling:
            self.sampler = sampling.Sampler_Continuous_merge(dim_features=pref_dim, alpha=args.merge_alpha)
        else:
            self.sampler = sampling.Sampler(dim_features=pref_dim)
        


    def social_inference(self, padded_human, poses_robot):
        
        test_data = np.array([np.concatenate((np.expand_dims(padded_human, axis=0),np.expand_dims(poses_robot, axis=0)), axis=0)])
        test_dataset = MPMotion_Inference(test_data, concat_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            self.ch_model.eval()
            for data in test_dataloader:
                input_seq = data
                input_seq=torch.tensor(input_seq,dtype=torch.float32).to(self.device) 
                use=[input_seq.shape[1]]
                input_=input_seq.view(-1,10,input_seq.shape[-1])
                input_ = dct.dct(input_)

                rec_ = self.ch_model.forward(input_[:,1:10,:]-input_[:,:9,:],dct.idct(input_[:,-1:,:]),input_seq,use)
                rec = dct.idct(rec_[-1])
                results = input_seq[0][:,-1:,:]
                for i in range(1,11):
                    results = torch.cat([results,input_seq[0][:,-1:,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
                results = results[:,1:,:] # this is target pose
                
        return results[0][:,:self.args.social_space]


    def update(self, buf, e, update_flag):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, self.ac, self.clip_ratio)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, self.ac).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, self.ac, self.clip_ratio)
            kl = np.sum(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()


        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, self.ac)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']


        if e % self.args.train_freq != 0 : # only do training for 5 epochs
            return

        # train social module
        training_data = MPMotion(buf.training_buf, concat_last=True)
        dataloader = torch.utils.data.DataLoader(training_data, batch_size=self.args.batch_size, shuffle=True)
        
        min_erorr = 100000
        torch.autograd.set_detect_anomaly(True)
        
        for epoch in range(self.epoch):    
            print('Training epoch', epoch)
            self.ch_model.train()

            if self.ch_model.training:
                print("Model is in training mode")
            else:
                print("Model is in evaluation mode")
        
            losses_all = []
            losses_recon = []
            losses_sum = AverageMeter()
            for k in range(self.args.k_levels + 1):
                losses_all.append(AverageMeter())
                losses_recon.append(AverageMeter())

            for j, d in tqdm(enumerate(dataloader)):

                use = None
                input_seq, output_seq = d
                B = input_seq.shape[0] # batch size
                input_seq = torch.tensor(input_seq,dtype=torch.float32).to(self.device) 
                output_seq = torch.tensor(output_seq,dtype=torch.float32).to(self.device) 
                input_ = input_seq.view(-1,10,input_seq.shape[-1]) 
                output_ = output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1, input_seq.shape[-1])
                
                input_dct = dct.dct(input_)
                rec_ = self.ch_model.forward(input_dct[:,1:10,:]-input_dct[:,:9,:], dct.idct(input_dct[:,-1:,:]), input_seq, use) 

                loss_sum = torch.tensor(0).to(self.device)
                
                for k in range(1, self.args.k_levels + 1):
                    bc = (k==self.args.k_levels)
                    gail = False
                    rec = dct.idct(rec_[k])                        
                    if bc:
                        loss_l2 = torch.mean((rec[:,:,:]-(output_[:,1:11,:]-output_[:,:10,:]))**2) 
                        loss_recon = loss_l2
                        losses_recon[k].update(loss_recon.item(), B)
                    else:
                        loss_recon = torch.tensor(0).to(self.device)

                    
                    loss_all = self.args.lambda_recon * loss_recon 
                    losses_all[k].update(loss_all.item(), B)
                    loss_sum = loss_sum + loss_all

                self.ch_optimizer.zero_grad()
                loss_sum.backward()
                self.ch_optimizer.step()
                self.ch_scheduler.step()
                losses_sum.update(loss_sum.item(), B)

            stats = {}
            for k in range(self.args.k_levels + 1):
                prefix = 'train/level%d/' % k
                stats[prefix + 'loss_recon'] = losses_recon[k].avg
                stats[prefix + 'loss_all'] = losses_all[k].avg
            stats['train/loss_sum'] = losses_sum.avg
            stats['epoch'] = epoch
            
        self.sampler.load_demo(data["pref_entries"]) # data returned as a dict
        samples = self.sampler.sample(N=self.n_samples_summ)
        mean_w = np.mean(samples, axis=0)
        mean_w = mean_w / np.linalg.norm(mean_w)
        var_w = np.var(samples, axis=0)
        return mean_w, var_w


class PPO(object):
    def __init__(self,
    state_dim,
	action_dim,
    load_policy=None,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    clip_ratio=0.2,
    target_kl=0.01,
    train_v_iters=80
    ):

        # could be put on CUDA
        self.ac = MLPActorCritic(state_dim, action_dim).to(device)
        if load_policy is not None:
            self.ac.load_weights(load_policy+'.pt')
        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.train_pi_iters = train_pi_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_v_iters = train_v_iters


    # need to modify update function
    def update(self, buf):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, self.ac, self.clip_ratio)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, self.ac).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, self.ac, self.clip_ratio)
            kl = np.sum(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()


        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, self.ac)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        
            
class PPO_Dempref(object):
    def __init__(self,
    pref_dim,
    state_dim,
	action_dim,
    load_policy=None,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    clip_ratio=0.2,
    target_kl=0.01,
    train_v_iters=80
    ):

        # could be put on CUDA
        self.ac = MLPActorCritic(state_dim, action_dim).to(device)
        if load_policy is not None:
            self.ac.load_weights(load_policy+'.pt')
        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.train_pi_iters = train_pi_iters
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_v_iters = train_v_iters
        self.n_samples_summ = 50000
        self.sampler = sampling.Sampler(dim_features=pref_dim)


    def update(self, buf):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, self.ac, self.clip_ratio)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, self.ac).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, self.ac, self.clip_ratio)
            kl = np.sum(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_optimizer.step()


        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, self.ac)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        self.sampler.load_demo(data["pref_entries"]) # data returned as a dict
        samples = self.sampler.sample(N=self.n_samples_summ)
        mean_w = np.mean(samples, axis=0)
        mean_w = mean_w / np.linalg.norm(mean_w)
        var_w = np.var(samples, axis=0)
        return mean_w, var_w