import torch.utils.data as data
import torch
import numpy as np

class MPMotion(data.Dataset):
    def __init__(self, data, in_len = 10, max_len = 20, concat_last = False):
        self.data = data[1:]
        # B, M, T, J = motion_data.shape
        print('Loading data:', self.data.shape)
        self.len = self.data.shape[0]
        self.max_len = max_len
        self.in_len = in_len
        self.concat_last = concat_last
            
    def __getitem__(self, index):
        input_seq=self.data[index][:,:self.in_len,:]             
        output_seq=self.data[index][:,self.in_len:self.max_len,:]
        if self.concat_last:
            last_input=input_seq[:,-1:,:]
            output_seq = np.concatenate([last_input, output_seq], axis=1)
        return input_seq, output_seq
        
    def __len__(self):
        return self.len


class MPMotion_Inference(data.Dataset):
    def __init__(self, data, in_len = 25,concat_last = False):
        self.data = data
        # B, M, T, J = motion_data.shape
        # print('Loading data:', self.data.shape)
        self.len = self.data.shape[0]
        self.in_len = in_len
        self.concat_last = concat_last
            
    def __getitem__(self, index):
        input_seq=self.data[index][:,:self.in_len,:]             
        return input_seq
        
    def __len__(self):
        return self.len



class MPMotion_trial(data.Dataset):
    def __init__(self, data, in_len = 10, max_len = 20, concat_last = False):
        self.data = np.load(data, allow_pickle=True)
        # self.data = data
        # B, M, T, J = motion_data.shape
        print('Loading data:', self.data.shape)
        self.len = self.data.shape[0]
        self.max_len = max_len
        self.in_len = in_len
        self.concat_last = concat_last
            
    def __getitem__(self, index):
        input_seq=self.data[index][:,:self.in_len,:]             
        output_seq=self.data[index][:,self.in_len:self.max_len,:]
        if self.concat_last:
            last_input=input_seq[:,-1:,:]
            output_seq = np.concatenate([last_input, output_seq], axis=1)
        return input_seq, output_seq
        
    def __len__(self):
        return self.len