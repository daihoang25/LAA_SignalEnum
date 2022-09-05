# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:33:10 2022

@author: Dai Hoang
"""
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
from additional_functions import *

SNRdB = 0
delta_SNRdB = 4
N_snapshots = 40
N_antenas = 65

env = 'inde'
feature_input = 'diag_spec'
model_type = 'CNN'

# SNRdB_arr = np.arange(-10, 10+2.5, 2.5)
SNRdB_arr = [0]
N_snapshots_arr = np.arange(10, 100+10, 10)
# N_snapshots_arr = [40]

for idx_SNRdB in range(len(SNRdB_arr)):
    SNRdB = SNRdB_arr[idx_SNRdB]
    SNRdB = int(SNRdB) if (abs(SNRdB) - abs(int(SNRdB)) == 0) else SNRdB
    for idx_snap in range(len(N_snapshots_arr)):
        N_snapshots = N_snapshots_arr[idx_snap]
        
        _, _, testset_path = path_reader(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type, env, training=False)
        data = np.load(testset_path, allow_pickle=True).item()
        X_src, y_src = data[feature_input], data['y_Nsignals']
        
        savemat('./dataset_CFAR/%s/testset_Nsignals_%s_dB_%d_delta_%d_antenas_%d_snap.mat'\
                %(env, str(SNRdB), delta_SNRdB, N_antenas, N_snapshots), {'diag_spec': X_src, 'onehot': y_src})


