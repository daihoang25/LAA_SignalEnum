# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:55:45 2022

@author: Dai Hoang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.signal import find_peaks
from additional_functions import *

# def training_func():
training_flag = False

# environment = 'inde' # Independent signals
# env_arr = ['inde', 'cohe'] # Coherent signals
# env_arr = ['inde'] # Coherent signals
# env = 'cohe'
env = 'inde'

model_type = 'CNN'
# model_type = 'ECNet'
# model_type = 'LogECNet'

feature_input = 'diag_spec' # ECNet takes eigenvalues as inputs
# feature_input = 'eigen' # ECNet takes eigenvalues as inputs


# SNRdB_arr = [-10, 0, 10, 20, 30]
# SNRdB_arr = np.arange(-10, 10+2.5, 2.5)

SNRdB_arr = [0]

# delta_SNRdB_arr = [0, 2, 8, 16]
delta_SNRdB_arr = [4]

# N_snapshots_arr = [10, 100, 1000, 10000] 
N_snapshots_arr = np.arange(10, 110, 10)
# N_snapshots_arr = [40]

# N_antenas_arr = np.arange(55, 95 + 2, 2)
N_antenas_arr = [65]

results = np.zeros([len(N_antenas_arr), len(SNRdB_arr), len(N_snapshots_arr)])

for idx_antennas in range(len(N_antenas_arr)):
    N_antenas = N_antenas_arr[idx_antennas] 
    for idx_SNRdB in range(len(SNRdB_arr)):
        SNRdB = SNRdB_arr[idx_SNRdB]
        SNRdB = int(SNRdB) if (abs(SNRdB) - abs(int(SNRdB)) == 0) else SNRdB
        
        for idx_snap in range(len(N_snapshots_arr)):
            N_snapshots = N_snapshots_arr[idx_snap]
            for delta_SNRdB in delta_SNRdB_arr:
                address = '%d_%d_%d'%(N_antenas, SNRdB, N_snapshots)
                # model path
                if training_flag:
                    train_DL(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type, env)
                # Fully utilized
                acc = test_DL(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type, env)
                # PSS technique
                # acc = test_DL_reuse(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type, env)
                results[idx_antennas, idx_SNRdB, idx_snap] = acc
 
if len(SNRdB_arr) > 1:
    savemat('./results/dec_prob_SNR_%s_%s_%s_%dsnaps.mat'%(env, model_type, feature_input, N_snapshots), {'results': results})
elif len(N_snapshots_arr) > 1:
    savemat('./results/dec_prob_snapshots_%s_%s_%s_%sdB.mat'%(env, model_type, feature_input, str(SNRdB)), {'results': results})
