# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 01:24:23 2021

@author: Dai Hoang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.signal import find_peaks
from additional_functions import *

training_flag = True

# environment = 'inde' # Independent signals
# env_arr = ['inde', 'cohe'] # Coherent signals
# env_arr = ['inde'] # Coherent signals
env_arr = ['inde', 'cohe'] # Coherent signals

# model_type = 'ECNet'
model_type = 'PSCNet'
# model_type = 'ECNet'
# model_type = 'LogECNet'


feature_input = 'diag_spec' # ECNet takes eigenvalues as inputs
# feature_input = 'eigen' # ECNet takes eigenvalues as inputs


# SNRdB_arr = np.arange(-5, 10+2.5, 2.5)
SNRdB_arr = [0]

# delta_SNRdB_arr = [0, 2, 8, 16]
delta_SNRdB_arr = [4]

# N_snapshots_arr = np.arange(10, 110, 10)
N_snapshots_arr = [40]   

# N_antenas_arr = np.arange(55, 119 + 2, 2)
N_antenas_arr = [65]

results = np.zeros([len(N_antenas_arr), len(SNRdB_arr), len(N_snapshots_arr)])

for env in env_arr:
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
                    acc = test_DL(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type, env)
                    results[idx_antennas, idx_SNRdB, idx_snap] = acc
    if len(SNRdB_arr) > 1:
        savemat('./results/dec_prob_SNR_%s_%s_%s.mat'%(env, model_type, feature_input), {'results': results})
    elif len(N_snapshots_arr) > 1:
        savemat('./results/dec_prob_snapshots_%s_%s_%s.mat'%(env, model_type, feature_input), {'results': results})

