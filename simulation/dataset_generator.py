# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:30:52 2020

@author: DaiHoang
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.utils import np_utils
#from parameters import doa_samples
from scipy.linalg import cholesky, ldl, toeplitz
from scipy.signal import find_peaks
from tqdm import tqdm

c = 3e8
fc = 1e9
lamda = c/fc
Dy = 30*lamda
alpha = 1 # Dy*Dz/lamda^2
resolution = 0.1
doa_samples = np.arange(-60, 60, resolution)

def normalize(spec):
    s_max, s_min = np.max(spec), np.min(spec)
    return (spec - s_min)/(s_max - s_min)

def attenuation_coef(var=0.2):
    sign_real = 1 if np.random.random() < 0.5 else -1
    sign_imag = 1 if np.random.random() < 0.5 else -1
    return (sign_real*np.random.normal(0, np.sqrt(var)) + sign_imag*1j*np.random.normal(0, np.sqrt(var)))/np.sqrt(2)

def DoAs_samples(N_signals=4, DoAs_spacing=1):
    DoAs_truth = []
    while len(DoAs_truth) < N_signals: # to ensure no same DoA in this list
        DoA_tmp = np.random.uniform(-60, 60)
        # Assign the very first DoA value or only append DoAs out of spacing
        if len(DoAs_truth) == 0 or np.sum(np.abs(np.array(DoAs_truth) - DoA_tmp) > DoAs_spacing) == len(DoAs_truth):
            DoAs_truth.append(DoA_tmp)
    return DoAs_truth

def sinc_function(x):
    return np.sin(np.pi*x)/(np.pi*x)

def DoA_onehot(DoAs, resolution):
    DoA_samples = np.arange(-60, 60, resolution)
    spec_truth = np.zeros(len(DoA_samples))
    for i in range(len(DoA_samples)):
        for DoA in DoAs:
            if np.abs(DoA_samples[i] - DoA) < 1e-2: 
                spec_truth[i] = 1
    return spec_truth

def array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, Dy_lamda, DOAs, N_coherent=None):
    N_signals = len(DOAs)
    M = N_antenas//2
    if N_coherent is None: N_coherent = np.random.randint(0, high=N_signals+1)
    # position of array
    
    snrdB = np.random.uniform(low=SNRdB - delta_SNRdB/2, high=SNRdB + delta_SNRdB/2);
    array_signal = 0
    S_hist = []
    for sig in range(N_signals):
        tmp = np.expand_dims(np.array(np.arange(-M, M+1)), axis=-1) - Dy_lamda*np.sin(DOAs[sig]*np.pi/180)
        A = np.sqrt(alpha)*sinc_function(tmp) # steering vectors'matrix, size = (N_antenas, N_signals) = (N_antenas, 1)
        if sig <= N_coherent:
            if sig == 0: 
                S_0 = np.random.normal(size=(1, N_snapshots)) + 1j*np.random.normal(size=(1, N_snapshots)) # path-gain, size = (N_signals, N_snapshots) = (1, N_snapshots)
                S = 1*S_0
            else:
                S = attenuation_coef()*S_0
        else:
            S = np.random.normal(size=(1, N_snapshots)) + 1j*np.random.normal(size=(1, N_snapshots)) # path-gain, size = (N_signals, N_snapshots) = (1, N_snapshots)
        S = 10**(snrdB/20)*S/np.sqrt(2)
        array_signal += A.dot(S) 
    # noise matrix
    N = (np.random.normal(size=(N_antenas, N_snapshots)) + 1j*np.random.normal(size=(N_antenas, N_snapshots)))/np.sqrt(2) # size = (N_antenas, N_snapshot)
    X = array_signal + N    
    return X

def data_eigens(R):
    eig_values, eig_vectors = np.linalg.eig(R)
    eig_values = eig_values.real
    # Decending sort
    eig_values = -np.sort(-np.real(eig_values)) 
    eig_vectors = eig_vectors[:, np.argsort(-np.real(eig_values))]
    return eig_values, eig_vectors

def dataset_Nsignals(SNRdB, delta_SNRdB, N_antenas, N_snapshots, Dy_lamda, N_signals_min, N_signals_max, environment, N_samples=10000, test_flag=False):
    delta_DoA = 10
    data = {}
    data['eigen'] = []
    # data['F_spectrum'] = []
    data['diag_spec'] = []
    data['y_Nsignals'] = []
    data['DoAs'] = []

    for rep in tqdm(range(N_samples)):
        N_rep = 1 if test_flag else 5
        for rep in range(N_rep):
            N_signals = np.random.randint(N_signals_min, N_signals_max+1)
            DOAs = DoAs_samples(N_signals, delta_DoA)
            N_coherent = np.random.randint(0, N_signals) if environment=='cohe' else 0
            X = array_signal_calculator(SNRdB, delta_SNRdB, N_antenas, N_snapshots, Dy_lamda, DOAs, N_coherent)
            R = X.dot(np.matrix.getH(X))/N_snapshots
            if environment == 'inde':
                eigen, _ = data_eigens(R)
                data['eigen'].append(eigen)
            
            data['diag_spec'].append(np.real(np.diag(R)))
            # Output for N_signals detection and DoA estimation
            data['y_Nsignals'].append(np_utils.to_categorical(N_signals, N_signals_max+1))
            data['DoAs'].append(DOAs)
    return data

# Generate dataset with various SNRs
env_arr = ['inde', 'cohe']
delta_SNRdB_arr = [0, 2, 8, 16]
N_antenas_arr = np.arange(55, 117, 2)
Dy_lamda_arr = [30, 40, 50]
SNRdB_arr = np.arange(-10, 10+2.5, 2.5)

N_snapshots_arr = [40]
# N_snapshots_arr = np.arange(10, 110, 10)
SNRdB_arr = [0]

for Dy_lamda in Dy_lamda_arr:
    for env in env_arr:
        for SNRdB in SNRdB_arr:
            SNRdB = int(SNRdB) if (abs(SNRdB) - abs(int(SNRdB)) == 0) else SNRdB
            for N_antenas in N_antenas_arr:
                for N_snapshots in N_snapshots_arr:
                    for delta_SNRdB in delta_SNRdB_arr:
                        # Data for training
                        print('\n<=========== Generating dataset for Nsignals detection: %s environment - %d dB - %d antennas - %d snapshots ============>'\
                              %(env, SNRdB, N_antenas, N_snapshots))
                            
                        dataset_path = './dataset_{}/train/dataset_Nsignals_{}_{}_{}_dB_{}_delta_{}_antenas_{}_snap.npy'\
                                        .format(env, r'Dy%lamda', Dy_lamda, str(SNRdB), delta_SNRdB, N_antenas, N_snapshots)
                        data = dataset_Nsignals(SNRdB, delta_SNRdB, N_antenas, N_snapshots, Dy_lamda, N_signals_min=1, N_signals_max=8, \
                                                environment=env, N_samples=10000)
                        np.save(dataset_path, data)
                        # Data for testing
                        testset_path = './dataset_{}/test/testset_Nsignals_{}_{}_{}_dB_{}_delta_{}_antenas_{}_snap.npy'\
                                        .format(env, 'Dy%lamda', Dy_lamda, str(SNRdB), delta_SNRdB, N_antenas, N_snapshots)
                        data = dataset_Nsignals(SNRdB, delta_SNRdB, N_antenas, N_snapshots, Dy_lamda, N_signals_min=1, N_signals_max=8, \
                                                environment=env, N_samples=10000, test_flag=True)
                        np.save(testset_path, data)
        

    