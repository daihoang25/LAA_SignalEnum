# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:08:19 2021

@author: Dai Hoang
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.signal import find_peaks

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, Flatten, Dropout, LeakyReLU, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate, Permute, Subtract, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

def data_shuffle(X_train, y_train, N_samples=None):
    if N_samples is None:
        N_samples = int(len(X_train))
    shuf_idx = np.random.permutation(X_train.shape[0])
    if N_samples is not None:
        if type(N_samples) == int:
            shuf_idx = shuf_idx[:N_samples]
        elif type(N_samples) == float:
            shuf_idx = shuf_idx[:int(len(X_train)*N_samples)]
    return X_train[shuf_idx], y_train[shuf_idx]

def normalize(spec):
    s_max, s_min = np.max(spec), np.min(spec)
    return (spec - s_min)/(s_max - s_min)

def dataset_preparation(data, SNRdB, feature_input, normalized_input=False, data_shuffling=True, model_type=None, M_=None):   
    print('<=========== Loading dataset (SNR = %sdB) ============>'%str(SNRdB))
    X_train, y_train = np.array(data[feature_input]), np.array(data['y_Nsignals'])
    M = X_train.shape[1] - 1
    if data_shuffling:
        print('<=========== Shuffling dataset (SNR = %sdB) ============>'%str(SNRdB))
        X_train, y_train = data_shuffle(X_train, y_train)
        
    if M_ is not None: # PSS technique
        X_train = X_train[:, (-M_+M):(M_+M+1)]

    N_samples, N_features = X_train.shape[0], X_train.shape[0]
    
    if model_type == 'CNN' or model_type == 'PSCNet': X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])

    if normalized_input:
        for i in range(N_samples):
            X_train[i] = normalize(X_train[i])
            
    return X_train, y_train

def model_path_reader(N_antenas, SNRdB, delta_SNRdB, N_snapshots, Dy_lamda=30, model_type='CNN', environment='cohe', reuse_flag=True):
    model_path = './saved_models_{}/{}_Nsignals_'.format(environment, model_type) + r'Dy%lamda' 
    if reuse_flag:
        M_ = int(min(np.ceil(Dy_lamda), np.floor(N_antenas//2)))
        M0_ = 2*M_ + 1

        model_path_ = model_path + '_{}_{}_antenas_{}_dB_{}_delta_{}_snap/cp.ckpt'.format(Dy_lamda, M0_, SNRdB, delta_SNRdB, N_snapshots)
        transfer_path = model_path + '_{}_{}_antenas_{}_dB_{}_delta_{}_snap_transfer/cp.ckpt'.format(Dy_lamda, N_antenas, SNRdB, delta_SNRdB, N_snapshots)
        checkpoint_path_transfer = os.path.abspath(transfer_path)
        return model_path_, transfer_path, checkpoint_path_transfer

    else:   
        model_path += '_{}_{}_antenas_{}_dB_{}_delta_{}_snap/cp.ckpt'.format(Dy_lamda, N_antenas, SNRdB, delta_SNRdB, N_snapshots)
        checkpoint_path_Nsignals = os.path.abspath(model_path)
        return model_path, checkpoint_path_Nsignals

def ECNet(input_shape, output_shape, N_hidden=None):
    drop_rate = 0.01    
    # if N_hidden is None: N_hidden = [16, 32, 16, 8]
    M = input_shape[0]
    if N_hidden is None: N_hidden = [int(5*M/4), int(5*M/4), int(5*M/4), int(5*M/4)]
    inputs = Input(shape=input_shape, name='Nsignals_input') 
    act_func = 'selu'
    #act_func = 'relu'
    x = inputs
    for h in range(len(N_hidden)):
        x = Dense(units=N_hidden[h], activation=act_func, name='FC_%d'%h)(x)
        x = Dropout(drop_rate)(x)
    
    outputs = Dense(units=output_shape, activation='softmax', name='FC_Nsignals_4')(x)
    return Model(inputs, outputs)

def PSCNet(input_shape, output_shape, N_filters=None, N_conv=None, N_hidden=None):
    drop_rate = 0.01    
    act_func = 'selu'
    M = input_shape[0]

    if N_filters is None: N_filters = 64
    if N_conv is None: N_conv = 2
    if N_hidden is None: N_hidden = [int(M/2), int(M/2), int(M/2), int(M/2)]

    inputs = Input(shape=input_shape, name='Nsignals_input') 
    x = inputs
    for d in range(N_conv):
        x = Conv1D(filters=N_filters, kernel_size=3, strides=1, kernel_initializer='orthogonal',
                   use_bias=False, padding='same', name='conv_%d'%d)(x)
        # x = BatchNormalization(momentum=0.0, epsilon=0.0001, name='BN_%d'%d)(x)
        x = Activation(act_func, name='activ_%d'%d)(x)

    DNN_inputs = Flatten(name='flatten')(x)
    x = DNN_inputs
    for h in range(len(N_hidden)):
        x = Dense(units=N_hidden[h], activation=act_func, name='FC_%d'%h)(x)
        x = Dropout(drop_rate)(x)
    outputs = Dense(units=output_shape, activation='softmax', name='FC_Nsignals_4')(x)
    return Model(inputs, outputs)

def accuracy(y_true, y_pred):
    pred = np.argmax(y_pred, axis=1)
    true = np.argmax(y_true, axis=1)
    return np.sum(pred==true)/len(pred)

def path_reader(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type=None, environment='cohe', training=True):
    # SNRdB = int(SNRdB) if (abs(SNRdB) - abs(int(SNRdB)) == 0) else SNRdB
    if training == True:
        data_folder = './dataset_%s/train/dataset'%environment
    else:
        data_folder = './dataset_%s/test/testset'%environment
        
    dataset_path = data_folder + '_Nsignals_%s_dB_%d_delta_%d_antenas_%d_snap.npy'\
                    %(str(SNRdB), delta_SNRdB, N_antenas, N_snapshots)
    
    if model_type is not None:
        # tmp = 10
        # SNRdB_tmp = tmp if (SNRdB <= tmp and feature_input == 'diag_spec' and model_type == 'CNN' and training == 'False')\
        #                    else SNRdB
        SNRdB_tmp = SNRdB
        
        model_path = './saved_models_%s/%s_Nsignals_%d_antenas_%d_dB_%d_delta_%d_snap/cp.ckpt'\
            %(environment, model_type, N_antenas, SNRdB_tmp, delta_SNRdB, N_snapshots)
        checkpoint_path_Nsignals = os.path.abspath(model_path)
    
        return model_path, checkpoint_path_Nsignals, dataset_path
    else:
        return dataset_path

def train_DL(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type='DNN', environment='inde'):
    model_path, checkpoint_path_Nsignals, dataset_path = path_reader(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type, environment)
    
    checkpoint_folder_Nsignals = os.path.dirname(checkpoint_path_Nsignals)
    cp_callback_Nsignals = ModelCheckpoint(filepath=checkpoint_path_Nsignals, verbose=0, 
                              save_weights_only=True, save_best_only=True, period=1)
        
    print('<=========== Loading dataset (SNR = %sdB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(str(SNRdB), delta_SNRdB, N_antenas, N_snapshots))
    data = np.load(dataset_path, allow_pickle=True).item()

    #X_train = X_train/np.sum(X_train, axis=1)
    print('<=========== Training %s model (SNR = %sdB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(model_type, str(SNRdB), delta_SNRdB, N_antenas, N_snapshots))
    loss_func = CategoricalCrossentropy(from_logits=False)
    X_train, y_train = dataset_preparation(data, SNRdB, feature_input, normalized_input=False, data_shuffling=True, \
                                       model_type=model_type)
    input_shape, output_shape = X_train.shape[1:], y_train.shape[1:][0]
    if model_type == 'DNN' or model_type == 'ECNet':
        model = ECNet(input_shape, output_shape)
    elif model_type == 'CNN' or model_type == 'PSCNet':
        model = PSCNet(input_shape, output_shape)
        
    model.compile(loss=loss_func, optimizer=Adam(5e-4), metrics=['accuracy'])
    plot_model(model, './model_Nsignals.png', show_shapes=True)
    hist_2 = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1, 
                        verbose=0, callbacks=[
                            tfdocs.modeling.EpochDots(), 
                            cp_callback_Nsignals,
                            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)])
    #np.save('./history/DNN_N_signals_%s_%d_dB.npy'%(feature_input, SNRdB), hist_2.history)
    print('\n<=========== Saving classification model ============>')
    tf.saved_model.save(model, os.path.abspath(model_path))
    
def test_DL(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, model_type='DNN', environment='cohe'):
    model_path, checkpoint_path_Nsignals, testset_path = path_reader(feature_input, N_antenas, SNRdB, delta_SNRdB, \
                                                                     N_snapshots, model_type, environment, training=False)
    print('<=========== Loading testset (SNR = %sdB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'
          %(str(SNRdB), delta_SNRdB, N_antenas, N_snapshots))
    model = tf.keras.models.load_model(model_path)   
    data = np.load(testset_path, allow_pickle=True).item()
    print('<=========== Testing model (SNR = %sdB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d) ============>'\
          %(str(SNRdB), delta_SNRdB, N_antenas, N_snapshots))
    X_test, y_test = dataset_preparation(data, SNRdB, feature_input, normalized_input=False, data_shuffling=True, \
                                       model_type=model_type)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print('Accuracy: %.2f'%accuracy(y_test, y_pred))
    return acc

def test_DL_resuse(feature_input, N_antenas, SNRdB, delta_SNRdB, N_snapshots, Dy_lamda, model_type='DNN', environment='cohe'):
    # This function applies PSS technique
    # load testset
    print('<=========== Loading testset (SNR = %sdB, delta_SNRdB = %ddB, N_antenas = %d, N_snapshots = %d, Dy_lamda = %d) ============>'
          %(str(SNRdB), delta_SNRdB, N_antenas, N_snapshots, Dy_lamda))
    testset_path = data_path_reader(N_antenas, SNRdB, delta_SNRdB, N_snapshots, Dy_lamda, environment, training=False)
    data = np.load(testset_path, allow_pickle=True).item()

    M_ = int(min(np.ceil(Dy_lamda), np.floor(N_antenas//2)))
    M0_ = 2*M_ + 1
    
    # load model
    model_path, _ = model_path_reader(M0_, SNRdB, delta_SNRdB, N_snapshots, Dy_lamda, model_type, environment, reuse_flag=True)
        
    model = tf.keras.models.load_model(model_path)   
    X_test, y_test = dataset_preparation(data, SNRdB, feature_input, normalized_input=True, data_shuffling=False, model_type=model_type, M_=M_)
    y_pred = model.predict(X_test)
    acc_ = accuracy(y_test, y_pred)
    print('PSS-based Accuracy: %.5f'%acc_)
    return acc_

