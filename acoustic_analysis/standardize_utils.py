#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:12:19 2019

@author: sylvia
"""



import wavfile
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

import scipy.signal.signaltools as spsg
from mkl_fft._numpy_fft import rfft, irfft, fft, ifft
import wave

import h5py
import ipdb

# Monkey patch ffts to use mkl to speed up standardize
spsg.fftpack.fft = fft
spsg.fftpack.ifft = ifft


def get_files(path = '.',ext='.wav',recursive = True):
    my_path = path + '/**/*'+ext
    file_list = glob.glob(my_path, recursive = recursive)
    return file_list


def make_h5(file_list,save_folder = '.',ext = '.h5',samp_new=192000,percentile = 99.9999, bitdepth = 32,stereo = False,
                       standardize = True, save_h5 = True):
    success = []
    sigs = []
    for file in file_list:
        
        filepath = os.path.splitext(file)[0]
        filename = os.path.split(filepath)[1]
        
        # read file stats
        try:
            wavarray = wavfile.read(file)
            samp_f0 = wavarray[0]
            sig = wavarray[1]

            ipdb.set_trace()
            if standardize: 
                sig_stand = standardize_wav(sig,samp_f0,samp_new,percentile,stereo)
                print(filename)
            else:
                sig_stand = sig
            h5_filename = save_folder+ '/' + filename + ext
            if save_h5:
                f = h5py.File(h5_filename,'w')
                f.create_dataset('signal',data = sig_stand)
                f.create_dataset('samp_rate', data = samp_new)
                f.create_dataset('bitdepth', data = bitdepth)
                f.close()
            sigs.append(sig_stand)
            success.append(True)
        except:
            print('Could not read file: '+file)
            success.append(False)    
    return success, sigs


def read_sigs(file_list):
    success = []
    sigs = []
    f0s = []
    filenames = []
    for file in file_list:
        
        filepath = os.path.splitext(file)[0]
        filename = os.path.split(filepath)[1]
        
        # read file stats
        try:
            wavarray = wavfile.read(file)
            samp_f0 = wavarray[0]
            sig = wavarray[1]
            sigs.append(sig)
            f0s.append(samp_f0)
            filenames.append(filename)
            success.append(True)
        except:
            print('Could not read file: '+file)
            success.append(False)    
    return success, sigs, f0s, filenames

def standardize_all(signals, f0s, filenames,samp_new=192000,percentile = 99.9999, bitdepth = 32,stereo = False):
    sigs_standard = []    
    for i in range(len(signals)):
        sig = standardize_wav(signals[i],f0s[i],samp_new,percentile,stereo)
        plt.plot(sig)
        sigs_standard.append(sig)
    return sigs_standard


def process_sound_list(file_list,samp_new=192000,percentile = 99.9999, bitdepth = 32,stereo = False,
                       standardize = True,make_summary_plots = True, save_h5 = True, 
                       make_specgrams = True):
    # get statistics for all files
    durations = []
    samp_freqs = []
    bitdepths = []
    amplitudes = []
    success = []
    for file in file_list:
        # read file stats
        
        try:
            file_wav = wave.open(file,mode = 'rb')
            samp_f0 = file_wav.getframerate()
            nframes = file_wav.getnframes()
            dur = nframes/samp_f0
            bdepth = file_wav.getsampwidth()
            file_wav.close()
            
             # FOR JI HYUN: This is how I use the wavfile code to open .wav 
            wavarray = wavfile.read(file)
            samp_f0 = wavarray[0]
            sig = wavarray[1]
            amp = np.amax(np.abs(sig))
        
            durations.append(dur)
            samp_freqs.append(samp_f0)
            bitdepths.append(bdepth)
            amplitudes.append(amp)
            if standardize:
                sig = standardize_wav(sig,samp_f0,samp_new,percentile,stereo)
            success.append(True)
        except:
            print('Could not read file: '+file)
            durations.append(0)
            samp_freqs.append(0)
            bitdepths.append(0)
            amplitudes.append(0)
            success.append(False)
                        
            
    if make_summary_plots:
        f,axes = plot_summary_statistics(duration=list(np.compress(durations,success)),sampling_frequency=list(np.compress(samp_freqs,success)),
                                         bit_depth=list(np.compress(bitdepths,success)),amplitude=list(np.compress(amplitudes,success)))
    return f, axes, durations, samp_freqs, bitdepths, amplitudes, success
        
            

def standardize_wav(sig,samp_in,samp_new = 192000,percentile = 99.9999, bitdepth = 32, stereo = False):
    # convert to mono
    dims = len(sig.shape)
    if stereo == False:
        if dims>1:
            numChannels = float(sig.shape[1])
            sig = np.sum(sig, axis=1,keepdims=0)/numChannels
        else:
            numChannels = 1.0
    # standardize sampling frequency
    numppoints0 = len(sig)  
    numpointsT = int(np.ceil(samp_new*numppoints0/samp_in))
    if samp_in<samp_new:
        sig = spsg.resample(sig,numpointsT)
    elif samp_in == samp_new:
        print('Sampling frequency already at target value!')
    else:
        print('f0 higher than target')
    # normalize and clip
    sig = normalize_and_clip(sig,percentile = percentile)
    # normalize bit depth
    sig = normalize_bit_depth(sig,bitdepth)
    return sig
        
    
    
def plot_summary_statistics(duration = None,sampling_frequency = None,bit_depth = None,amplitude = None):
    numfigs = 0
    titles = []
    arrays = []
    if duration != None:
        titles.append('Duration (s)')
        arrays.append(duration)
        numfigs+=1
    if sampling_frequency != None:
        titles.append('Sampling Frequency (Hz)')
        arrays.append(sampling_frequency)
        numfigs += 1
    if bit_depth !=None:
        titles.append('Bit Depth')
        arrays.append(bit_depth)
        numfigs+=1
    if amplitude!=None:
        titles.append('Amplitude')
        arrays.append(amplitude)
        numfigs+=1
    fig, axes = plt.subplots(numfigs,1)
    plt.subplots_adjust(hspace = 1.5)
    for i in range(numfigs):
        n,bins,patches = axes[i].hist(x = arrays[i],bins = 500,alpha = 0.7,rwidth = 0.85)
        axes[i].grid(True,alpha = 0.75)
        axes[i].set_xlabel(titles[i])
        print(i)
    return fig, axes


def normalize_and_clip(sig,percentile = 99.9999):
    sig_clipped = np.copy(sig)
    norm_factor = np.percentile(abs(sig),99.9999)
    sig_clipped[sig>norm_factor] = norm_factor
    sig_clipped[sig<-norm_factor] = -norm_factor
    sig_norm = sig_clipped/norm_factor
    return sig_norm


def normalize_bit_depth(sig, bitdepth):
    sig_max = np.abs(sig).max()
    sig_norm = sig/sig_max
    # quantize (bit-depth)
    q = np.round(2**(bitdepth-1)*sig_norm);
    sig_out = sig_max*q/(2**(bitdepth-1));
    return sig_out


        