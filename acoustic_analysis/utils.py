#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:58:38 2021

@author: sylvia
"""

import wavfile

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import soundsig.sound as ssnd
import h5py
import scipy.signal as spsg
import json

def get_files(path = '.',ext='.wav',recursive = True):
    
    """
    
    Parameters
    ----------
    path : string
        starting folder to look for matching files
    ext : string
        type of file to look for
    third : boolean
        if True, will also look in sub-folders
    
    Returns
    -------
    list
        names of all files in given path
    
    """
    my_path = path + '/**/*'+ext
    file_list = glob.glob(my_path, recursive = recursive)
    return file_list

def make_bioSound(signal, fs, cutoff_freq=1000, max_freq = 80000,window = 64, 
                  ampenv = True, spectrum = True, fund = True, mps = True):
    
    """
    Create biosound object from the given 1D array (meant for sound pressure timeseries)
    
    Parameters
    ----------
    signal: 1D array
        time series of sound wave form
    fs : int
        sampling rate
    cutoff_freq: int
        Default to 1000, highest frequency of the amplitude envelope
    max_freq: int
        Default to 80K, highest frequency calculated in psd and spectrogram
    window: int
        Default to 64 points, width of the kernel for calculating modulation power spectrum--this parameter is a bit finicky
    ampenv, spectrum, fund, mps: boolean
        Default to True. Sets whether to calculate the named parameter. MPS is the most time-consuming, so if looking to save time
        set mps = False
    
    Returns
    -------
    biosound object
    
    """
    
    
    biosound = ssnd.BioSound(soundWave = signal, fs = fs)
    if ampenv:
        biosound.ampenv(cutoff_freq=cutoff_freq)
    if spectrum:
        biosound.spectrum(f_high=max_freq)
        biosound.spectroCalc(max_freq=max_freq)
    if fund:
        biosound.fundest(highFc = max_freq, maxFund = max_freq)
    if mps:
        biosound.mpsCalc(window = window)
    return biosound
    

def make_frame_biosounds(all_h5,fs = 192000,frame_ms = 200.0, stride_ms = 25.0,cutoff_freq=1000, window = 64, save_h5 = True, save_folder = './'):
    frame_size = int(frame_ms*fs/1000.0)
    stride_size = int(stride_ms*fs/1000.0)
    bioSound_list = []
    feature_vectors = []
    for file in all_h5:
        print(file)
        hf = h5py.File(file, 'r')
        signal = np.array(hf.get('signal'))
        hf.close()
        num_points = len(signal)
        start_frame = 0
        i = 0
        while start_frame < num_points-frame_size:
            frame = signal[start_frame:start_frame+frame_size]
            bioSound = make_bioSound(frame, fs)      
            bioSound_list.append(bioSound)
            features = make_feature_vector(bioSound)
            feature_vectors.append(features)
            start_frame+=stride_size
            if save_h5:
                filename = save_folder + os.path.splitext(os.path.split(file)[1])[0] +'_'+str(i)+'.h5'
                bioSound.saveh5(filename)               
            i+=1
    return bioSound_list, feature_vectors

def make_feature_vector(bioSound):
    
    """
    Create an array of averaged features
    Parameters
    ----------
    bioSound : biosound object

    
    Returns
    -------
    29x1 array
        
    
    """
    
    
    # make a feature vector from the features in a biosound object by averaging
    # across the total time frame of the signal
    features = np.zeros(29)
    # Temporal statistics
    if bioSound.meantime.size != 0:
        features[0] = bioSound.meantime
    if bioSound.stdtime.size != 0:
        features[1] = bioSound.stdtime
    if bioSound.skewtime.size != 0:
        features[2] = bioSound.skewtime
    if bioSound.kurtosistime.size != 0:
        features[3] = bioSound.kurtosistime
    if bioSound.entropytime.size != 0:
        features[4] = bioSound.entropytime
    # Pitch statistics
    if bioSound.sal.size != 0:
        features[5] = bioSound.meansal
    if bioSound.fund.size != 0:
        features[6] = bioSound.fund
    if bioSound.F1.size != 0:
        goodF1 = bioSound.F1[~np.isnan(bioSound.F1)]
        meanF1 = np.mean(goodF1)
        features[7] = meanF1
    if bioSound.F2.size != 0:
        goodF2 = bioSound.F2[~np.isnan(bioSound.F2)]
        meanF2 = np.mean(goodF2)
        features[8] = meanF2
    if bioSound.F3.size != 0:
        goodF3 = bioSound.F3[~np.isnan(bioSound.F3)]
        meanF3 = np.mean(goodF3)
        features[9] = meanF3
    # Spectral statistics
    if bioSound.meanspect.size != 0:
        features[10] = bioSound.meanspect
    if bioSound.stdspect.size != 0:
        features[11] = bioSound.stdspect
    if bioSound.skewspect.size!= 0:
        features[12] = bioSound.skewspect
    if bioSound.kurtosisspect.size != 0:
        features[13] = bioSound.kurtosisspect
    if bioSound.entropyspect.size != 0:
        features[14] = bioSound.entropyspect
    if bioSound.q1.size != 0:
        features[15] = bioSound.q1
    if bioSound.q2.size != 0:
        features[16] = bioSound.q2
    if bioSound.q3.size != 0:
        features[17] = bioSound.q3
    # Modulation statistics
    if bioSound.mps.size != 0:
        logMPS = 10*np.log10(bioSound.mps)
        temporal_dist = np.sum(logMPS,axis=0)
        spectral_dist = np.sum(logMPS,axis=1)
        tdata = bioSound.wt
        sdata = bioSound.wf
        (dist_spec, mean_spec, std_spec, skew_spec, kurtosis_spec, entropy_spec) = calc_moments(sdata, spectral_dist)
        (dist_temp, mean_temp, std_temp, skew_temp, kurtosis_temp, entropy_temp) = calc_moments(tdata, temporal_dist)
        s= np.linalg.svd(bioSound.mps,compute_uv=False)
        alpha_sep = s[0]/np.sum(s)
        
        features[18] = mean_spec
        features[19] = std_spec
        features[20] = skew_spec
        features[21] = kurtosis_spec
        features[22] = entropy_spec
        
        features[23] = mean_temp
        features[24] = std_temp
        features[25] = skew_temp
        features[26] = kurtosis_temp
        features[27] = entropy_temp
        
        features[28] = alpha_sep
        
    return features

def calc_moments(x, dist):
    """
    Parameters
    ----------
    x : array
        x-values or timestamps of the signal
    dist : array
        magnitude of the distribution for each corresponding value of x

    
    Returns
    -------
    dist: array
        normalized and absolute value distribution
    mean, std, skew, kurtosis: int
        1st-4th moments of dist
    entropy: int
        entropy of dist
    
    """
    # normalize distribution
    dist_min = np.amin(dist)
    dist-=dist_min
    dist = dist/np.sum(dist)  
    # calculate statistics of distributions
    mean = np.sum(x*dist)
    std = np.sqrt(np.sum(dist*((x-mean)**2)))
    skew = np.sum(dist*(x-mean)**3)
    skew = skew/(std**3)
    kurtosis = np.sum(dist*(x-mean)**4)
    kurtosis = kurtosis/(std**4)
    indpos = np.where(dist>0)[0]
    entropy = -np.sum(dist[indpos]*np.log2(dist[indpos]))/np.log2(np.size(indpos))
    return (dist, mean, std, skew, kurtosis, entropy)

def find_family_tree(keyword, hierarchy_file, add_additional = False):
    """
    find all nested keywords pertaining to a semantic label by iterating over .json file
    Parameters
    ----------
    keyword : string
        most specific semantic label known

    hierarchy_file : string
        json-formatted file listing the 'parent' label for all known semantic labels
    add_additional: boolean
        if True, list includes all keywords in the "additional tags" field
    Returns
    -------
    list
        names of allparent keywords, including original
    
    """
    family = []
    with open(hierarchy_file,'r') as file:
        tree = json.load(file)
        while keyword in tree:
            family.append(keyword)
            if add_additional:
                if 'additional' in tree[keyword][0]:
                    family.extend(tree[keyword][0]['additional'])
            keyword = tree[keyword][0]['parent']
        family.append(keyword)
    return family




def plot(biosound, DBNOISE=50, f_low=250, f_high=10000,powerlog = True):
    # Plots a biosound in figures 1, 2, 3
    
        # Plotting Variables
        soundlen = np.size(biosound.sound)
        t = np.array(range(soundlen))
        t = t*(1000.0/biosound.samprate)

        # Plot the oscillogram + spectrogram
        plt.figure(1)
        plt.clf()

        
        
        # The oscillogram
        ax = plt.axes([0.45, 1.4, 0.855, 0.20])      
        ax.plot(t,biosound.sound, 'k')
        # plt.xlabel('Time (ms)')
        plt.xlim(0, t[-1])               
        # Plot the amplitude enveloppe  
        if biosound.tAmp.size != 0 :      
            ax.plot(biosound.tAmp*1000.0, biosound.amp, 'r', linewidth=2)
        ax.set_xticks([])
        ax.set_ylim((-1,1))
      
        # Plot the spectrogram
        ax = plt.axes([0.45, 0.75, 1.07, 0.6])
        cmap = plt.get_cmap('binary')
        
        if biosound.spectro.size != 0 :
            soundSpect = biosound.spectro
            if soundSpect.shape[0] == biosound.to.size:
                soundSpect = np.transpose(soundSpect)
            maxB = soundSpect.max()
            minB = maxB-DBNOISE
            soundSpect[soundSpect < minB] = minB
            minSpect = soundSpect.min()
            plt.imshow(soundSpect, extent = (biosound.to[0]*1000, biosound.to[-1]*1000, biosound.fo[0], biosound.fo[-1]), aspect='auto', interpolation='nearest', origin='lower', cmap=cmap, vmin=minSpect, vmax=maxB)
            plt.colorbar()
       
        plt.ylim(f_low, f_high)
        plt.xlim(0, t[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        

        
        # Power Spectrum
        
        ax = plt.axes([0.1,0.75,0.3,0.6])
        if biosound.psd.size != 0 :
            if powerlog:
                psd = np.log(biosound.psd)
                xlab = 'Power Log10'
            else:
                psd = biosound.psd
                xlab = 'Power Linear'
            plt.plot(psd,biosound.fpsd, 'k-') 
            plt.ylabel('Frequency Hz')
            plt.xlabel(xlab)
            
        yl, yh, xl, xh = plt.axis()
        xl = 0.0
        xh = 10000.0
        plt.axis((yl, yh, xl, xh))
    
        if biosound.q1.size != 0:
            plt.plot([yl, yh], [biosound.q1, biosound.q1], 'k--')
            plt.plot([yl, yh], [biosound.q2, biosound.q2], 'k--')
            plt.plot([yl, yh], [biosound.q3, biosound.q3], 'k--') 
        
        if biosound.F1.size != 0:        
            F1Mean = biosound.F1[~np.isnan(biosound.F1)].mean()
            F2Mean = biosound.F2[~np.isnan(biosound.F2)].mean()
            F3Mean = biosound.F3[~np.isnan(biosound.F3)].mean()
            plt.plot([yl, yh], [F1Mean, F1Mean], 'r--', linewidth=2.0)
            plt.plot([yl, yh], [F2Mean, F2Mean], 'c--', linewidth=2.0)
            plt.plot([yl, yh], [F3Mean, F3Mean], 'b--', linewidth=2.0)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ylim)
                     
    # Plot the fundamentals

        if biosound.f0.size != 0 :

            ax = plt.axes([0.45,0.1, 0.855, 0.6])
            ax.plot(biosound.to*1000.0, biosound.f0, 'k', linewidth=3, label = 'fundamental')
            ax.plot(biosound.to*1000.0, biosound.f0_2, 'm', linewidth=3, label = 'fundamental 2')
            ax.plot(biosound.to*1000.0, biosound.F1, 'r--', linewidth=3, label = 'formant 1')
            ax.plot(biosound.to*1000.0, biosound.F2, 'c--', linewidth=3, label = 'formant 2')
            ax.plot(biosound.to*1000.0, biosound.F3, 'b--', linewidth=3, label = 'formant 3')
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (ms)')
            plt.legend()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        plt.show()
            
           


  
    # Table of results
        plt.figure(2)
        plt.clf()

        textstr = '%s  %s' % (biosound.emitter, biosound.type)
        plt.text(0.4, 1.0, textstr)
        if biosound.fund.size != 0:
            if biosound.fund2.size != 0:
                textstr = 'Mean Fund = %.2f Hz Mean Saliency = %.2f Mean Fund2 = %.2f PF2 = %.2f%%' % (biosound.fund, biosound.sal, biosound.fund2, biosound.voice2percent)
            else:
                textstr = 'Mean Fund = %.2f Hz Mean Saliency = %.2f No 2nd Voice Detected' % (biosound.fund, biosound.sal)
            plt.text(-0.1, 0.8, textstr)
            
        if biosound.fund.size != 0:
            textstr = '   Max Fund = %.2f Hz, Min Fund = %.2f Hz, CV = %.2f' % (biosound.maxfund, biosound.minfund, biosound.cvfund) 
            plt.text(-0.1, 0.7, textstr)
        textstr = 'Mean Spect = %.2f Hz, Std Spect= %.2f Hz' % (biosound.meanspect, biosound.stdspect)
        plt.text(-0.1, 0.6, textstr)
        textstr = '   Skew = %.2f, Kurtosis = %.2f Entropy=%.2f' % (biosound.skewspect, biosound.kurtosisspect, biosound.entropyspect)
        plt.text(-0.1, 0.5, textstr)
        textstr = '   Q1 F = %.2f Hz, Q2 F= %.2f Hz, Q3 F= %.2f Hz' % (biosound.q1, biosound.q2, biosound.q3 )
        plt.text(-0.1, 0.4, textstr)
        if biosound.F1.size != 0:
            textstr = '   For1 = %.2f Hz, For2 = %.2f Hz, For3= %.2f Hz' % (F1Mean, F2Mean, F3Mean )
            plt.text(-0.1, 0.3, textstr)
        textstr = 'Mean Time = %.2f s, Std Time= %.2f s' % (biosound.meantime, biosound.stdtime)
        plt.text(-0.1, 0.2, textstr)
        textstr = '   Skew = %.2f, Kurtosis = %.2f Entropy=%.2f' % (biosound.skewtime, biosound.kurtosistime, biosound.entropytime)
        plt.text(-0.1, 0.1, textstr)
        if biosound.rms.size != 0 and biosound.maxAmp.size != 0 :
            textstr = 'RMS = %.2f, Max Amp = %.2f' % (biosound.rms, biosound.maxAmp)
            plt.text(-0.1, 0.0, textstr)
        
        plt.axis('off')        
        plt.show()
        
    # Plot Modulation Power spectrum if it exists
    
        #ex = (spectral_freq.min(), spectral_freq.max(), temporal_freq.min(), temporal_freq.max())
        cmap = plt.get_cmap('binary')
        
        if biosound.mps.size != 0 :
        # Plot the modulation power spectrum
            plt.figure(3)
            ax = plt.axes([0.45,0.75,1.07,0.6])
            ex = (biosound.wt.min(), biosound.wt.max(), biosound.wf.min()*1e3, biosound.wf.max()*1e3)
            logMPS = 10.0*np.log10(biosound.mps)
            maxMPS = logMPS.max()
            minMPS = maxMPS-50
            logMPS[logMPS < minMPS] = minMPS
            plt.imshow(logMPS, interpolation='nearest', aspect='auto', origin='lower', cmap=cmap, extent=ex)
            plt.xlabel('Temporal Frequency (Hz)')
            plt.colorbar()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_yticks([])
            plt.ylim((0,biosound.wf.max()*1e3))
            ylim = ax.get_ylim()
        
        # Plot the temporal modulation
            ax = plt.axes([0.45,1.4,0.855,0.20])
            temporal_dist = np.sum(10*np.log10(biosound.mps),axis=0)
            spectral_dist = np.sum(10*np.log10(biosound.mps),axis=1)
            spec_start = np.where(biosound.wf == 0)[0][0]
            spec_end = np.where(biosound.wf == biosound.wf.max())[0][0]
            plt.plot(biosound.wt,temporal_dist)
            ax.set_xlim(xlim)
            ax.set_xticks([])
        
        # Plot the spectral modulation
            ax = plt.axes([0.1,0.75,0.3,0.6])
            plt.plot(spectral_dist[spec_start:spec_end], 1e3*biosound.wf[spec_start:spec_end])
            #plt.plot(spectral_dist, 1e3*biosound.wf)
            plt.ylabel('Spectral Frequency (Cycles/KHz)')
            ax.set_ylim(ylim)
            ax.set_xlim(ax.get_xlim()[::-1])
        
        plt.pause(1)   # To flush the plots?
        plt.show()
        
