#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:38:38 2022

@author: Brainclinics
08-01-2024: adapted the kubios export readout, also fixed the sampling rate to 130 Hz, since higher sampling rates seem to have an unforseen effect on the TFR computations, and make the data uncomparable to the original data, this needs to be checked out and fixed.

"""

import sys
import os
import numpy as np
from mne.time_frequency import tfr_array_morlet
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, hilbert, convolve, boxcar, medfilt
from scipy.signal.windows import hann
from scipy.stats import zscore
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import datetime
from tqdm.auto import tqdm
import hrvanalysis

from seaborn import heatmap
from matplotlib.backends.backend_pdf import PdfPages

from ncg_code.inout import FilepathFinder as ff

#import warnings
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning, module="matplotlib")

startdir='/home/'
#startdir = ''
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
 
def _morlet(sfreq, freqs, n_cycles=10.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.
    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : array
        Frequency range of interest (1 x Frequencies).
    n_cycles : float | array of float, default 7.0
        Number of cycles. Fixed number or one per frequency.
    sigma : float, default None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, default False
        Make sure the wavelet has a mean of zero.
    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)

    freqs = np.array(freqs)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be "
                         "greater than 0.")

    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        gaussian_enveloppe = np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
        if zero_mean:  # to make it zero mean
            real_offset = np.exp(- 2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        W = oscillation * gaussian_enveloppe
        W /= np.sqrt(0.5) * np.linalg.norm(W.ravel())
        Ws.append(W)
    return Ws


def main_HBC_independent(subid, in_dur=5, out_dur=11, repetitions=16):
    '''
    main_HBC_independent 
    code used to make the HBC report, and that is used in the app. 
    Needs 
    '''
    #
    #
    import sys
    import os
    import numpy as np
    from mne.time_frequency import tfr_array_morlet
    import mne
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import find_peaks
    from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, hilbert, convolve, boxcar, medfilt
    from scipy.signal.windows import hann
    from scipy.stats import zscore
    from matplotlib.patches import Rectangle
    from scipy.stats import linregress
    import datetime
    from tqdm.auto import tqdm
    import hrvanalysis

    from seaborn import heatmap
    from matplotlib.backends.backend_pdf import PdfPages

    from ncg_code.inout import FilepathFinder as ff
    from warnings import filterwarnings
    filterwarnings("ignore", category=UserWarning, module="matplotlib")

    #import warnings
    
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #mne.utils.set_config('MNE_USE_CUDA', 'true') 
    
    # the sampling frequency used in these calculations (default is 130Hz)
    # here the functions variables are implemented 
    sF = 130
    onecycle = (in_dur+out_dur) # in seconds, 5 seconds stim 11 seconds rest
    totalstimlength = (onecycle*(repetitions)*sF) #15 intensities +1 because python starts counting at 0 
    end = totalstimlength
    
    if repetitions<16:
        #raise Exception(f'The analysis requires at least 15 repetitions and only {repetitions} are provided')
        print(f'The analysis requires at least 16 cycles/ repetitions and only {repetitions} are provided')
        exit()
    
    # determine the expected frequecy
    freq_exp = np.round(1/onecycle, decimals=4) #1 sec/ cycle sec = Hz

    # Find the folders to be used in analysis (each subject has a folder with one or more .txt or 
    # .csv or .npy files in it, containing the ECG or the RR intervals)
    ignore = ['._','DS','.ipynb', 'result','stats', 'pdf'] # Files with these characters in their names should be ignored
    if subid == 'all':
        dirs = os.listdir(startdir+'data/input/')
        dirs = [d for d in dirs if not any([ig in d for ig in ignore])]
        dirs = np.sort(dirs)
        alldata = np.zeros((1,onecycle*repetitions*sF))
        print(dirs)
        all_files=[]
    else:
        dirs = [subid]

    for d in dirs:
        variables = ['fdata', 'startdata', 'data', 'hanndata', 'dat', 'dat1']
        for var in variables:
            globals().pop(var, None)

        #print(d)
        files = []
        files = [f for f in os.listdir(startdir+'data/input/'+d) if not any([ig in f for ig in ignore])]
        #print(files)
        _,ext = os.path.splitext(files[0])
        del files
        if ext == '.txt':
            datatype = 'raw'#if there are .txt files in the folder code will assume it is raw ECG
            print('[WARNING] Only valid for .txt files from the polar H10 device')
        elif ext == '.csv':
            datatype = 'RR_intervals_csv'#if there are .csv files in the folder code will assume it is kubios exported RR intervals
        try:
            ecg = ff(ext,startdir+'data/input/'+d)
            ecg.get_filenames()
            #print(datatype)
            files = [f for f in ecg.files if not any([ig in f for ig in ignore])] # find all usable files within a subjects folder
            files = np.sort(files)
            print(f'Files used for analysis:\n {files}')
            artefacted_flag = np.zeros((len(files)))# initiate array for artefact flagging in the end-report 
            path,tail = os.path.split(files[0])
            filen, ext = os.path.splitext(tail)
            locs=[]


            start = 0
            trainstart = np.arange(0,(onecycle*(repetitions+1)*sF),onecycle*sF) # needs a plus 1 because it would otherwise ignore the last intensity 
            os.mkdir(startdir+'data/output/'+d+'/')
            with PdfPages(startdir+'data/output/'+d+'/'+d+'_deart.pdf') as pp:
                nr=1 #used for titles in the plots
                BHmarker=np.zeros((repetitions));normBHmarker = np.zeros((repetitions));logBHmarker=np.zeros((repetitions));slopes=[];rvalues=[];pvalues=[]
                for f in tqdm(files):
                    #print(f)
                    fpath, ext = os.path.splitext(f)
                    head,tail = os.path.split(fpath)
                    fname = tail.rsplit('.')[0]
                    try:#try to load the .txt (raw ECG) data if avalaible even if there are otherwise annotated data available
                        startdata = np.array(pd.read_csv(startdir+'data/input/'+d+'/'+str(fname)+".txt",sep=',').iloc[:,0])
                        #print(len(startdata))
                        if len(startdata) < totalstimlength:
                            print(f'The provided datalength ({np.round(len(startdata)/sF, decimals = 2)}) does not fit with the requested datalength ({np.round(totalstimlength/sF, decimals = 2)}) seconds')
                            exit()
                        if '_mkr' in f: # In the beginning we were using another app and used markers to annotate the beginning of intensity 1, the following code handles that
                            highpass = 5 #/ nyq
                            lowpass = 63.9999
                            # ''' bandpassfilter '''
                            sos = butter(4,[highpass,lowpass], btype='bandpass', analog=False, output = 'sos', fs=sF)
                            fdata = sosfiltfilt(sos, startdata)

                            marker =  np.where(zscore(fdata)>8.5)[0][0] #QDS rest period before starting first stim       
                            if marker-(onecycle*sF)<0:
                                start = 0
                                end = marker+(onecycle*repetitions*sF)#15 intensities +1 rest
                            else:
                                start = marker-(onecycle*sF)
                                end = marker+(onecycle*repetitions*sF)#15 intensities +1 rest
                            data = startdata[int(start):int(end)]
                        else:
                            start = 0
                            end = (onecycle*(repetitions+2)*sF)#15 intensities +1 rest and whatever is left after tgat

                            data = startdata[int(start):int(end)]
                            #print(len(data))
                            del startdata

                        highpass = 5 #/ nyq
                        lowpass = 49
                        # ''' bandpassfilter '''
                        sos = butter(4,[highpass,lowpass], btype='bandpass', analog=False, output = 'sos', fs=sF)
                        fdata = sosfiltfilt(sos, data)
                        annotdata = np.zeros((len(fdata))); annotdata[:] = np.nan #initiate for plotting, to show the annotated peaks
                        peaks,amps = find_peaks(fdata, prominence=4*np.std(fdata)+np.mean(fdata), wlen=.25*sF)                  
                        #only take r-peaks that fall within 2SD of the mean peak-hight
                        peakz = zscore(fdata[peaks])
                        rpeaks = peaks[np.where(np.logical_and(peakz>=-7,peakz<7))]#Only take peaks into account that are within a 'normal' amplitude range
                        annotdata[rpeaks]=fdata[peaks][np.where(np.logical_and(peakz>=-7,peakz<7))] #for plotting, to show the annotated peaks

                    except Exception as e:
                        #print(e)
                        pass
                    
                    if datatype == 'RR_intervals_csv': #load the kubios exported RR-intervals
                        #fixed while it looks like with the higher sampling rate the resolution is higher, and therefore freq result of the app as well.
                        print('[INFO] Analyzing from Kubios export')
                        try:
                            df = pd.read_csv(f, sep = ',')
                        except:
                            df = pd.read_csv(f, sep = ';')
                        df = df.rename(columns = {'sep=,':'test'})
                        max_commas = df['test'].str.split(',').transform(len).max()
                        df[[f'test_{x}' for x in range(max_commas)]] = df['test'].str.split(',', expand=True)
                        df=df.drop('test',axis=1)
                        index = np.where(df['test_0']=='RR INTERVAL DATA')[0][0]
                        dftest = df.truncate(before=index)
                        
                        df_time = dftest['test_1'].str.strip().reset_index()
                        df_time.drop('index',axis=1)
                        df_time['test_1']= df_time['test_1'].replace('',np.nan)
                        df_time = df_time.dropna(subset = 'test_1',axis=0)
                        index = np.where(df_time['test_1']=='(s)')[0][0]
                        df_time = df_time.truncate(before=index+1).reset_index()
                        rpeaks=np.array(np.round(np.array(df_time.values,dtype=float)*sF), dtype=int)[:,2]
                        
                        #rpeaks = np.array(rpeaks.values[3:,0],dtype=float)*sF #first three lines are headers
                        #rpeaks = np.array(rpeaks,dtype=int)

                        try:
                            annotdata[rpeaks]=fdata[rpeaks]#for plotting, to show the annotated peaks, only when the raw ECG is available
                        except:
                            pass
                    
                    ##============Old method================#
                    #if datatype == 'RR_intervals_csv': #load the kubios exported RR-intervals
                    #    print('analyzing from manually scored r-peaks')
                    #    try:
                    #        rpeaks = pd.read_csv(f, sep = ',')
                    #        rpeaks = np.array(rpeaks.values[3:,0],dtype=float)*sF #first three lines are headers
                    #        rpeaks = np.array(rpeaks,dtype=int)
                    #    except:
                    #        rpeaks = pd.read_csv(f, sep = ';')
                    #        rpeaks = np.array(rpeaks.values[3:,0],dtype=float)*sF #first three lines are headers
                    #        rpeaks = np.array(rpeaks,dtype=int)
                    #    try:
                    #        annotdata[rpeaks]=fdata[rpeaks]#for plotting, to show the annotated peaks, only when the raw ECG is available
                    #    except:
                    #        pass

                    RRinterval = np.diff(rpeaks)/sF #compute the RR intervals
                    #!!! I don't know if this is still necesarry if we do the interpolations at the deartifacting step
                    # Only take RR intervals in further analysis that are not to fast.
                    zRR = zscore(RRinterval)
                    toofast= np.where(zRR<-5)[0] 
                    rpeaks2 =rpeaks.copy()
                    RRinterval2 = RRinterval.copy()
                    if len(toofast)>0:
                        skip = np.where(np.diff(toofast)==1)[0]
                        if len(skip)>0:
                            toofast = np.delete(toofast,skip+1)
                        rpeaks2 = np.delete(rpeaks2,toofast+1)
                        RRinterval2 = np.diff(rpeaks2)/sF

                    HR = 60/RRinterval2
                    # Reverse engineered based on kubios. Fill the datapoints between the R-peaks with HR data that is computed from the HR before the specific R-peak
                    # for the loop to work we add interpolation of HR at end (because of the diff computation in rpeaks2)
                    HR = np.hstack((HR,HR[-1]+(HR[-1]-HR[-2])))
                    i=0
                    try:
                        meanHR = np.zeros((len(fdata)));meanHR[:]=np.nan
                        for h in range(len(fdata)):
                            if i == len(rpeaks2):
                                meanHR[h]=HR[i-1]
                            else:
                                meanHR[h]=np.mean(HR[i:i+1])
                                if h == rpeaks2[i]:
                                    i = i+1
                    except Exception as e:
                        meanHR = np.zeros((onecycle*repetitions*sF));meanHR[:]=np.nan
                        for h in range(onecycle*repetitions*sF):
                            if i == len(rpeaks2):
                                meanHR[h]=HR[i-1]
                            else:
                                meanHR[h]=np.mean(HR[i:i+1])
                                if h == rpeaks2[i]:
                                    i = i+1
                    paddedHR = np.pad(meanHR[~np.isnan(meanHR)],2*sF, mode='reflect') #zeropad the data to be able to use more (long) wavelets
                    hanndata = convolve(paddedHR, hann(int(2*sF)), mode ='same', method ='auto')/sum(hann(int(2*sF))) #smooth the data out with a hanning window
                    hanndata = hanndata[2*sF:]
                    hanndata = hanndata[:-2*sF]
                    #+======== Artifact detection
                    peaks_art=[]
                    if np.std(hanndata)>12: # this means the whole measurement is bad, and will be flagged
                        artefacted_flag[nr-1] = 1 #nr counts from 1 forwards
                    else:   
                        #interpolation of artefacts such as ectopic heartbeats if necesarry and possible
                        dat = hanndata.copy()
                        dat1 = hanndata.copy() #keep original for plotting
                        z = zscore(dat)    
                        peaks_art,amps_art = find_peaks(abs(z), prominence=4, wlen=5*sF) # find artefactual peaks
                        if len(peaks_art)>0:
                            #print(peaks_art)
                            for l,r in zip(amps_art['left_bases'],amps_art['right_bases']): # replace data with nan's from left to right base of peak
                                if l==0: # function does not interpolate if first datapoint is a NaN
                                    l=1
                                dat[np.arange(l,r)]=np.nan
                            hanndata = hrvanalysis.preprocessing.interpolate_nan_values(dat) # do a linear interpolation of the data surrounding the artifact

                    hanndata = np.array(hanndata)
                    #all_files.append(d+f)
                    #alldata = np.vstack((alldata,hanndata[:16*16*sF]))
                    #==========
                    prepadding=[];postpadding=[] #use reflective padding to create enough data to do the TFR analysis
                    for p in range(50):
                        if p%2==0:
                            prepadding = np.hstack((prepadding, hanndata[:trainstart[1]][::-1]))
                            postpadding = np.hstack((postpadding, hanndata[trainstart[-2]:][::-1]))
                        else:
                            prepadding = np.hstack((prepadding, hanndata[:trainstart[1]]))
                            postpadding = np.hstack((postpadding, hanndata[trainstart[-2]:]))

                    paddeddata = np.hstack((prepadding,hanndata,postpadding))

                    #============App freqanalysis
                    #hanndataApp = np.array(hanndata[:onecycle*(repetitions+1)*sF])

                    #prepaddingApp=[];postpaddingApp=[] #use reflective padding to create enough data to do the TFR analysis
                    #for p in range(50):
                    #    if p%2==0:
                    #        prepaddingApp = np.hstack((prepaddingApp, hanndataApp[:trainstart[1]][::-1]))
                    #        postpaddingApp = np.hstack((postpaddingApp, hanndataApp[trainstart[-2]:][::-1]))
                    #    else:
                    #        prepaddingApp = np.hstack((prepaddingApp, hanndataApp[:trainstart[1]]))
                    #        postpaddingApp = np.hstack((postpaddingApp, hanndataApp[trainstart[-2]:]))

                    #paddeddataApp = np.hstack((prepaddingApp,hanndataApp,postpaddingApp))

                    #W = _morlet(sF,[freq_exp],n_cycles=10.0)
                    #tfrApp = convolve(paddeddataApp, np.conj(W[0])[::-1], mode='same')
                    #tfrApp = abs(tfrApp)**2
                    #tfrApp = tfrApp[len(prepaddingApp)::]
                    #tfrApp = tfrApp[:-len(postpaddingApp)]
                    #================    

                    fortfrdata = np.expand_dims(np.expand_dims(paddeddata,axis=0),axis=0)#mne requires a matrix with dimensions: segments x channels x samples

                    # define the frequencies to be analyzed
                    freqs=np.round(np.arange(0.02,0.18,0.0005),decimals=4)

                    tfr3 = tfr_array_morlet(fortfrdata[:,:,:], sfreq=sF,freqs=freqs, output='power', n_cycles=3)
                    tfr3 = tfr3[0,0,:,len(prepadding):] #cut the actual data out again
                    tfr3 = tfr3[:,:-len(postpadding)]

                    tfr10 = tfr_array_morlet(fortfrdata[:,:,:], sfreq=sF,freqs=freqs, output='power', n_cycles=10)
                    tfr10 = tfr10[0,0,:,len(prepadding):] #cut the actual data out again
                    tfr10 = tfr10[:,:-len(postpadding)]

                    meanTFR = np.array(tfr10[np.where(freqs==freq_exp)[0][0],:]) #extract the power at the expected induced frequency
                    normTFR = (meanTFR-np.min(meanTFR))/(np.max(meanTFR)-np.min(meanTFR))*10 #min max normalize between 0 and 10
                    logTFR = np.log(meanTFR)
                    #tfrApp = (tfrApp-np.min(tfrApp))/(np.max(tfrApp)-np.min(tfrApp))*10


                    # compute average power at expected induced frequency for each intensity
                    HBmarker = []
                    a=0
                    for x in trainstart[1:]:
                        if a>0 and a == x:
                            break
                        HBmarker.append(np.mean(meanTFR[a:x]))
                        a=x
                    HBmarker = np.array(HBmarker)

                    normHBmarker = []
                    a=0
                    for x in trainstart[1:]:
                        if a>0 and a == x:
                            break
                        normHBmarker.append(np.mean(normTFR[a:x]))
                        a=x
                    normHBmarker = np.array(normHBmarker)
                    
                    logHBmarker = []
                    a=0
                    for x in trainstart[1:]:
                        if a>0 and a == x:
                            break
                        logHBmarker.append(np.mean(logTFR[a:x]))
                        a=x
                    logHBmarker = np.array(logHBmarker)
                    print(logHBmarker)

                    #AppHBmarker = []
                    #a=0
                    #for x in trainstart[1:]:
                    #    if a>0 and a == x:
                    #        break
                    #    AppHBmarker.append(np.mean(tfrApp[a:x]))
                    #    a=x
                    #AppHBmarker = np.array(AppHBmarker)


                    # for plotting make the data same length as original ECG
                    plotscore = np.zeros((trainstart[-1]))
                    a = 0;b=0
                    for x in trainstart[1:]:
                        if a>0 and a == x:
                            break
                        plotscore[a:x] = HBmarker[b]
                        a=x;b=b+1

                    # stack the HBmarker for all locations for end-summary
                    logBHmarker = np.vstack((logBHmarker,logHBmarker))
                    BHmarker = np.vstack((BHmarker,HBmarker))
                    normBHmarker = np.vstack((normBHmarker, normHBmarker))
                    #AppBHmarker = np.vstack((AppBHmarker, AppHBmarker))                        
                    #print(normBHmarker.shape)

                    # plotting for report
                    fig,axs=plt.subplots(4,1,figsize=(8.23,11.69))
                    fig.suptitle(str(nr)+': '+tail, fontsize=10)

                    try:# if RR intervals as input there will be no raw ECG
                        axs[0].plot(fdata)
                        axs[0].set_title('ECG')
                        axs[0].set_xlim(0,len(fdata))
                        axs[0].plot(annotdata,marker=3,linewidth=.001,label='R peaks')
                        axs[0].set_ylabel('ECG')
                        axs[0].vlines(trainstart,axs[0].get_ylim()[0],axs[0].get_ylim()[1],'k', linewidth = .5)
                        axs[0].set_xticklabels([])
                        axs[0].set_xticks([])
                    except:
                        pass

                    #highpass = 0.01 #/ nyq
                    #lowpass = 0.115
                    ## ''' bandpassfilter '''
                    #sos = butter(4,[highpass,lowpass], btype='bandpass', analog=False, output = 'sos', fs=sF)
                    #fhanndata = sosfiltfilt(sos, hanndata)

                    axs[1].set_title('mean HR')
                    if len(peaks_art)>0:
                        axs[1].plot(dat1, label = 'original')
                        axs[1].plot(hanndata,label = 'interpolated')
                    else:
                        axs[1].plot(hanndata)
                    if artefacted_flag[nr-1]==1:
                        axs[1].fill_between(np.arange(0,len(hanndata)),np.min(hanndata),np.max(hanndata), alpha = 0.3,color='r')
                        axs[1].text(len(hanndata)/2,80,'DATA NOT USABLE')#halfway between 30 and 130
                    axs[1].set_ylabel('filtered HR (BPM)')
                    axs[1].set_ylim(30,130)#np.nanmean(fhanndata)-5*np.nanstd(fhanndata),np.nanmean(fhanndata)+5*np.nanstd(fhanndata))
                    axs[1].vlines(trainstart, 30,130,'k',linewidth=.5)#axs[1].get_ylim()[0], axs[1].get_ylim()[1], 'k', linewidth =  .5)
                    axs[1].set_xlim(0,len(hanndata))
                    axs[1].set_xticklabels([])
                    axs[1].set_xticks([])
                    axs[1].legend()

                    if artefacted_flag[nr-1]==0:
                        #tfr3=np.log(tfr3)
                        axs[2].set_title('TFR high time resolution')
                        tfr3mean = 2*np.nanmean(tfr3[np.where(freqs==freq_exp),:])
                        tfr3std = 2*np.nanstd(tfr3[np.where(freqs==freq_exp),:])
                        im3 = axs[2].imshow(tfr3[:,:], alpha=1, cmap = 'coolwarm', aspect='auto', vmin=tfr3mean-tfr3std, vmax=tfr3mean+tfr3std)#tfr3mean-tfr3std,
                        masked = np.ma.masked_where(np.log(tfr3[:,:]) > 9.5,tfr3[:,:])
                        im3 = axs[2].imshow(masked,alpha=1,cmap = 'Greys', aspect='auto', vmin=tfr3mean-tfr3std, vmax=tfr3mean+tfr3std)#tfr10mean-tfr10std,

                        im3 = axs[2].plot(np.ones((tfr3.shape[1]))*np.where(freqs==freq_exp)[0],'g', alpha = 0.8, linewidth = .8)
                        axs[2].vlines(trainstart, axs[2].get_ylim()[0], axs[2].get_ylim()[1], 'k', linewidth =  .5)
                        axs[2].set_ylabel('Frequency (Hz)')
                        axs[2].set_yticks([0, len(freqs)/2, len(freqs)])
                        axs[2].set_yticklabels(['0.02', '0.1', '0.18'])
                        axs[2].set_ylim(0,len(freqs))
                        axs[2].set_xticklabels([])
                        axs[2].set_xticks([])
                        axs[2].set_xlim(0,len(hanndata))

                        #tfr10=np.log(tfr10)
                        axs[3].set_title('TFR high frequency resolution')
                        tfr10mean = np.nanmean(tfr10[np.where(freqs==freq_exp),:])
                        tfr10std = 1.5*np.nanstd(tfr10[np.where(freqs==freq_exp),:])
                        im4 = axs[3].imshow(tfr10[:,:],alpha=1, cmap = 'coolwarm', aspect='auto', vmin=tfr10mean-tfr10std, vmax=tfr10mean+tfr10std)#tfr10mean-tfr10std,
                        masked = np.ma.masked_where(np.log(tfr10[:,:])> 9.5,tfr10[:,:])
                        im4 = axs[3].imshow(masked,alpha=1,cmap = 'Greys', aspect='auto', vmin=tfr10mean-tfr10std, vmax=tfr10mean+tfr10std)#tfr10mean-tfr10std,

                        im4 = axs[3].plot(np.ones((tfr10.shape[1]))*np.where(freqs==freq_exp)[0],'g', linewidth = .8, alpha = 0.8)
                        axs[3].vlines(trainstart, axs[3].get_ylim()[0], axs[3].get_ylim()[1], 'k', linewidth =  .5)
                        axs[3].set_yticks([0, len(freqs)/2, len(freqs)])
                        axs[3].set_yticklabels(['0.02', '0.1', '0.18'])
                        axs[3].set_xlim(0,len(hanndata))
                        axs[3].set_xticklabels([])
                        axs[3].set_xticks([])
                        axs[3].set_ylim(0,len(freqs))

                    pp.savefig()#dpi=1200,transparent=False)
                    plt.close()
                    nr = nr+1

                #=========logpower============
                # Start end-report heatmap comparing the locations
                fig, axes = plt.subplots(1,2,sharey=True,gridspec_kw={'width_ratios': [16, 3]},figsize=(8.23,4))
                fig.suptitle('HBC (logpower)')
                fig.set_tight_layout(True)

                #here is where we do the log transform
                logBHmarker=logBHmarker[1:]
                #print(logBHmarker.shape)
                artmask = np.zeros((logBHmarker.shape));artmask[:]=True
                idx = np.where(artefacted_flag==1)[0]

                if len([idx])>0:
                    for i in idx:
                        artmask[i,:]=False
                        logBHmarker[i,:]=np.nan
                        axes[0].text((logBHmarker.shape[-1]-1)/2, i+.5,'Artefacted Data')

                locations = np.array(np.arange(1,logBHmarker.shape[0]+1),dtype=str)
                intensities = np.array(np.arange(0,logBHmarker.shape[1]),dtype=str)
                #print(locations)
                #print(intensities)
                scaledpower = np.round(np.nanmean(logBHmarker[:,3:],axis=1),decimals=2)
                #print(scaledpower.shape)
                scslopes=[];scrvalues=[];scpvalues=[]
                for s in range(logBHmarker.shape[0]):
                    if not np.isnan(logBHmarker[s,3]):
                        scslope, scintercept, scrvalue, scpvalue, _= linregress(np.arange(0,len(logBHmarker[s,3:])),logBHmarker[s,3:])
                        scslopes.append(scslope)
                        scrvalues.append(scrvalue)
                        scpvalues.append(np.round(scpvalue,decimals=2))
                    else:
                        scslopes.append(np.nan)
                        scrvalues.append(np.nan)
                        scpvalues.append(np.nan)

                logBHmarker = np.round(logBHmarker,decimals=2)
                #print(logBHmarker.shape)
                log = pd.DataFrame(logBHmarker.T, columns=locations, index=intensities)
                #        df = df.drop(0,axis=0)
                heatmap(log.transpose(),ax=axes[0],annot=True,square=False,cmap='Greys', cbar=False, vmin=2, vmax=14)
                b = logBHmarker.copy()
                b[:,:3]=0
                dflog = pd.DataFrame(b.T)
                #print(df2.shape)
                heatmap(log.transpose(),ax=axes[0],annot=True,square=False,cmap='Oranges', cbar=False, vmin=2, vmax=14, mask=(dflog.transpose().values<=9.5))
                heatmap(log.transpose(),ax=axes[0],annot=False,square=False,cmap='Greys', cbar=False, vmin=2, vmax=14, mask=artmask)

                axes[0].set_yticklabels(locations, rotation = 0, fontsize = 8)
                axes[0].set_xlabel('TMS pulse intensity')

                dfstats = pd.DataFrame([scaledpower, np.array(scslopes)**2, np.array(scaledpower*scslopes), np.array(scrvalues),np.array(scpvalues)], columns=locations,index=['pow','slope','p*s','r','p'])

                df2 = pd.concat([log,dfstats])
                df2.to_csv(startdir+'data/output/'+d+'/'+d+'_HBstats_logpower_deart.csv')
                heatmap(dfstats.transpose(),ax=axes[1],annot=True,cmap = 'Greens',cbar=False, annot_kws={"size": 6})

                axes[1].set_title('Statistics')
                pp.savefig(dpi=1200,transparent=False)
                plt.close()

                ##====Normalized normalized=====

                # Start end-report heatmap comparing the locations
                fig, axes = plt.subplots(1,2,sharey=True,gridspec_kw={'width_ratios': [16, 3]},figsize=(8.23,4))
                fig.suptitle(f'min max normalized power @ {freq_exp} Hz \nnormalized over locations (within subject)')
                fig.set_tight_layout(True)
                #print(f'normBHmarker_orig: {normBHmarker.shape}')
                normBHmarker = normBHmarker[1:]
                #print(f'normBHmarker: {normBHmarker}')
                artmask = np.zeros((normBHmarker.shape));artmask[:]=True
                idx = np.where(artefacted_flag==1)[0]
                #print(f'idx: {idx}')

                if len([idx])>0:
                    for i in idx:
                        artmask[i,:]=False
                        normBHmarker[i,:]=np.nan
                        axes[0].text((normBHmarker.shape[-1]-1)/2, i+.5,'Artefacted Data')

                #print(f'normBHmarker artmask: {normBHmarker}')

                scaledBHmarker = np.asarray((normBHmarker-np.nanmin(normBHmarker))/(np.nanmax(normBHmarker)-np.nanmin(normBHmarker)))#BHmarker[1:])/10
                scaledpower = np.round(np.mean(scaledBHmarker[:,3:],axis=1),decimals=2)#(powers-np.nanmin(powers))/(np.nanmax(powers)-np.nanmin(powers)),decimals=2)
                #print(f'scaledBHmarker: {scaledBHmarker.shape}')
                scslopes=[];scrvalues=[];scpvalues=[]
                for s in range(scaledBHmarker.shape[0]):
                    if not np.isnan(scaledBHmarker[s,3]):
                        #print(s)
                        scslope, scintercept, scrvalue, scpvalue, _= linregress(np.arange(0,len(scaledBHmarker[s,3:])),scaledBHmarker[s,3:])
                        scslopes.append(scslope)
                        scrvalues.append(scrvalue)
                        scpvalues.append(np.round(scpvalue,decimals=2))
                    else: 
                        scslopes.append(np.nan)
                        scrvalues.append(np.nan)
                        scpvalues.append(np.nan)

                scaledBHmarker = np.round(scaledBHmarker,decimals=2)
                #print(f'scalednormBHmarker: {scaledBHmarker.shape}')
                df = pd.DataFrame(scaledBHmarker.T, columns=locations, index=intensities)

                #print(f'df: {df.values.shape}')
                #        df = df.drop(0,axis=0)
                heatmap(df.transpose(),ax=axes[0],annot=True,square=False,cmap='Greys', cbar=False, vmin=0, vmax=1)
                #b = scaledBHmarker.copy()
                #b[:,:3]=0
                #df2 = pd.DataFrame(b.T)
                heatmap(df.transpose(),ax=axes[0],annot=True,square=False,cmap='Oranges', cbar=False, vmin=0, vmax=1, mask=(dflog.transpose().values<=9.5))
                heatmap(df.transpose(),ax=axes[0],annot=False,square=False,cmap='Greys', cbar=False, vmin=0, vmax=1, mask=artmask)

                axes[0].set_yticklabels(locations, rotation = 0, fontsize = 8)
                axes[0].set_xlabel('TMS pulse intensity')

                smarker = np.zeros((len(scpvalues)))
                smarker[np.array(scpvalues)<0.05]=1
                #determining maximum power
                pmarker = np.zeros((len(scaledpower)))
                pmarker[np.nanargmax(scaledpower)]=1

                maxp = np.nanargmax(scaledpower)
                maxr = np.nanargmax(scrvalues)
                sigslopes = np.where(np.array(scpvalues)<0.05)[0]
                if len(sigslopes)>0 and any(np.array(scrvalues)[sigslopes]>0):#only positive slopes
                    for marker in sigslopes[np.array(scrvalues)[sigslopes]>0]:
                        axes[0].add_patch(Rectangle((0,marker), 0, 1, edgecolor='lightskyblue', fill=False, lw=10))
                        axes[0].margins(x=0, y=0)
                        axes[0].get_yticklabels()[marker].set_weight('bold')
                        axes[0].get_yticklabels()[marker].set_size(12)
                        if marker>0:
                            axes[0].get_yticklabels()[marker].set_color('lightskyblue')
                axes[0].add_patch(Rectangle((0,maxp), 0, 1, edgecolor='dodgerblue', fill=False, lw=10))
                axes[0].margins(x=0, y=0)
                axes[0].get_yticklabels()[maxp].set_weight('bold')
                axes[0].get_yticklabels()[maxp].set_size(12)
                axes[0].get_yticklabels()[maxp].set_color('dodgerblue')

                dfstats = pd.DataFrame([scaledpower, np.array(scslopes)**2, np.array(scaledpower*scslopes), np.array(scrvalues),np.array(scpvalues)], columns=locations,index=['pow','slope','p*s','r','p'])

                df2 = pd.concat([df,dfstats])
                df2.to_csv(startdir+'data/output/'+d+'/'+d+'_HBstats_normalizedpowernorm_deart.csv')
                heatmap(dfstats.transpose(),ax=axes[1],annot=True,cmap = 'Greens',cbar=False, annot_kws={"size": 6})

                axes[1].set_title('Statistics')

                pp.savefig(dpi=1200,transparent=False)
                plt.close()


                try:
                    del fdata
                except:
                    pass
                try:
                    del BHmarker
                except:
                    pass
                
        except Exception as e:
            print(str(e),e.__traceback__.tb_lineno)
            try:
                del fdata
            except:
                pass
            try:
                del BHmarker
            except:
                pass

if __name__ == '__main__':
    if len(sys.argv)==1:
        print('[ERROR] subjects id (foldername) is needed to continue the analysis')
    if len(sys.argv) == 2:
        print('\n[INFO] use as python main_HBC_independent.py <subid> <in_dur> <out_dur> <repetitions> and <fsample>\n       now using defaults for QDS-HBC: python main_HBC_independent.py <subid> 5 11 16 130\n       if you are using another protocol replace <in_dur>, <out_dur>, <repetitions> and <fsample> with your own values.\n')
        subid = sys.argv[1]
        main_HBC_independent(subid)
    elif len(sys.argv)>2:
        subid = sys.argv[1]
        in_dur = int(sys.argv[2])
        out_dur = int(sys.argv[3])
        repetitions = int(sys.argv[4])
        main_HBC_independent(subid, in_dur, out_dur, repetitions)
