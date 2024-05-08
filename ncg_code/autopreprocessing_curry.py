#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:20:25 2019

@author: Brainclinics

copyright: Brainclinics Foundation, Nijmegen, the Netherlands

name: autopreprocessing.py

version: 1.0

log:
    10/12/2020 hvd: cleaned code for beta-testing
    07/11/2020 hvd: added correlation over channels as artifact determination for extremely
                        noisy data correlated with some external source
    01/11/2020 hvd: found bug in interpolating (computing the distance between channels) and fixed it
    17/01/2020 hvd: fixed missing channel issue for .edf --> filled with nan when not in measurement for preprocessing
                        and then repaired (with channel name in 'repaired' as well as 'empty' 'info' field)
    29/11/2019 hvd: fixed (channel) bug for .edf data reading
    11/11/2019 hvd: update preprocessing documentation
    16/10/2019 hvd: fixed bug in artifact_trl_samps subfunction
    15/10/2019 hvd: changed the classification of bad data, based on amount
                        of segments in addition to the already implemented amount of good data left.
    11/10/2019 hvd:
    30/09/2019 hvd: fixed several bugs for when some data is not available using
                        the perfect subject (signal generator) data
    30/09/2019 hvd: added option to skip notch-filter
    26/09/2019 hvd: changed the copy routine filename is now determined by ds.info['fileID']
    17/09/2019 hvd: added a function to collect residual mess, such as residual
                        eyeblinks or strange peaks at the end of processing.
    17/09/2019 hvd: removed EMG detection that I programmed in the BRC way. This
                        was not used.
    16/09/2019 hvd: for jump detection, made sure to do row (channelwise) filtering
                        and detection, also introduced absolute threshold (based on uV).
    09/09/2019 hvd: for EMG detection added an absolute threshold as well to
                        only select EMG instances that are evident and boxcarsmoothing
                        to pull toghether closely neighbouring EMG instances.
    05/09/2019 hvd: changes scipy.io into sio in 'save' module for .mat files
    01/09/2019 hvd: save pdf's in separate pdf folder
    30/08/2019 hvd: locked the version
    30/08/2019 hvd: changed the way the name of the outputfile is defined
    30/08/2019 hvd: changed the defaults to the ones that validated best

"""
#%%
# Import the python packages that are needed in multiple functions
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, hilbert, convolve, boxcar, medfilt
from scipy import signal
from scipy.signal.windows import hann
from scipy.stats import zscore
from mne.io import read_raw_curry, RawArray
#from mne.export import export_raw
#from code_BC_Syn001.inout import _export_raw
from mne import events_from_annotations, create_info
import mne
from pathlib import Path
import gc


# %% initiate object
class dataset:
    '''
        Create a dataset object to preprocess data, including:

        NAME:
            dataset

        DESCRIPTION:
            DataSet
        ================

        Parameters
        -----------------------------------------------------------------------
        filename:       name of the file that should be preprocesed, input files can
                        be in .csv or .edf format
        preprocpath:    path to the file that should be preprocessed.
        Fs:             sampling frequency of the data default Fs = 2048.

        -----------------------------------------------------------------------
        Provides
            1. an object as a dataset, with Brainclincis Diagnostics EEG formatting.
            2. EEG artifact correction and rejection functions described below.
            3. Options to save the preprocessed EEG datasets as .csv, .npy (pickeled),
                or .mat

        Returns:
        -----------------------------------------------------------------------
        A dataset object including information about
        - detected artifacts and their spatiotemporal locations
        - epoch information
        - original filename
        - sampling frequency
        - labels and neighbouring labels

        you can type dir(dataset) to see all the variables/functions that are
        included in the object

        '''

    def __init__(self, filename):
        # initiate place holders for the variables that will be created by the
        # functions within the class
        self.artifacts = {}
        self.info={}
        self.data = []
        self.trl = []
        self.artidata = []
        self.arttrl = []
        self.inname = filename #from where the input data originates

    def loaddata(self):
        '''
            Load data from the filename and path that was defined when initiating the
            object, without loading the data no preprocessing can be performed.

            Parameters:
            -------------------------------------------------------------------
            Are included in during the initiation of the dataset object.

            Returns
            -------------------------------------------------------------------
            The dataset object including the data.

        '''
        # extract the ID code of the subject from the filename (this filenaming is
        # standardized for Brainclinics Diagnostics EEG's)
        # subjID = self.info['fileID'].rsplit('/')[-1].split('.')[0]
        # #print feedback in the console so user can check if everything is running
        # #print('loading raw data from subject: '+ subjID)
        ''' reading the data '''
        ''' determine the extention of the data '''
        # read the data in the right order. In the .csv files extracted from the .brc
        # files there are a lot of spaces included into the heading. Make sure this
        # is in the same order


        if self.inname[-4:] =='.cdt':
            raw = read_raw_curry(self.inname)
            self.data = raw.get_data().astype(float)*1e6#unit was V now in uV
            self.labels= np.array(raw.info['ch_names'])
            self.sF = int(raw.info['sfreq'])
            self.info = dict(raw.info)
            self.info['fileID']=self.inname
            self.events,self.events_id =  events_from_annotations(raw, event_id=int) ##!TODO revert change

    def rereference(self, newrefchan = 'LM'):

        ref = np.empty(self.data[1,:].shape);ref[:]=np.nan
        
        if newrefchan == 'LM':
            eegchans = []
            for s in range(len(self.info['chs'])):
                if int(self.info['chs'][s]['kind'])==2 and not 'M' in self.info['chs'][s]['ch_name']:
                    eegchans.append(s)
            eegchans = np.array(eegchans)
            
            idx = np.where(np.logical_or(self.labels=='M1',self.labels=='M2'))[0]
            ref = np.nanmean(self.data[idx,:], axis=0)

            m1c7 = self.data[np.where(self.labels=='M1')[0][0],:]-self.data[np.where(self.labels=='C7')[0][0],:]
            #m1c7 = m1c7 - np.mean(m1c7)
            m2c7 = self.data[np.where(self.labels=='M2')[0][0],:]-self.data[np.where(self.labels=='C7')[0][0],:]
            #m2c7 = m2c7 - np.mean(m2c7)
            self.labels[np.where(self.labels=='C7')[0]] = 'Cz'

            self.data =np.vstack((self.data, m1c7, m2c7))
            self.labels = np.hstack((self.labels, 'M1-C7','M2-C7'))
            for s in np.arange(0,29):
                self.data[s,:] = self.data[s,:] - ref

        elif newrefchan == 'avgref':
            O2idx=np.where(self.labels=='O2')[0][0]
            #print(O2idx)
            eegchans = np.arange(0,O2idx+1)
            ref = np.nanmean(self.data[:O2idx+1,:],axis =0)
            for s in eegchans:#np.arange(0,O2idx+1):
                self.data[s,:] = self.data[s,:] - ref
        else:
            idx = np.where(self.labels==newrefchan)
            ref = np.nanmean(self.data[idx,:])

            for s in np.arange(0,29):
                self.data[s,:] = self.data[s,:] - ref

        if newrefchan == 'LM':
            self.data[np.where(self.labels=='Cz')[0],:] = -1*ref
        self.n_eegchans = len(np.arange(0,29))

    def bipolarEOG(self):
        '''
            Compute the bipolar EOG from the ['VPVA','VNVB','HPHL','HNHR'] EOG
            recordings (only for the raw (.csv) data!!).

            Parameters:
            -------------------------------------------------------------------
            Are included in during the initiation of the dataset object.

            Returns
            -------------------------------------------------------------------
            The dataset object including the data, the ['VPVA','VNVB','HPHL','HNHR']
            replaced by ['VEOG','HEOG'].
        '''
        #HEOG
        HPHL = np.where(self.labels=='FHL11')[0][0];HNHR = np.where(self.labels=='FHR12')[0][0]
        #channels 0-n_eegchans are EEG channels 30+ are additional channels
        self.data = np.vstack((self.data,[(self.data[HPHL,:] - self.data[HNHR,:])]))
        self.labels = np.hstack((self.labels,['HEO']))

    def correct_order(self):
        df = pd.DataFrame(self.data.T,columns=self.labels)
        ch_order = ['FHL11','Fp1','Fp2','FHR12','AFz','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','O1','O2','VEO','HEO','M1-C7','M2-C7','Trigger']
        df = df[ch_order]
        self.labels = np.array(df.keys())
        self.data = df.values.transpose()
        setattr(self, "n_eegchans", int(np.where(self.labels=='O2')[0][0])+1)

    def demean(self):
        '''
            Subtract the overal mean (over time) from each EEG and EOG channel, as a sort of
            baseline correction to have all data revolve around the zero line.
            Parameters:
            -------------------------------------------------------------------
            Are included in during the initiation of the dataset object.

            Returns
            -------------------------------------------------------------------
            The dataset object with the data demeaned.
        '''

        self.data[:self.n_eegchans,:] = self.data[:self.n_eegchans,:]-(np.nanmean(self.data[:self.n_eegchans,:],axis=1).reshape((self.data[:self.n_eegchans,:].shape[0],1)))
        self.info['demeaned']= 'all channels'

    def apply_filters(self, trlpadding=10, hpfreq=0.5, lpfreq=100, notchfilt = 'yes', notchfreq=50, Q=100):
        '''
            Apply filters, to remove low frequency trends, high-, and notch frequencies.
            - a bidirectional (zero phase) IIR filter will be aplied for the
                notch frequency.
            - for the high as wel as low pass filter a bidirectional (zero phase)
                a 4rth orde Butterworth filter is applied.
            - order in which the filters are applied:
                1. notch
                2. hp
                3. lp

            Parameters:
            -------------------------------------------------------------------
            - notchfreq: notch frequency, float
            - hpfreq: highpass frequency, float > 0.5
            - lpfreq: lowpass frequency, float

            Returns
            -------------------------------------------------------------------
            The dataset object with filtered data.

        '''
        # Determine the nyquist frequency for filtering
        nyq = 0.5 * self.sF
        for r in range(self.n_eegchans+4):
            if notchfilt=='yes':
                ''' notch filter '''
                b, a = iirnotch(notchfreq, Q, fs=self.sF)
                data = filtfilt(b, a, self.data[r,:])

            ''' highpass filter '''
            normal_cutoff = hpfreq / nyq
            b, a = butter(4, normal_cutoff, btype='highpass', analog=False)
            hpdata = filtfilt(b, a, data)
            ''' lowpass filter '''
            normal_cutoff = lpfreq / nyq
            b, a = butter(4, normal_cutoff, btype='lowpass', analog=False)
            self.data[r,:] = filtfilt(b, a, hpdata)

        del hpdata, data, b, a
        self.info['filtered']= ['hp: '+str(hpfreq) +' ,lp: '+ str(lpfreq) + ' ,notch: '+str(notchfreq)]

    def apply_bpfilter(self,freqrange):
        nyq = 0.5 * self.sF
        high_pass = freqrange[0] / nyq
        low_pass = freqrange[1] / nyq

        ''' bandpassfilter '''
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        self.data = sosfiltfilt(sos, self.data)

    def export_edf(self,outpath):
        name,_ = os.path.splitext(os.path.split(self.info['fileID'])[1])
        ch_types = ['eeg' for e in range(self.n_eegchans)]
        ch_types.append('eog');ch_types.append('eog');ch_types.append('misc');ch_types.append('misc');ch_types.append('misc')
        info = create_info(list(self.labels),self.sF,ch_types)
        raw = RawArray(self.data*1e-6, info, first_samp=0, copy='auto')
        export_raw(outpath+name+'.edf', raw)

    def extract_conditions(self):

        conds={3:'EO',5:'EC',4:'ODDB',9:'BREATH'}

        trigs = np.array(self.events[:,2])
        trigtimes = np.array(self.events[:,0])

        trigs[np.where(trigs==1)]=99
        trigs[np.where(trigs==2)]=111
        trigs2 = trigs[np.where(trigs<90)[0]]
        trigtimes = trigtimes[np.where(trigs<90)[0]]
        trigs = trigs2
        events=[];eventtimes=[]
        for t in range(len(trigs[:-2])):
            if trigs[t] in conds:
                eventarr = np.array(trigs[t:t+3])
                timesarr = np.array(trigtimes[t:t+3])
                if eventarr[1]==eventarr[0] and eventarr[2]==eventarr[0]:
                    events.append(eventarr)
                    eventtimes.append(timesarr)
        events = np.array(events)
        eventtimes = np.array(eventtimes)
        a = 0
        for e in range(events.shape[0]):
            if a == events.shape[0]:
                continue
            if a == events.shape[0]-1:
                ##print(conds[events[a][0]])
                time = (eventtimes[a][1]-eventtimes[a][0])/self.sF
                minutes = int(time/60)
                seconds = int(time%60)
                ##print(str(minutes)+':'+str(seconds)+' minutes')
                setattr(self, conds[events[a][0]], self.data[:,eventtimes[a][0]:eventtimes[a][1]])
            elif events[a][0] == events[a+1][0]:
                ##print(conds[events[a+1][0]])
                ##print('two recordings, selecting the last')
                time = (eventtimes[a+1][1]-eventtimes[a+1][0])/self.sF
                minutes = int(time/60)
                seconds = int(time%60)
                ##print(str(minutes)+':'+str(seconds)+' minutes')
                setattr(self, conds[events[a+1][0]], self.data[:,eventtimes[a+1][0]:eventtimes[a+1][1]])
                a=a+1
            elif events[a+1][0] != events[a][0]:
                #print(conds[events[a][0]])
                time = (eventtimes[a][1]-eventtimes[a][0])/self.sF
                minutes = int(time/60)
                seconds = int(time%60)
                ##print(str(minutes)+':'+str(seconds)+' minutes')
                setattr(self, conds[events[a][0]], self.data[:,eventtimes[a][0]:eventtimes[a][1]])
            a=a+1


    def correct_EOG(self, lpfreq = 15, vthreshold = 1, vpadding = 3, hthreshold = 0.2, hpadding =0.3):
        '''
            Detect eyemovements and regress them onto the EEG data for each
            channel. Then remove the modeled eyemovement deflection from the
            data in the EEG channels.

            - * - * - * - * - * -
            |
            *  Method adapted from Gratton et al. 1999):
            |  compute the deviations/ regressioncoefficients with the EEG and
            *  subtract the 'weighted' VEOG and HEOG
            |
            *  numpy.linalg.lstsq(a, b, rcond='warn')
            |
            *  Solves the equation a x = b by computing a vector x that minimizes
            |  the Euclidian 2-norm EOG*x = EEG
            *  Gratton --> EEG = k * EOG
            |
            - * - * - * - * - * -

            The function first takes care of the vertical eyemovements (blinks)
            and then the horizontal eyemovements.

            Parameters:
            -------------------------------------------------------------------
            - lpfreq:       lowpass frequency, float, default = 15
            - threshold:    threshold, z-value, default = 0.5
            - padding:      padding around where the threshold is reached,
                            relative to the length of the artifact, default = 2
                            (from experience 2 times the length of the artifact
                            before and after the artifact reaches the best results)


            Returns
            -------------------------------------------------------------------
            The dataset object with the data corrected for the vertical and
            horizontal eyemovements, and the start and end sample of each artifact
            that was detected can be found in the artifact field.

        '''

        eye_channel = ['VEO','HEO']
        trlpadding = 1

        #TODO include actual padding here
        Aweight = np.zeros((len(eye_channel),self.n_eegchans,int(2*trlpadding*self.sF)+self.data.shape[1]))
        datapaddedEOG = np.zeros((len(eye_channel),int(2*trlpadding*self.sF)+self.data.shape[1]))

        for n in range(len(eye_channel)):
            #print(n)
            if n == 0:
                padding = vpadding
                threshold = vthreshold
            elif n == 1:
                padding = hpadding
                threshold = hthreshold
            #Atrl = []
            EOG = self.data[np.where(self.labels==eye_channel[n])[0]][0]

            '''initiate variables'''
            hilEOG = np.zeros((1,self.data.shape[1]))
            hilbEOG = hilEOG.copy()
            filtEOG = hilEOG.copy()

            '''filter the EOG'''
            nyq = 0.5 * self.sF
            normal_cutoff = lpfreq / nyq
            b, a = butter(4, normal_cutoff, btype='lowpass', analog=False)
            filtEOG = filtfilt(b, a, EOG); del b,a,EOG
            hilEOG  = hilbert(filtEOG.copy(), N=np.int(len(filtEOG)+
                                             len(filtEOG)*0.20), axis = -1);
            hilbEOG = hilEOG[:len(filtEOG)]; del hilEOG
            amplenv = np.abs(hilbEOG); del hilbEOG

            boxdata = convolve(amplenv, boxcar(np.int(0.2*self.sF)), mode ='same', method ='direct'); del amplenv

            '''.........................................................................'''
            ''' Regression NUMPY way'''
            ''' compute the deviations/ regressioncoefficients with the EEG             '''
            ''' and subtract the 'weighted' VEOG and HEOG                               '''
            ''' numpy.linalg.lstsq(a, b, rcond='warn')                                  '''
            ''' Solves the equation a x = b by computing a vector x that minimizes      '''
            ''' the Euclidean 2-norm    EOG*x=EEG                                       '''
            ''' Gratton --> EEG = k * EOG                                               '''
            '''.........................................................................'''

            ''' apply datapadding '''

            datapaddedEOG = np.hstack((filtEOG[:trlpadding*self.sF],filtEOG,filtEOG[len(filtEOG)-trlpadding*self.sF:]))
            datapaddedboxdata = np.hstack((boxdata[:trlpadding*self.sF],boxdata,boxdata[len(filtEOG)-trlpadding*self.sF:]))
            del boxdata
            Atrl, Asamps = self._detect_artifact(datapaddedboxdata,threshold);del datapaddedboxdata

            datapaddeddata = np.hstack((self.data[:,:trlpadding*self.sF],self.data[:,:],self.data[:,len(filtEOG)-trlpadding*self.sF:]))
            del filtEOG
            artsamples = np.zeros(datapaddeddata.shape[1],dtype=int)
            if len(Asamps) > 0:
                for i in range(1,Atrl.shape[0]):
                    if Atrl[i,0]==0:
                        artsamples[0:Atrl[0,1]+np.int((Atrl[0,1]-0)*padding)]=1
                    elif Atrl[i,1]==datapaddeddata.shape[1]:
                        artsamples[Atrl[i,0]-np.int((Atrl[i,1]-Atrl[i,0])*padding):datapaddeddata.shape[1]]=1
                    else:
                        artsamples[Atrl[i,0]-np.int((Atrl[i,1]-Atrl[i,0])*padding):Atrl[i,1]+np.int((Atrl[i,1]-Atrl[i,0])*padding)]=1
            ''' Define the starts and endings of the collapsed EOG artifacts '''
            p = np.where(artsamples==1)[0]
            startidxs=0
            if len(p) > 1:
                if p[0]==0:
                    startidxs = np.append(startidxs,0)

                startidxs = np.append(startidxs,[np.where(np.diff(artsamples)==1)[0]+1])# diff =1
                startidxs = startidxs[1:]

                endidxs = np.hstack([np.where(np.diff(artsamples)==-1)[0]+1]); del artsamples#diff = -1
                if len(endidxs)<len(startidxs):
                    endidxs = np.append(endidxs,datapaddeddata.shape[1])

                ARTtrl = np.array([0,0],dtype=int)
                for i in range(len(startidxs)):
                    ARTtrl = np.vstack((ARTtrl,[startidxs[i],endidxs[i]]))
                ARTtrl = ARTtrl[1:]

                ##print('Eye artifact correction: correcting '+str(ARTtrl.shape[0])+ ' '+ eye_channel[n] + ' eye artifact(s)')
                self.artifacts[eye_channel[n]] =  ARTtrl

                EOGweight = np.zeros((self.n_eegchans,datapaddeddata.shape[1]))
                
                Atmpweight = EOGweight.copy()

                EOG_row_vec = datapaddedEOG.reshape((datapaddedEOG.shape[0], 1))

                newdata = np.zeros((self.n_eegchans,datapaddeddata.shape[-1]))
                for r in range(self.n_eegchans):
                    for k in range(ARTtrl.shape[0]):
                        ''' convolve with taper to avoid inducing broadband signal into EEG...
                        channels (jump) '''
                        artbegin = ARTtrl[k,0]
                        artend = ARTtrl[k,1]


                        Atmpweight[r,artbegin:artend] = np.linalg.lstsq(EOG_row_vec[artbegin:artend],datapaddeddata[r,artbegin:artend],rcond=None)[0]

                        Attaper = signal.tukey(len(np.arange(artbegin,artend)),alpha = 0.025)
                        EOGweight[r,artbegin:artend] = Attaper*Atmpweight[r,artbegin:artend]
                        
                    ''' and correct EOG '''
                    newdata[r,:] = datapaddeddata[r,:]-((EOGweight[r,:])*datapaddedEOG)
                    
                self.data[:self.n_eegchans,:] = newdata[:,(trlpadding*self.sF):datapaddedEOG.shape[0]-(trlpadding*self.sF)]
                del newdata
                self.info[eye_channel[n]] = str(ARTtrl.shape[0])+ ' artifacts detected @ threshold: '+str(threshold)+' and corrected'

            else:
                ARTtrl = []
                #print('Eye artifact correction: correcting 0 '+ eye_channel[n] + ' eye artifact(s)')
                self.info[eye_channel[n]] = '0 artifacts detected @ threshold: '+str(threshold)+' and corrected'
            gc.collect()


    def detect_emg(self, hpfreq = 75, lpfreq = 95, threshold = 4, padding=0.1):
        ''' Detect EMG activity in the complete EEG channel set
            Parameters:
            -------------------------------------------------------------------
            - hpfreq: highpass frequency
            - lpfreq: lowpass frequency
            - threshold: no. standard deviations
            - padding: padding the artifact (in seconds)

            Returns
            -------------------------------------------------------------------
            The dataset object with the start and end sample of each artifact
            that was detected can be found in the artifact field. The start and
            endsample of each artifact is kept in the artifacts field for later
            removal.
        '''

        nyq = 0.5 * self.sF
        high_pass = hpfreq / nyq
        low_pass = lpfreq / nyq

        ''' bandpassfilter '''
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        filtEMG = sosfiltfilt(sos, self.data)

        N=np.int(filtEMG.shape[1])#+filtEMG.shape[1]*0.10)
        if N % 2 == 0:
            N=N
        else:
            N=N+1

        hilbEMG  = hilbert(filtEMG.copy(), N=N, axis = -1)
        amplenv = np.abs(hilbEMG[:,:filtEMG.shape[1]])

        # number of EEG channels
        EMGsamps = np.zeros((self.n_eegchans,self.data.shape[1]))

        hanndata = np.zeros((self.n_eegchans,self.data.shape[1]))
        ''' hanning smooth '''
        for r in range(self.n_eegchans):
            hanndata[r,:] = convolve(amplenv[r,:], hann(np.int(0.5*self.sF),sym=True), mode ='same')#, method ='direct')
        ''' zvalue threshold '''
        Zdata = zscore(hanndata,axis=1)
        tmpEMGsamps = np.zeros((hanndata.shape[0],hanndata.shape[1]))
        inpEMGsamps = np.zeros((hanndata.shape[0],hanndata.shape[1]))

        for r in range(hanndata.shape[0]):
            if ~np.isnan(Zdata[r,0]):
                sidx = np.where(Zdata[r,:] > np.nanmean(Zdata)+threshold)[0]
                #introduce an absolute threshold to extract only EMG data that is evident?
                didx = np.where(amplenv[r,sidx]>3)
                tmpEMGsamps[r,sidx[didx]]=1
                boxdata = convolve(tmpEMGsamps[r,:], boxcar(np.int(0.5*self.sF)), mode ='same', method ='direct')
                inpEMGsamps[r,np.where(boxdata>0)]=1

        EMGtrl, EMGsamps = self._artifact_samps_trl(inpEMGsamps, padding,self.sF, self.data[-1].shape[0])

        #print('EMG detection: detected '+str(len(EMGtrl))+' artifact(s)')

        self.info['EMG detection'] = str(len(EMGtrl))+' artifacts detected @ threshold: Z = '+str(threshold)
        self.artifacts['EMGsamps'] = EMGsamps
        self.artifacts['EMGtrl'] = EMGtrl

    def detect_jumps(self, padding=0.01, threshold = 5):
        '''
            Detect artifactual sharp baseline shifts or jumps in the data, for each
            channel. The start and ensample of each artifact is kept in the artifacts
            field for later removal.

            Parameters:
            -------------------------------------------------------------------
            - threshold: z-value, default = 2
            - padding: padding around the artifact in seconds, default = 0.05

            Returns
            -------------------------------------------------------------------
            The dataset object with the start and end sample of each artifact
            that was detected included in the artifacts field.
        '''


        inpJUMPsamps = np.zeros((self.n_eegchans,self.data.shape[1]))
        filtdata = np.zeros(self.data.shape)
        diffdata = np.zeros(self.data.shape)
        Zdata = np.zeros(self.data.shape)
        for r in range(self.n_eegchans):
            if ~np.isnan(self.data[r,0]):
                filtdata[r,:] = medfilt(self.data[r,:],kernel_size=(9))
                diffdata[r,1:] = np.abs(np.diff(filtdata[r,:],n=1))
                Zdata[r,:] = zscore(diffdata[r,:])
                sidx = (np.where(Zdata[r,:] > np.nanmean(Zdata)+threshold)[0])
                didx = np.where(diffdata[r,sidx]>30)[0]
                inpJUMPsamps[r,sidx[didx]]=1

        JUMPtrl, JUMPsamps = self._artifact_samps_trl(inpJUMPsamps, padding, self.sF, self.data[-1].shape[0])

        #print('Jump/ baseline shift : '+str(len(JUMPtrl))+ ' jumps/baselineshifts detected')

        self.info['jump detection'] = str(len(JUMPtrl))+' jumps/baseline shifts detected @ threshold: Z = '+str(threshold)
        self.artifacts['JUMPsamps'] = JUMPsamps
        self.artifacts['JUMPtrl'] = JUMPtrl


    def detect_kurtosis(self, threshold=8, padding=0.1, overlap=0.1, winlen = 4):
        '''
            Detect segments where there is extreme kurtosis, for each
            channel. This is performed on a moving window with overlap, so only
            the parts of the data where kurtosis is extreme are marked for later
            removal, and not entire trials will have to be discarded.

            Parameters:
            -------------------------------------------------------------------
            - threshold:    z-value, default = 4
            - padding:      padding around the artifact in seconds, default = 0.1
            - overlap:      amount of overlap of the moving windows, defines the resolution,
                            in seconds, default = 0.05
            - winlen:       length of the sliding windows to evaluate the kurtosis

            Returns
            -------------------------------------------------------------------
            The dataset object with the start and end sample of each artifact
            that was detected included in the artifacts field.

        '''

        from scipy.stats import kurtosis

        if winlen == 'all':
            winlen = self.data.shape[-1]/self.sF

        winstarts = np.arange(0,self.data.shape[1]-(winlen*self.sF),overlap*self.sF)
        winends = winstarts+winlen*self.sF

        kurt = np.zeros((self.n_eegchans,self.data.shape[-1]))
        inpKURTsamps = kurt.copy()
        for r in range(self.n_eegchans):
            if ~np.isnan(self.data[r,0]):
                for w in range(len(winstarts)):
                    kurt[r,np.int(winstarts[w]):np.int(winends[w])] = kurtosis(self.data[r,np.int(winstarts[w]):np.int(winends[w])],fisher = True)

                if len(np.where(kurt[r,:]>threshold)[0]) > 0:
                    inpKURTsamps[r,np.where(kurt[r,:]>threshold)[0]]=1

        del kurt

        KURTtrl, KURTsamps = self._artifact_samps_trl(inpKURTsamps, padding, self.sF, self.data[-1].shape[0])

        if len(KURTtrl)>0:
            self.artifacts['KURTsamps'] = KURTsamps
            #print('kurtosis: '+str(KURTtrl.shape[0])+ ' samples with kurtosis detected')
            self.info['kurtosis detection'] = str(len(KURTtrl))+' samples with kurtosis detected @ threshold: Z = '+str(threshold)
        else:
            KURTtrl = np.array([])
            #print('kurtosis: 0 samples with kurtosis detected')
            self.info['kurtosis detection'] = '0 samples with kurtosis detected @ threshold: Z = '+str(threshold)
        self.artifacts['KURTtrl'] = KURTtrl


    def detect_extremevoltswing(self, threshold = 200, padding = 0.05, overlap = 0.05, winlen = 0.5):

        '''
        Detect segments where there is extreme voltage swing, for each
        channel. This is performed on a moving window with overlap, so only
        the parts of the data where voltage is extreme are marked for later
        removal, and not entire trials will have to be discarded.

        Parameters:
        -------------------------------------------------------------------
        - threshold:    z-value, default = 120
        - padding:      padding around the artifact in seconds, default = 0.1
        - overlap:      amount of overlap of the moving windows, defines the resolution,
                        in seconds, default = 0.05
        - winlen:       length of the sliding windows to find the maximum and minimum voltage
                        and evaluate the swing

        Returns
        -------------------------------------------------------------------
        The dataset object with the start and end sample of each artifact
        that was detected included in the artifacts field.

        '''
        if winlen == 'all':
            winlen = self.data.shape[-1]/self.sF

        winstarts = np.arange(0,self.data.shape[1]-(winlen*self.sF),overlap*self.sF)
        winends = winstarts+winlen*self.sF

        swing = np.zeros((self.n_eegchans,self.data.shape[-1]))
        inpSWINGsamps = np.zeros((self.n_eegchans,self.data.shape[-1]))
        for r in range(self.n_eegchans):
            if ~np.isnan(self.data[r,0]):
                for w in range(len(winstarts)):
                    swing[r,np.int(winstarts[w]):np.int(winends[w])] = np.nanmax(self.data[r,np.int(winstarts[w]):np.int(winends[w])])-np.nanmin(self.data[r,np.int(winstarts[w]):np.int(winends[w])])
                if len(np.where(np.abs(swing[r,:])>threshold)[0]) > 0:
                    inpSWINGsamps[r,np.where(swing[r,:]>threshold)[0]]=1

        del swing

        SWINGtrl, SWINGsamps = self._artifact_samps_trl(inpSWINGsamps, padding, self.sF, self.data[-1].shape[0])

        if len(SWINGtrl)>0:
            self.artifacts['SWINGsamps'] = SWINGsamps
            #print('swing-detection: '+str(SWINGtrl.shape[0])+ ' samples with extreme voltage swing detected')
            self.info['swing-detection'] = str(len(SWINGtrl))+' samples with extreme voltage swing detected @ threshold: Z = '+str(threshold)
        else:
            SWINGtrl = np.array([])
            #print('swing-detection: 0 samples with swing-detection')
            self.info['swing-detection'] = '0 samples with extreme voltage swing @ threshold: Z = '+str(threshold)

        self.artifacts['SWINGtrl'] = SWINGtrl

    def residual_eyeblinks(self, threshold = 0.5, padding = 0.1):
        nyq = 0.5 * self.sF
        high_pass = 0.5 / nyq
        low_pass = 6/ nyq

        ''' bandpassfilter '''
        sos = butter(4, [high_pass, low_pass], btype='bandpass', analog=False, output = 'sos')
        filtEB = sosfiltfilt(sos, self.data)
#        filtpadding = filtEB.shape[1]*0.10

        N=np.int(filtEB.shape[1]+filtEB.shape[1]*0.10)
        if N % 2 == 0:
            N=N
        else:
            N=N+1

        hilbEB  = hilbert(filtEB.copy(), N=N, axis = -1)
        amplenv = np.abs(hilbEB[:,:filtEB.shape[1]])

        EBsamps = np.zeros((self.n_eegchans,self.data.shape[1]))

        hanndata = np.zeros((self.n_eegchans,self.data.shape[1]))
        ''' hanning smooth '''
        for r in range(self.n_eegchans):
            hanndata[r,:] = convolve(amplenv[r,:], hann(np.int(1*self.sF),sym=True), mode ='same')#, method ='direct')

        ''' zvalue threshold '''
        Zdata = zscore(hanndata, axis=1)

        inpEBsamps = np.zeros((self.data.shape[0],hanndata.shape[1]))

        for r in range(hanndata.shape[0]):
            if ~np.isnan(self.data[r,0]):
                sidx = np.where(Zdata[r,:] > np.nanmean(Zdata)+threshold)[0]
                didx = np.where(amplenv[r,sidx]>60)
                inpEBsamps[r,sidx[didx]]=1

        EBtrl, EBsamps = self._artifact_samps_trl(inpEBsamps, padding, self.sF, self.data[-1].shape[0])

        #print('EB detection: detected '+str(len(EBtrl))+' artifact(s)')

        self.info['EB detection'] = str(len(EBtrl))+' artifacts detected @ threshold: Z = '+str(threshold)
        self.artifacts['EBsamps'] = EBsamps
        self.artifacts['EBtrl'] = EBtrl

    def define_artifacts(self, time_threshold = 1/3, z_threshold = 1.96):
        '''
            Define the artifacts that were detected, taking care of possible
            overlap in artifacts.

            Bad channels: If the amount of artifacts in a channel exceed
            a relative amount of time the channel will be marked as bad, and repaired
            by interpolation through a weighted average (based on the Euclidian
            distance) of selected neighbouring channels.

            Bridging channels are also defined in this function and repaired.

            Parameters:
            -------------------------------------------------------------------
            - time_threshold: ratio of alowed amount of time removed for artifactual
                            data, if this is exceeded the channel will be
                            marked as bad, and repaired,  default = 1/3
            - ztheshold: amount of standard-deviations alowed for general broadband signal

            Returns
            -------------------------------------------------------------------
            The dataset object with the start and end sample of each artifact
            that was detected included in the artifacts field.
        '''

        if 'EMGtrl' in self.artifacts and len(self.artifacts['EMGtrl'])>0:
            emgtrl = self.artifacts['EMGtrl']
            emgsamps = self.artifacts['EMGsamps']
        else:
            emgtrl = []
            emgsamps = []
        if 'JUMPtrl' in self.artifacts and len(self.artifacts['JUMPtrl'])>0:
            jmptrl = self.artifacts['JUMPtrl']
            jmpsamps = self.artifacts['JUMPsamps']
        else:
            jmptrl = []
            jmpsamps = []

        if 'KURTtrl' in self.artifacts and len(self.artifacts['KURTtrl'])>0:
            kurttrl = self.artifacts['KURTtrl']
            kurtsamps = self.artifacts['KURTsamps']
        else:
            kurttrl = []
            kurtsamps = []

        if 'SWINGtrl' in self.artifacts and len(self.artifacts['SWINGtrl'])>0:
            swingtrl = self.artifacts['SWINGtrl']
            swingsamps = self.artifacts['SWINGsamps']
        else:
            swingtrl = []
            swingsamps = []

        if 'EBtrl' in self.artifacts and len(self.artifacts['EBtrl'])>0:
            ebtrl = self.artifacts['EBtrl']
            ebsamps = self.artifacts['EBsamps']
        else:
            ebtrl = []
            ebsamps = []

        '''========= colapse the artifacts ======== '''
        ''' the EMG artifacts, some might be overlapping because of artifact padding
        that will be dealt with here '''

        artsamps = np.zeros((self.data.shape[0],self.data.shape[1]),dtype=int)

        for r in range(self.n_eegchans):
            if len(emgtrl) > 0:
                artsamps[r,np.where(emgsamps[r,:]==1)[0]]=1
            ''' the JUMP/ baseline shift artifacts '''
            if len(jmptrl) > 0:
                artsamps[r, np.where(jmpsamps[r,:]==1)[0]]=1
            ''' the kurtosis artifacts '''
            if len(kurttrl) > 0:
                artsamps[r, np.where(kurtsamps[r,:]==1)[0]]=1
            if len(swingtrl) >0:
                artsamps[r, np.where(swingsamps[r,:]==1)[0]]=1
#            if len(extrtrl) >0:
#                artsamps[r, np.where(extrsamps[r,:]==1)[0]]=1
            if len(ebtrl) >0:
                artsamps[r, np.where(ebsamps[r,:]==1)[0]]=1


        '''======= check for bad channels and if they are there, interpolate  ======='''
        '''if more than the threshold of time (default is half of the time) is
        rejected in one channel --> channel = bad '''

        badchan = np.zeros((self.n_eegchans),dtype = int)
        for r in range(self.n_eegchans):#the VEOG channels doesn't count :)
            if len(np.where(artsamps[r,:]==1)[0]) > self.data.shape[1]*time_threshold:
                badchan[r]=1

        '''======= check if a channel is bad (high broadband / EMG signal) for the whole measurement ======='''
        '''default threshold = 1.96 '''
        from scipy.fftpack import fft
        from scipy.signal import hann

        hannwin = hann(np.int(self.data.shape[-1]))
        power = np.abs(fft(self.data[:self.n_eegchans,:])*hannwin)**2
        freqs = np.linspace(0, self.sF/2,int(len(power[0,:])/2))
        fid = [(np.where((freqs > 55) & (freqs < 95)))][0][0]
        #fid2 = np.append(fid,[np.where((freqs > 55) & (freqs < 95))][0][0])
        overallpower = power[:self.n_eegchans,fid]
        zdat = np.nanmean(zscore(overallpower,axis=0),axis=1)
        badchan[np.where(zdat> np.nanmean(zdat)+ z_threshold)[0]] = 1
        idxbadchan = np.where(badchan ==1)[0]

        self.artifacts['bad channels']=[]
        if len(idxbadchan)>0:
            self.info['bad channels'] = str(len(idxbadchan)) + ' detected @ threshold: '+str(time_threshold)
            for b in range(len(idxbadchan)):
                self.artifacts['bad channels'] = np.append(self.artifacts['bad channels'],self.labels[idxbadchan[b]])
        else:
            self.info['bad channels'] = '0 bad channels @ threshold: '+ str(time_threshold)

        '''======== check for bridging (electronic distance (Alschuler et al. 2014, Tenke & Kaiser, 2001)) ======='''
        bridgeidx = self._bridging_check(self.data)[0]
        self.artifacts['bridging channels'] = []
        if len(bridgeidx)>0:
            ##print('reparing bridging channels')
            self.info['bridging channel check'] = 'reparing bridging channels'
            for b in range(len(bridgeidx)):
                self.artifacts['bridging channels'] = np.append(self.artifacts['bridging channels'],self.labels[bridgeidx[b]])
        else:
            self.info['bridging channels'] = str(0)+' briding channels'

        '''======== check for empty (nan) channels ========'''
        idxemptychan = np.where(np.isnan(self.data[:self.n_eegchans,1]))[0]
        self.artifacts['empty channels'] = []
        if len(idxemptychan)>0:
            self.info['empty channels'] = str(len(idxemptychan)) + 'empty channels detected'
            for b in range(len(idxemptychan)):
                self.artifacts['empty channels'] = np.append(self.artifacts['empty channels'],self.labels[idxemptychan[b]])
        else:
            self.info['empty channels'] = str(0)+' empty channels'

        '''======== colapse badchannel and bridging channel array ======='''
        combidx=np.array((np.unique(np.hstack((idxbadchan,bridgeidx,idxemptychan))))).astype(int)

        '''======== interpolate the data based on the average of neighbouring channels (using the Euclidian Distance) ========'''
        if len(combidx)>=1:
            repaireddata =np.zeros((self.data.shape))
            #print('Remove artifacts: repairing/ interpolating bad, empty and bridging channel(s) \n')
            repaireddata, self.info['repairing channels'], repaired, intchan = self._interpolate_data(self.data,combidx)
            if repaired == 'yes':
                self.info['repaired channels'] = []
                self.data = repaireddata
                for b in range(len(intchan)):
                    self.info['repaired channels'] = np.append(self.info['repaired channels'],self.labels[intchan[b]])
                    artsamps[intchan[b],:] = 0
            elif repaired == 'no':
                self.info['data quality'] = 'bad'

        artsamples = np.nanmax(artsamps,axis=0)
        if  len(np.where(artsamples==1)[0]) > self.data.shape[-1]*(1-time_threshold) or len(combidx)>3:# if 2/3 of the data is artifacts or there are 6 bad channels...
            self.info['data quality'] = 'bad'
        else:
            self.info['data quality'] = 'OK'

        Och = np.squeeze(np.where(np.array(self.labels)=='O2')[0])
        self.trl = np.array([0,self.data.shape[-1]],dtype=int)
        self.data = np.vstack((self.data[:Och+1,:],artsamples, self.data[Och+1:,:]))
        self.labels = np.hstack((self.labels[:Och+1], 'artifacts', self.labels[Och+1:]))
        self.info['no. segments']=0
        #print(str(int((len(np.where(artsamples==1)[0])/self.data.shape[-1])*100))+'% of the data is defined as artifact')

    def segment(self, marking = 'no', trllength = 2, remove_artifact = 'no', overlap =0):
        '''
        Segment the data into epochs, either removing the artifacted epochs at
        the same time or not, based on the input. If removing artifacts the data
        around the artifacts is used in the hope to retain as much data as possible.

        Parameters
        -----------------------------------------------------------------------
        epochlength:         integer, in seconds
        remove_artifacts:   string, 'yes', or 'no' (default = 'no')

        Returns:
        -----------------------------------------------------------------------
        object data, including the fields:

        trl:    n x 2 array; (begin- and end- samples)
        data:   n x nchannels x nsamples matrix; (actual EEG data)
        info:   adds what function has been performed including some details
                about the results
        arttrl: n X 2 array; when artifacts have been removed begin and end samples of all
                compiled artefacts are kept here

        '''
        totallength = self.data.shape[-1]

        if trllength == 'all':
            epochlength = totallength/self.sF
        else:
            epochlength = trllength

        if 'artifacts' in self.labels:
            artidx = np.where(self.labels=='artifacts')[0]

            artsamples = self.data[artidx,:][0]
            ##print('segmenting into trials of: '+str(epochlength)+' seconds')

            p = np.where(artsamples==1)[0]

            if len(p)>0:
                startidxs = np.hstack([np.where(np.diff(artsamples)==1)[0]+1])# diff =1
                endidxs = np.hstack([np.where(np.diff(artsamples)==-1)[0]+1])#diff = -1

                if len(endidxs)==0:
                    endidxs = np.hstack([endidxs,self.data.shape[-1]])
                if len(startidxs)==0:
                    startidxs = np.hstack([startidxs,0])

                if startidxs[-1] > endidxs[-1]:
                    endidxs = np.hstack([endidxs,self.data.shape[-1]])

                if type(endidxs)==int:
                    if endidxs < startidxs:
                        startidxs = np.hstack([0,startidxs])
                elif endidxs[0] < startidxs[0]:
                        startidxs = np.hstack([0,startidxs])

                ARTtrl = np.array([0,0],dtype=int)
                for i in range(len(startidxs)):
                    ARTtrl = np.vstack((ARTtrl,[startidxs[i], endidxs[i]]))
                ARTtrl = ARTtrl[1:]

                if remove_artifact == 'yes' and len(p) > 1:
                    ''' select the segments around the artifacts (as much as possible) '''
                    ''' from the first sample to the beginning of the last artifact '''
                    t = 0
                    trials=np.zeros((1,self.data.shape[0],np.int(self.sF*epochlength)));marktrials = trials.copy();
                    trl = np.array([0,0],dtype=int)
                    for i in range(ARTtrl.shape[0]):
                        if (ARTtrl[i,0]-t)>(np.int(epochlength*self.sF)):
                            tmp = self.data[:,t:ARTtrl[i,0]]
                            segs,segstrl = self._EEGsegmenting(np.asarray(tmp),epochlength, overlap = overlap)
                            trials = np.concatenate([trials,segs],axis=0)
                            trl = np.vstack([trl,segstrl+t])
                            if marking=='yes':
                                tmpmarks = self.marking[:,t:ARTtrl[i,0]]
                                markedsegs = self._EEGsegmenting(np.asarray(tmpmarks),epochlength, overlap = overlap)
                                marktrials = np.concatenate([marktrials,markedsegs],axis=0)
                        t = ARTtrl[i,1]

                    ''' data from last artifact untill end of recording '''
                    if ARTtrl[-1,1] < self.data.shape[-1]-epochlength*self.sF:
                        tmp = self.data[:,t:self.data.shape[-1]]
                        segs, segstrl = self._EEGsegmenting(np.asarray(tmp),epochlength, overlap = overlap)
                        trials = np.concatenate([trials,segs],axis=0)
                        trl = np.vstack([trl,segstrl+t])
                        if marking=='yes':
                            tmpmarks = self.marking[:,t:ARTtrl[i,0]]
                            markedsegs = self._EEGsegmenting(np.asarray(tmpmarks),epochlength, overlap = overlap)
                            marktrials = np.concatenate([marktrials,markedsegs],axis=0)

                    ''' data from the artifacts themselves '''
                    self.artidata=np.zeros((ARTtrl.shape[0],self.data.shape[0],np.nanmax(np.diff(ARTtrl))))
                    for i in range(ARTtrl.shape[0]):
                        self.artidata[i,:,:np.diff(ARTtrl[i,:])[0]] = self.data[:,ARTtrl[i,0]:ARTtrl[i,1]]

                    ''' keep the data in 'trials' '''
                    self.trl = trl[1:]
                    self.data = trials[1:]
                    self.arttrl = ARTtrl
                    self.info['artifact removal'] = 'detected artifacts removed'
                    self.info['no. segments'] = len(trl)-1
                    if self.info['no. segments'] < ((1/3)* (totallength/(epochlength*self.sF))):
                        self.info['data quality'] = 'bad'

                elif remove_artifact == 'no':
                    ##print('no artifact removal')
                    self.data,self.trl = self._EEGsegmenting(self.data, epochlength, overlap = overlap)
                    if marking == 'yes':
                        self.marking = self._EEGsegmenting(self.marking, epochlength, overlap = overlap)[0]
                    self.arttrl=ARTtrl
                    self.info['artifact removal'] = 'none removed'
                    self.info['no. segments'] = len(self.trl)
                    if trllength == 'all':
                        if  len(p) > ((2/3) * totallength):
                            self.info['data quality'] = 'bad'
                        else:
                            self.info['data quality'] = 'OK'

                '''if there are no artefacts '''
            else:
                self.data,self.trl = self._EEGsegmenting(self.data, epochlength, overlap = overlap)
                if marking == 'yes':
                    self.marking = self._EEGsegmenting(self.marking, epochlength, overlap = overlap)[0]
                self.info['artifact removal'] = 'no artifacts detected'
                self.info['no. segments'] = len(self.trl)-1
                self.arttrl = [0]
                if self.info['no. segments'] < ((1.3) * (totallength/(epochlength*self.sF))):
                    self.info['data quality'] = 'bad'

        else:
            self.data,self.trl = self._EEGsegmenting(self.data, epochlength, overlap = overlap)
            if marking=='yes':
                self.marking = self._EEGsegmenting(self.marking, epochlength, overlap = overlap)[0]

            self.info['artifact removal'] = 'no artifacts detected'
            self.info['no. segments'] = len(self.trl)-1
            self.arttrl = [0]
            if trllength == 'all':
                if len(p) > (0.33 * totallength):
                    self.info['data quality'] = 'bad'
                else:
                    self.info['data quality'] = 'OK'
            elif self.info['no. segments'] < (0.33 * (totallength/(epochlength*self.sF))):
                self.info['data quality'] = 'bad'


    def save_pdfs(self, savepath, cond='NaN', inp='data', scaling =[-50, 50]):
        """ This function saves the complet set of raw data as EEG plots for 10 second segments for visual
        inspection for use after preprocessing, including the artifact-channel"""

        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.collections import LineCollection
        #from matplotlib.ticker import MultipleLocator
        from matplotlib.backends.backend_pdf import PdfPages

        plt.ioff()
        idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0]

        trllength = str(self.data.shape[-1]/self.sF)
        if self.info['data quality'] == 'OK':
            outname = 'RawReport_'+cond +'_'+ idcode + '_' + '_' + trllength + 's'
        elif self.info['data quality'] == 'bad':
            outname = 'RawReport_'+cond +'_BAD_'+ idcode + '_' + cond + '_' + trllength + 's'
            #print('saving: data has been marked as BAD')
        if self.info['artifact removal'] == 'none removed':
            outname = 'RawReport_'+ cond +'_' + idcode + '_' + cond + '_' + trllength + 's'
        elif self.info['artifact removal'] == 'no artifact detected':
            outname = 'RawReport_'+ cond + '_'+ idcode + '_' + trllength + 's'

        '''======== make folder per idcode ========'''

        if not os.path.exists(savepath+'/pdf/'):
            os.mkdir(savepath+'/pdf/')
        pdfpath = savepath+ '/pdf/'

        '''======== get the data ========='''
        data = getattr(self, inp)[:,:-1,:]
        data[:,-5,:]=data[:,-5,:]*.5*scaling[1]

        """======== downsample for saving=========="""
        dsmplfrq = 128
        steps=int(self.sF/dsmplfrq)
        dsidx = np.arange(0,data.shape[-1],steps)

        dsdata = data[:,:,dsidx]
        data = dsdata
        self.sF = dsmplfrq

        if inp =='artidata':
            trl = np.array(self.arttrl/steps, dtype=int)
        else:
            trl = np.array(self.trl/steps, dtype=int)

        n_trials, n_rows ,n_samples = data.shape[0], data.shape[1], data.shape[2]

        import datetime
        with PdfPages(pdfpath+outname+'.pdf') as pp:
            #pp = PdfPages(savepath+outname+'test.pdf')
            firstPage = plt.figure(figsize=(11.69,8.27))
            firstPage.clf()
            t =  datetime.datetime.now()
            txt = idcode + ' ' + cond + ' Report created on ' + str(t)[:16] + ' \nBrainclinics Foundation Nijmegen, the Netherlands'
            firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=12, ha="left", va="top")
            pp.savefig()

            for seg in range(n_trials):

                plt.close()
                t = np.arange(0,n_samples/self.sF, (n_samples/self.sF)/n_samples)

                fig = plt.figure(num = seg, figsize = (11.69,8.27), tight_layout=True)
                ax1 = fig.add_subplot(1,1,1)
                plt.subplots_adjust(bottom = 0.2)
                ax1.set_title(idcode + ' ' + cond +'\n Segment: '+ str(seg+1) +' of '+str(n_trials))

                dmin = scaling[0]#data.min()
                dmax = scaling[1]#data.max()
                dr = (dmax - dmin) * 0.7  # Crowd them a bit.
                y0 = dmin
                y1 = (n_rows-1) * dr + dmax

                segments = []
                yticklocs = []
                #ticks = np.arange(0,np.int(n_samples/self.sF),np.around((np.int((n_samples/self.sF))/10),decimals=1))
                for i in range(n_rows):
                    segments.append(np.column_stack((t, data[seg,i,:])))
                    yticklocs.append(i * dr)

                xticks = np.arange(0,(data.shape[-1]/self.sF)+((data.shape[-1]/self.sF)/10),(data.shape[-1]/self.sF)/10)
                ax1.set_xticks(xticks,minor=False)
                ax1.vlines(np.arange(1,(self.data.shape[-1]/self.sF),1),ymin=y0,ymax=y1, color='dodgerblue',alpha = 0.3)

                xticksl = np.arange(np.around(trl[seg,0]/self.sF,decimals=2),np.around((trl[seg,0]/self.sF)+(n_samples/self.sF),decimals=2)+1,np.around((n_samples/self.sF)/10,decimals=2))
                xticklabels = list(xticksl)#np.arange(ticks)

                xlabels = [ '%.1f' % elem for elem in xticklabels]
                xlabels = np.array(xlabels,dtype=str)
                ax1.set_xlim(xticks[0],xticks[-1])
                ax1.set_xticklabels(xlabels)

                offsets = np.zeros((n_rows, 2), dtype=float)
                offsets[:,1] = yticklocs

                ax1.set_ylim(y0,y1)
                ax1.set_yticks(yticklocs)
                ax1.set_yticklabels(self.labels[:data.shape[1]][::-1])

                lines = LineCollection(np.flipud(segments), offsets=offsets,linewidths=.8, colors = 'k')
                ax1.add_collection(lines)

                ax1.set_xlabel('Time (s)')

                pp.savefig(dpi=1200,transparent=False)
                plt.close()
        #reset the sF for further processing
        self.sF = 2048

    def save(self, savepath, matfile='no', csv = 'no', npy = 'yes'):
        '''
        This function is used to save the EEG data (only data in csv, the dataset
        class object is pickled to .npy, in matlab format it is saved in a dictonary format,
        which can be opened as a structure array in Matlab).

        Parameters
        -----------------------------------------------------------------------
        preprocpath:    the path to which the from the preprocessing resulting data
                        should be saved
        matfile:        should it be saved in matlab format? 'yes' / 'no'
        csvfile:        should it be saved in csv format? 'yes' / 'no'
        npyfile:        should it be saved in npy format? 'yes' / 'no'
        '''
        import pandas as pd
        import scipy.io as sio

        #print('saving data \n')
        '''======== collect information about the data ========'''

        idcode = self.info['fileID'].rsplit('/')[-1].split('.')[0]
        cond = self.info['fileID'].rsplit('/')[-1].split('.')[1]

        trllength = str(self.data.shape[-1]/self.sF)
        if self.info['data quality'] == 'OK':
            outname = idcode + '_' + cond + '_' + trllength + 's'
        else:
            outname = 'BAD_'+ idcode + '_' + cond + '_' + trllength + 's'
            #print('saving: data has been marked as BAD')

        if csv == 'yes':
            '''======== save (only) the data in csv format ========'''
            if os.path.isdir(savepath + '/csv_data_' + cond + '_' + trllength + 's/'):
                csvpath = savepath + '/csv_data_' + cond + '_' + trllength + 's/'
            else:
                os.mkdir(savepath + '/csv_data_' + cond + '_' + trllength + 's/')
                csvpath = savepath + '/csv_data_' + cond + '_' + trllength + 's/'

            for i in range(self.data.shape[0]):
                if len(self.data.shape) == 3:
                    df = pd.DataFrame(self.data[i,:,:].T)
                    df.to_csv(csvpath + str((self.trl[i,0]/self.sF)*1000) + '.csv',sep=',',header = list(self.labels),compression = None)
                else:
                    df = pd.DataFrame(self.data[:,:].T)
                    df.to_csv(csvpath + str(0)+'.csv',sep=',',header = list(self.labels),compression = None)

            #'''======== save info in txt format (per condition) ========'''
            #df = pd.DataFrame(self.info)
            #df.T.to_csv(csvpath + outname + '_info.txt',header=None, sep=' ', mode='a')

        if npy == 'yes':
            '''======== save the data for deep learning in Pickle ========='''
            import pickle
            npypath=os.path.join(savepath,outname +'.npy')
            with open(npypath, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(vars(self), output, -1)

        '''======== optionally save as matlab structure array ========='''
        if matfile == 'yes':

            mat_dataset = {'labels': self.labels,
                           'trials': self.data,
                           'dimord' :'rpt_chan_time',
                           'artifacts': self.arttrl,
                           'Fs':self.Fs,
                           'time': np.arange(0,(self.data.shape[-1]/self.sF),1/self.sF),
                           'info': self.info}
            sio.savemat(savepath + '/' + outname +'.mat', mat_dataset)

    def plot_EEG(self, inp='data' , scaling=[-70,70], title=None):

        '''
        This function is used to plot the EEG data.

        Parameters
        -----------------------------------------------------------------------
        inp:        'data' or 'artidata', choose to display the data or the artifacts
                    if available (only after segmenting the data)
        scaling:    array of 1x2, minimum and maximum value on y-axis to dispay,
                    in microvolts

        Returns
        -----------------------------------------------------------------------
        plots the EEG using matplotlib, using the arrows to browse through the
        segments (if there are multiple segments)

        '''

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from matplotlib.collections import LineCollection

        class GUIButtons(object):

            def __init__(self, tmpdata, sF,axs, t, trls):
                self.tmpdata = tmpdata
                self.axs = axs
                self.t = t
                self.trls = trls
                self.sF = sF

            trl = 0
            end = 0

            def nextb(self, event):
                ''' move to the next segment '''
                self.trl += 1
                i = self.trl

                ''' determine the shape of the data '''
                n_trials = self.tmpdata.shape[0]
                n_rows = self.tmpdata.shape[1]
                ''' make a 'list' of the data in the new segment '''

                if i >= n_trials:
                    i = n_trials
                    self.axs['ax1'].set_title('Last sample reached. Cannot go forwards')
                else:
                    segments=[];
                    for r in range(n_rows):
                        segments.append(np.column_stack((self.t, self.tmpdata[i,r,:])))

                    ''' fill the current plot's LineCollection (called lines) with the new segment's data '''
                     # get the current plot's axes

                    linesn = self.axs['ax1'].collections[0]
                    linesn.set_segments(np.flipud(np.array(segments)))
                    self.axs['ax1'].set_title('Segment: '+str(i+1) + ' of ' + str(n_trials))
                    #self.axs['ax1'].set_xticks(ticks,minor=False)
                    ticksl = np.arange(np.around(self.trls[i,0]/self.sF,decimals=2),np.around((self.trls[i,0]/self.sF)+(self.tmpdata.shape[-1]/self.sF),decimals=2)+((data.shape[-1]/self.sF)/10),np.around((self.tmpdata.shape[-1]/self.sF)/10,decimals=2))

                    ticklabels = list(ticksl)#np.arange(ticks)
                    xlabels = [ '%.1f' % elem for elem in ticklabels]
                    xlabels = np.array(xlabels,dtype=str)

                    self.axs['ax1'].set_xticklabels(xlabels)
                    self.axs['ax1'].vlines(np.arange(0,self.tmpdata.shape[-1]/self.sF,1),ymin=np.min(self.axs['ax1'].get_ylim()),ymax=np.max(self.axs['ax1'].get_ylim()),color='dodgerblue',alpha=0.3)
                    plt.show()


            ''' button for previous segment '''
            def prevb(self, event):
                ''' move to the next segment '''
                self.trl -= 1
                i = self.trl

                ''' determine the shape of the data '''
                n_trials = self.tmpdata.shape[0]
                n_rows = self.tmpdata.shape[1]

                ''' make a 'list' of the data in the new segment '''
                if i < 0:
                    i = 0
                    self.axs['ax1'].set_title('First sample reached. Cannot go backwards')
                else:
                    segments=[];
                    for r in range(n_rows):
                        segments.append(np.column_stack((self.t, self.tmpdata[i,r,:])))

                    ''' fill the current plot's LineCollection (called lines) with the new segment's data '''
                     # get the current plot's axes
                    linesn = self.axs['ax1'].collections[0]
                    linesn.set_segments(np.flipud(np.array(segments)))
                    self.axs['ax1'].set_title('Segment: '+str(i+1) + ' of ' + str(n_trials))
                    ticksl = np.arange(np.around(self.trls[i,0]/self.sF,decimals=2),np.around((self.trls[i,0]/self.sF)+(self.tmpdata.shape[-1]/self.sF),decimals=2)+((data.shape[-1]/self.sF)/10),np.around((self.tmpdata.shape[-1]/self.sF)/10,decimals=2))

                    ticklabels = list(ticksl)#np.arange(ticks)
                    xlabels = [ '%.2f' % elem for elem in ticklabels]
                    xlabels = np.array(xlabels,dtype=str)
                    self.axs['ax1'].set_xticklabels(xlabels)
#                    self.axs['ax1'].vlines(np.arange(0,self.tmpdata.shape[-1]/self.sF,1),ymin=np.min(self.axs['ax1'].get_ylim()),ymax=np.max(self.axs['ax1'].get_ylim()),color='dodgerblue',alpha=0.3)

                    plt.show()

        data = getattr(self, inp)

        if inp =='artidata':
            trl = self.arttrl
        else:
            trl = self.trl

        if len(data.shape) == 3:
            n_samples, n_rows, n_trials = data.shape[2], data.shape[1], data.shape[0]
            # if 'artifacts' in self.labels:
            #     artifacts = np.where(self.labels == 'artifacts')[0]
            #     data[:,artifacts,:]=data[:,artifacts,:]*50 #upscale artifacts


        elif len(data.shape) == 2:
            n_samples, n_rows = data.shape[1], data.shape[0]
            # if 'artifacts' in self.labels:
            #     artifacts = np.where(self.labels == 'artifacts')[0]
            #     data[artifacts,:]=data[artifacts,:]*50 #upscale artifacts

            n_trials = 1
            trl = np.array([0,0],dtype=int)

        t = np.arange(0,n_samples/self.sF, (n_samples/self.sF)/n_samples)

        if title == None:
            fig = plt.figure(self.info['fileID'].rsplit('/')[-1], figsize = (6,9))
        else:
            fig = plt.figure(title, figsize = (6,9))

        ax1 = fig.add_subplot(1,1,1)
        plt.subplots_adjust(bottom = 0.2)
        ax1.set_title('Segment: '+ str(1) +' of '+str(n_trials))

        dmin = scaling[0]#data.min()
        dmax = scaling[1]#data.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (n_rows-1) * dr + dmax
        ax1.set_ylim(y0, y1)

        segments = []
        ticklocs = []
        for i in range(n_rows):
            if len(data.shape) == 3:
                segments.append(np.column_stack((t, data[0,i,:])))
            elif len(data.shape) == 2:
                segments.append(np.column_stack((t, data[i,:])))

            ticklocs.append(i * dr)

        ticks = np.arange(0,(data.shape[-1]/self.sF)+((data.shape[-1]/self.sF)/10),(data.shape[-1]/self.sF)/10)
        ax1.set_xticks(ticks,minor=False)
        ax1.vlines(np.arange(1,(self.data.shape[-1]/self.sF),1),ymin=y0,ymax=y1, color='dodgerblue',alpha = 0.3)

        ticksl = np.arange(np.around(trl.flat[0]/self.sF,decimals=2),np.around((trl.flat[0]/self.sF)+(n_samples/self.sF),decimals=2)+1,np.around((n_samples/self.sF)/10,decimals=2))

        ticklabels = list(ticksl)#np.arange(ticks)
        xlabels = [ '%.1f' % elem for elem in ticklabels]
        xlabels = np.array(xlabels,dtype=str)
        ax1.set_xticklabels(xlabels)

        offsets = np.zeros((n_rows, 2), dtype=float)
        offsets[:,1] = ticklocs

        lines = LineCollection(np.flipud(segments), linewidths=(0.8), offsets=offsets, transOffset=None, colors = 'k')
        ax1.add_collection(lines)

        ax1.set_yticks(ticklocs)

        ax1.set_yticklabels(self.labels[::-1])

        ax1.set_xlabel('Time (s)')
        ax1.set_xlim(0,self.data.shape[-1]/self.sF)


        '''locations on figure '''
        axs = {}
        axs['ax1'] = ax1
        axs['axnext'] = plt.axes([0.84, 0.10, 0.10, 0.04])#next button
        axs['axprev'] = plt.axes([0.72, 0.10, 0.10, 0.04])#previous button

        callback = GUIButtons(data,self.sF,axs,t,trl)

        ''' buttons '''
        bnext = Button(axs['axnext'], '>')
        bnext.on_clicked(callback.nextb)
        axs['axnext']._button = bnext

        bprev = Button(axs['axprev'], '<')
        bprev.on_clicked(callback.prevb)
        axs['axprev']._button = bprev

        plt.show()
        plt.axis('tight')

        return bnext, bprev


    '''========================================================================='''
    '''===========================   SUBFUNCTIONS   ============================'''
    '''========================================================================='''

    def _detect_artifact(self,inp,threshold):
        ''' detect if and when there is a artifact (with zscores) for inter-individual...
        comparability/usability'''
        from scipy.stats import zscore
        '''compute zscore for thresholding eyemovements'''
        zdata = zscore(inp)
        ##print(inp)

        Asamps = [np.where((zdata > threshold) | (zdata < -1*threshold))][0][0]
        ''' initiate eyemovements vector '''
        Atrl = np.array([0,0],dtype=int)
        ''' define the segments that contain vertical eyemovements '''
        if len(Asamps) > 0:
            begin = Asamps[0] #first sample of Vsamps is start of first eyeblink
            for e in range(len(Asamps)):
                if e >= len(Asamps)-1:
                    end = Asamps[-1]
                    Atrl = np.vstack((Atrl,[begin,end]))
                elif Asamps[e+1] == Asamps[e]+1:
                    continue
                else:
                    end = Asamps[e]
                    Atrl = np.vstack((Atrl,[begin,end]))
                    begin = Asamps[e+1]
            Atrl = Atrl[1:] #remove the first row (containing only zeros)
            
        return Atrl, Asamps

    def _EEGsegmenting(self,inp, trllength, overlap=0):

        epochlength = np.int(trllength*self.sF)
        stepsize = (1-overlap)*epochlength

        ''' define the size of the data '''
        n_totalsamples, n_samples, n_rows = inp.shape[1],epochlength, inp.shape[0]
        n_trials = np.int(n_totalsamples/stepsize)

        trl = np.array([0,0],dtype=int)
        t = 0
        for i in range(n_trials):
            trl = np.vstack((trl,[t,t+n_samples]))
            t += stepsize

        trl = np.array(trl[1:], dtype=int)

        data = np.zeros((n_trials, n_rows, np.int(n_samples)))
        for i in range(n_trials):
            if trl[i,0] <= n_totalsamples-n_samples:
                data[i,:,:]= inp[:,trl[i,0]:trl[i,1]]

        return data, trl

    def _interpolate_data(self, inp, intchan):
        from scipy.spatial import distance
        ''' define channel locations '''
        chlocs =[]
        eegchans = np.arange(self.n_eegchans)

        for chan in self.labels[eegchans]:
            if chan =='Cz':
                chan = 'C7' #C7 is plugged in Cz location for recoding, and Cz is computed in rereferencing procedure
            ch = np.where(np.array(self.info['ch_names'])==chan)[0][0]
            chlocs.append(self.info['chs'][ch]['loc'][:3])

        chlocs = np.array(chlocs)

        dist = np.zeros((len(chlocs),len(chlocs)))
        for s1 in range(len(chlocs)):
            for s2 in range(len(chlocs)):
                dist[s1,s2]=distance.euclidean(chlocs[s1,:],chlocs[s2,:])
        repaired=[]
        for sens in intchan:
            idx =  np.where(np.logical_and(dist[sens,:]>0, dist[sens,:]<0.1))[0]

            neighbidx = idx[np.argsort(dist[sens,idx])[:3]]
            interpneighbs = neighbidx[np.where(np.in1d(neighbidx,intchan, invert=True))[0]]

        #    #print(sens + '\t'+str(len(neighidx)))

            if len(interpneighbs) >= 2:
                wghtneighbs = np.zeros((len(interpneighbs),self.data.shape[1]))

                sumweights = np.sum(dist[sens,interpneighbs])

                W = (sumweights-dist[sens,interpneighbs])#/np.nanmax(sumweights-dist[sens,neighbidx])
                wghts = W/np.sum(W)

                for nb in range(len(interpneighbs)):
                    wghtneighbs[nb,:] = (inp[interpneighbs[nb],:]*wghts[nb])#/sumweights

                inp[sens,:] = np.sum(wghtneighbs,axis=0)
                feedback = 'repaired '+str(len(intchan))+' bad, empty and/or bridging channels'
                repaired = np.append(repaired,'yes')
            else:
                #print('to many bad neighbours, not possible to repair channel: '+ self.labels[sens] +' ('+str(sens)+')')
                feedback = 'not repaired the '+str(len(intchan))+ ' channels, there were to many bad neighbouring channels'
                repaired = np.append(repaired,'no')

            if 'no' in repaired:
                repair = 'no'
            else:
                repair ='yes'

        return inp, feedback, repair, intchan

    def _bridging_check(self,inp):
        """ Based on
        Tenke, C. E. & Kayser, J. A convenient method for detecting electrolyte bridges in multichannel
        electroencephalogram and event-related potential recordings. Clin Neurophysiol 112, 545–550 (2001).
        This function detects when two channels are bridged by gel = essentially measuring the same signal, identified
        by low-amplitude difference waveforms (electrical distance)"""
        ED = np.zeros((self.n_eegchans,self.n_eegchans))

        for r1 in range(self.n_eegchans):
            for r2 in range(self.n_eegchans):
                ED[r1,r2] = np.squeeze(np.nanmean(np.square((inp[r1,:]-inp[r2,:])-(np.nanmean(inp[r1,:]-inp[r2,:])))))

        tmpidx = np.where(ED == 0)
        bridgeidx = np.where(np.not_equal(tmpidx[0],tmpidx[1]) == True)[0]
        tmpidx = np.asarray(tmpidx).T
        bridgepairs = []
        for x,y in tmpidx[bridgeidx,:]:
            if (x, y) not in bridgepairs and (y, x ) not in bridgepairs:
                bridgepairs.append((x, y))

        bridgechanidx = np.unique(bridgepairs)

        return bridgechanidx, bridgepairs

    def _artifact_samps_trl(self,ARTsamps, artpadding, Fs, totalsamps):
        """ subfunction for the define_artifacts function, identifying in which
        sample there is an identified artifact"""
        def find_artifacts(inpdata, totalsamps, artpadding = 0):

            tmpARTsamps=np.zeros((inpdata.shape))
            p = np.where(inpdata==1)[0]
            if p[0] == 0:
                upidxs = np.append(0,np.where(np.diff(inpdata)==1)[0])# diff =1
            else:
                upidxs = np.where(np.diff(inpdata)==1)[0]
            if p[-1] == totalsamps:
                downidxs = np.append(np.where(np.diff(inpdata)==-1)[0],totalsamps)# diff =1
            else:
                downidxs = np.where(np.diff(inpdata)==-1)[0]

            if len(upidxs)>len(downidxs):
                downidxs = np.append(np.where(np.diff(inpdata)==-1)[0],totalsamps)

            startidxs = upidxs-np.int(artpadding*Fs)
            endidxs = downidxs+np.int(artpadding*Fs)

            tmpARTtrl = np.array([0,0],dtype=int)
            for k in range(len(startidxs)):
                if startidxs[k] <= 0:
                    startidxs[k]=0
                if endidxs[k] >= totalsamps:
                    endidxs[k] = totalsamps

                tmpARTsamps[startidxs[k]:endidxs[k]]=1
                tmpARTtrl = np.vstack((tmpARTtrl,[startidxs[k],endidxs[k]]))
            tmpARTtrl = tmpARTtrl[1:]

            return tmpARTsamps, tmpARTtrl
    ####----------------------------------
        paddedARTsamps=np.zeros((ARTsamps.shape))

        for r in range(self.n_eegchans):
            cart = ARTsamps[r,:]
            p = np.where(cart==1)[0]
            if len(p) > 1:
                paddedARTsamps[r,:] = find_artifacts(cart, totalsamps, artpadding = artpadding)[0]
            else:
                paddedARTsamps[r,:]= np.zeros((cart.shape))

        maxARTsamps = np.nanmax(paddedARTsamps,axis=0)
        p = np.where(maxARTsamps==1)[0]
        if len(p) > 1:
            ARTtrl = find_artifacts(maxARTsamps, totalsamps, artpadding=0)[1]
        else:
            ARTtrl = []

        return ARTtrl, paddedARTsamps
