# for data transformation
import numpy as np
import os
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy
# for dataframe manipulation
import pandas as pd
#######
import librosa


#######################################################################
# Mel-frequency cepstrum: encapsulates the timbre of a sound performed by either a voice or an instrument
# this function gets the first 10 coefficients of each window of analysis
# wavo: .wav audio file route
# mm number of features to extract


def mel_freq_cepstrum(xx, fs, mm,  m, note_played):
    mfccs = librosa.feature.mfcc(y=xx, sr=fs)
#    df_iteration = pd.DataFrame({'index': [], 'mfccs_envelope':[], 'rms':[], 'spec_cent':[], 'spec_bw':[], 'rolloff':[], 'zcr':[],
#                                 'instrument': [], 'note_played': []})
    rms = librosa.feature.rms(y=xx)
    spec_cent = librosa.feature.spectral_centroid(y=xx, sr=fs)
    spec_bw = librosa.feature.spectral_bandwidth(y=xx, sr=fs)
    rolloff = librosa.feature.spectral_rolloff(y=xx, sr=fs)
    zcr = librosa.feature.zero_crossing_rate(xx)
    df_iteration = pd.DataFrame()
    df_iteration['index'] = range(0,rms.shape[1])
    df_iteration['rms']= rms[0,:]
    df_iteration['spec_cent']= spec_cent[0,:]
    df_iteration['spec_bw']= spec_bw[0,:]
    df_iteration['rolloff']= rolloff[0,:]
    df_iteration['zcr']= zcr[0,:]
    for i in range(0,mm):
        df_iteration['mfccs_{}'.format(i)] = mfccs[i,:]
    df_iteration['instrument']= [m]*rms.shape[1]
    df_iteration['note_played']= [note_played]*rms.shape[1]
    return df_iteration


def dataset_merge(input_data_path, instrument_folder, final_data_path):
    if not os.path.exists(final_data_path):
        os.makedirs(final_data_path)
    df_final = pd.DataFrame([])
#    print(os.listdir(path))
    for file in os.listdir(input_data_path):
        if file == '{}_youtube_database_enrichment.csv'.format(instrument_folder):
            continue
        else:
            df_iteration = pd.read_csv(input_data_path+'/'+file)
            df_final = pd.concat((df_final,df_iteration), axis=0).reset_index(drop=True)
    df_final['instrument_name'] = [instrument_folder]*len(df_final)
    df_final.to_csv(final_data_path+'{}.csv'.format(instrument_folder))
#    final_data_path = common_path + dummy_path + instrument_folder

    
