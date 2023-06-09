import pandas as pd
import scipy.io.wavfile as wavfile
import pickle
import re
import librosa
import numpy as np
import os
import repet
from sklearn.preprocessing import LabelEncoder, StandardScaler
from feature_extraction import  mel_freq_cepstrum, dataset_merge #chunks, extract_peaks_and_freqs, final_data_collection, data_collection_only_peaks
from audio_from_link import delete_spaces, download_audio, remove_audio, from_mp4_to_wav #obtain_youtube_link

instruments_reduced = {'woodwind': ['Clarinet in Bb','Flute','Oboe', 'Bassoon', 'Alto Saxophone', 'Wind Instrument'],
                       'brass':['Bass Tuba','French Horn','Trombone', 'Trumpet in C'],
                       'bowed_string_instruments': ['Cello','Viola','Violin', 'Bowed string instrument'],
                       'plucked_string_instruments':['Guitar','Electric guitar','Acoustic guitar', 'Tapping', 'Bass guitar','Contrabass'],
                       'keyboard': ['Keyboard','Accordion'],
                       'percussion': ['Steelpan','Percussion', 'Drum and bass'],
                       'singing': ['Single Voice Singing', 'Group Singing'],
                       'noise': ['Noise']}

instrument_type = list(instruments_reduced.keys())

common_path = '/home/jacs/Documents/DataScience/Personal/'
dummy_path = 'song_similarity/'

inst_0 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/woodwind_model.pkl"
with open(inst_0, 'rb') as file:
    inst_0_model = pickle.load(file)
inst_1 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/brass_model.pkl"
with open(inst_1, 'rb') as file:
    inst_1_model = pickle.load(file)
inst_2 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/bowed_string_model.pkl"
with open(inst_2, 'rb') as file:
    inst_2_model = pickle.load(file)
inst_3 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/plucked_string_model.pkl"
with open(inst_3, 'rb') as file:
    inst_3_model = pickle.load(file)
inst_4 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/keyboard_model.pkl"
with open(inst_4, 'rb') as file:
    inst_4_model = pickle.load(file)
inst_5 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/percussion_model.pkl"
with open(inst_5, 'rb') as file:
    inst_5_model = pickle.load(file)
inst_6 = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/singing_model.pkl"
with open(inst_6, 'rb') as file:
    inst_6_model = pickle.load(file)


def min_to_sec(number):
    number = str(number).split(':')
    number = int(number[0])*60 + int(number[1])
    return number

def audio_partition(audio, Fs, range_1, range_2):
    if range_1=='begin':
        range_1='0:00'
    if  range_2=='end':
        range_2_min = str(int((audio.shape[0]/Fs)/60))
        range_2_seg = str(int((audio.shape[0]/Fs)%60))
        range_2 = range_2_min+':'+range_2_seg
    length = audio.shape[0]/Fs
    range_1_index = int(min_to_sec(range_1)*Fs)
    range_2_index = int(min_to_sec(range_2)*Fs)
    audio_cut = audio[range_1_index:range_2_index]
    return audio_cut


def song_feature_extraction_cepstrum(kk):
    database_name = str(titles[kk])
    indexoo = 0
    range_1 = str(df_links['from'].iloc[kk])
    range_2 = str(df_links['to'].iloc[kk])
    print(range_1,range_2)
    linko = links_audio[kk]
    title_file = str(download_audio(linko))
    df_final = pd.DataFrame()
    try:
        audio, Fs = librosa.load(title_file+'.wav')
    except:
        from_mp4_to_wav(title_file+'.mp4',common_path+dummy_path)
        audio, Fs = librosa.load(title_file+'.wav')
        remove_audio(title_file+'.mp4')
    # Estimate the background signal, and the foreground signal
    print(audio.shape)
    background_signal = repet.original(audio, Fs)
    audio_signal = np.reshape(audio, audio.shape + (1,))
    foreground_signal = audio_signal - background_signal
    # Write the background and foreground signals
    repet.wavwrite(background_signal, Fs, "background_signal.wav")
    repet.wavwrite(foreground_signal, Fs, "foreground_signal.wav")
    audio = audio_partition(foreground_signal, Fs, range_1, range_2)
    length = audio.shape[0] / Fs
    audio = np.squeeze(audio, axis=1)
    print(audio.shape)
    df_final_2 = mel_freq_cepstrum(audio, Fs, 13, kk, title_file)
    df_final = pd.concat((df_final,df_final_2), axis=0).reset_index(drop=True)
    df_final.to_csv(database_name+'.csv', index=False)
    remove_audio(title_file+'.wav')
    remove_audio('background_signal'+'.wav')
    remove_audio('foreground_signal'+'.wav')
    return df_final

def instrument_identification(X):
    Y_pred = []
    Y_name = []
    for i in range(0,len(X)):
        XX = X.iloc[i,:]
        if XX['instrument_type_predicted'] == 0:
            inst_model = inst_0_model
            kk = 0
        if XX['instrument_type_predicted'] == 1:
            inst_model = inst_1_model
            kk = 1
        if XX['instrument_type_predicted'] == 2:
            inst_model = inst_2_model
            kk = 2
        if XX['instrument_type_predicted'] == 3:
            inst_model = inst_3_model
            kk = 3
        if XX['instrument_type_predicted'] == 4:
            inst_model = inst_4_model
            kk = 4
        if XX['instrument_type_predicted'] == 5:
            inst_model = inst_5_model
            kk = 5
        if XX['instrument_type_predicted'] == 6:
            inst_model = inst_6_model
            kk = 6
        if XX['instrument_type_predicted'] == 7:
            inst_model = 'plopo'
            kk = 7
        XX = np.array(X.iloc[i,:-3], dtype = float).reshape(1, -1)
#        XX = XX[:-3]
#        print(XX)
#        XX = np.array(list(XX.values), dtype=float).reshape(1,-1)
        try:
            y_predo = inst_model.predict(XX)[0]
            Y_pred.append(y_predo)
            Y_name.append(instruments_reduced[instrument_type[kk]][y_predo])
        except:
            kk = 7
            Y_pred.append(7)
            Y_name.append(instruments_reduced[instrument_type[kk]][0])
    return Y_pred, Y_name


if __name__ == '__main__':
    df_links = pd.read_csv('audio_model.csv')
    print(df_links)
    links_audio = list(df_links['youtube_links'])
    titles = list(df_links['title'])
    for kk in range(0,len(links_audio)):
        x = 'Algo shady'
        while x == 'Algo shady':
            try:
                df_final = song_feature_extraction_cepstrum(kk)
#                df_final = pd.read_csv('{}.csv'.format(titles[kk]))
            except:
                continue
            df_final = df_final[['rms', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
                                 'mfccs_0', 'mfccs_1', 'mfccs_2', 'mfccs_3', 'mfccs_4', 'mfccs_5',
                                 'mfccs_6', 'mfccs_7', 'mfccs_8', 'mfccs_9', 'mfccs_10', 'mfccs_11',
                                 'mfccs_12']]
            scaler = StandardScaler()
#            X = np.array(df_final.iloc[:, :-1], dtype = float)
            X = scaler.fit_transform(np.array(df_final.iloc[:, :-1], dtype = float))
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            inst_type = "/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/models/instrument_type_model.pkl"
            with open(inst_type, 'rb') as file:
                inst_type_model = pickle.load(file)
            Ypredict = inst_type_model.predict(X)
            df_final['instrument_type_predicted'] = Ypredict
            df_final['instrument_type_name'] = [instrument_type[i] for i in Ypredict]
            df_final['instrument_identification'], df_final['instrument_identification_name']= instrument_identification(df_final)
            df_final.to_csv('blop.csv', index=False)
            print(df_final.groupby(['instrument_type_name'])['instrument_type_name'].count())
            print(df_final.groupby(['instrument_identification_name'])['instrument_type_name'].count())
            x = 'Otra cosa'
        #    print(Ypredict)
