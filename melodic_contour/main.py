import pandas as pd
import pickle
import numpy as np
import librosa
import json
import sys

from extract_melodic_contour import main_frequencies_songs

# adding Folder_2 to the system path
sys.path.insert(0, '/home/jacs/Documents/DataScience/Personal/song_similarity/')
import repet
from audio_from_link import delete_spaces, download_audio, remove_audio, from_mp4_to_wav #obtain_youtube_link

models_path = '/home/jacs/Documents/DataScience/Personal/song_similarity/melodic_contour/models/'

exists_path = '/home/jacs/Documents/DataScience/Personal/song_similarity/melodic_contour/'

octaves_names = ['4th_octave' , '5th_octave', '6th_otave']

notes_names = ['A#', 'A', 'B', 'C#', 'C', 'D#', 'D', 'E', 'F#', 'F', 'G#', 'G']

def exists(route):
    df_song = pd.read_csv(route)
    df_song = df_song[['song_played', '(200, 219)', '(220, 239)',
                       '(240, 259)', '(260, 279)', '(280, 299)', '(300, 319)', '(320, 339)',
                       '(340, 359)', '(360, 379)', '(380, 399)', '(400, 419)', '(420, 439)',
                       '(440, 459)', '(460, 479)', '(480, 499)', '(500, 519)', '(520, 539)',
                       '(540, 559)', '(560, 579)', '(580, 599)', '(600, 619)', '(620, 639)',
                       '(640, 659)', '(660, 679)', '(680, 699)', '(700, 719)', '(720, 739)',
                       '(740, 759)', '(760, 779)', '(780, 799)', '(800, 819)', '(820, 839)',
                       '(840, 859)', '(860, 879)', '(880, 899)', '(900, 919)', '(920, 939)',
                       '(940, 959)', '(960, 979)', '(980, 999)', '(1000, 1019)',
                       '(1020, 1039)', '(1040, 1059)', '(1060, 1079)', '(1080, 1099)',
                       '(1100, 1119)', '(1120, 1139)', '(1140, 1159)', '(1160, 1179)',
                       '(1180, 1199)', '(1200, 1219)', '(1220, 1239)', '(1240, 1259)',
                       '(1260, 1279)', '(1280, 1299)', '(1300, 1319)', '(1320, 1339)',
                       '(1340, 1359)', '(1360, 1379)', '(1380, 1399)', '(1400, 1419)',
                       '(1420, 1439)', '(1440, 1459)', '(1460, 1479)', '(1480, 1499)',
                       '(1500, 1519)', '(1520, 1539)', '(1540, 1559)', '(1560, 1579)',
                       '(1580, 1599)', '(1600, 1619)', '(1620, 1639)', '(1640, 1659)',
                       '(1660, 1679)', '(1680, 1699)', '(1700, 1719)', '(1720, 1739)',
                       '(1740, 1759)', '(1760, 1779)', '(1780, 1799)', '(1800, 1819)',
                       '(1820, 1839)', '(1840, 1859)', '(1860, 1879)', '(1880, 1899)',
                       '(1900, 1919)', '(1920, 1939)', '(1940, 1959)', '(1960, 1979)',
                       '(1980, 1999)', 'rms', 'spec_cent', 'rolloff', 'zcr']]
    return df_song

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

def background_foreground(title, input_folder, output_folder):
    audio, Fs = librosa.load(input_folder + title+ '.wav')
    background_signal = repet.original(audio, Fs)
    audio_signal = np.reshape(audio, audio.shape + (1,))
    foreground_signal = audio_signal - background_signal
    df_final = pd.DataFrame()
    # Write the background and foreground signals
    repet.wavwrite(background_signal, Fs, "background_signal.wav")
    repet.wavwrite(foreground_signal, Fs, "foreground_signal.wav")
    audio = audio_partition(foreground_signal, Fs, range_1, range_2)
    length = audio.shape[0] / Fs
    audio = np.squeeze(audio, axis=1)
    print(audio.shape)
    df_final_2 = main_frequencies_songs(title, input_folder, output_folder, 1)
#    df_final = pd.concat((df_final,df_final_2), axis=0).reset_index(drop=True)
#    df_final.to_csv(title+'.csv', index=False)
#    remove_audio(title_file+'.wav')
    remove_audio('background_signal'+'.wav')
    remove_audio('foreground_signal'+'.wav')
    return df_final_2


mel_octave4 = models_path + "melodic_contour_octave4.pkl"
with open(mel_octave4, 'rb') as file:
    mel_octave4_model = pickle.load(file)
mel_octave5 = models_path + "melodic_contour_octave5.pkl"
with open(mel_octave5, 'rb') as file:
    mel_octave5_model = pickle.load(file)
mel_octave6 = models_path + "melodic_contour_octave6.pkl"
with open(mel_octave6, 'rb') as file:
    mel_octave6_model = pickle.load(file)
    
def note_identification(X):
    Y_pred = []
    Y_name = []
    for i in range(0,len(X)):
        XX = X.iloc[i,:]
        if XX['prediction'] == 0:
            inst_model = mel_octave4_model
            kk = 0
        if XX['prediction'] == 1:
            inst_model = mel_octave5_model
            kk = 1
        if XX['prediction'] == 2:
            inst_model = mel_octave6_model
            kk = 2
        XX = np.array(X.iloc[i,1:-2], dtype = float).reshape(1, -1)
#        XX = XX[:-3]
#        print(XX)
#        XX = np.array(list(XX.values), dtype=float).reshape(1,-1)
        y_predo = inst_model.predict(XX)[0]
        Y_pred.append(y_predo)
        Y_name.append(notes_names[y_predo])
    return Y_pred, Y_name

existir = 1

if __name__ == '__main__':
    input_folder = '/home/jacs/Documents/DataScience/Personal/song_similarity/'
    output_folder = '/home/jacs/Documents/DataScience/Personal/song_similarity/melodic_contour/'
    df_links = pd.read_csv(input_folder +'audio_model.csv')
    print(df_links)
    links_audio = list(df_links['youtube_links'])
    titles = list(df_links['title'])
    for kk in range(0,len(links_audio)):
        range_1 = str(df_links['from'].iloc[kk])
        range_2 = str(df_links['to'].iloc[kk])
        linko = links_audio[kk]
        df_final = pd.DataFrame()
        title = titles[kk]
        x = 'Algo shady'
        while x == 'Algo shady':
            try:
                if existir == 0:
                    title_file = str(download_audio(linko))
                    ######### RUN WITH DOWNLOAD
                    ## separate background and frontground
                    df_song = background_foreground(title_file, input_folder, output_folder)
                    ## No filters
#                    df_song = main_frequencies_songs(title_file, input_folder, output_folder, 0)
                    ##########
                else:
                    title_file = title
                    #### RE-RUN WITHOUT DOWNLOAD
                    existe = exists_path +title+'.csv'
                    df_song = exists(existe)
            except:
                print('elol')
                continue
            ########
#            X = scaler.fit_transform(np.array(df_song.iloc[:, :], dtype = float))
            X = np.array(df_song.iloc[:, 1:], dtype = float)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            mel_octave_recognition = models_path + "melodic_contour_scale_recognition.pkl"
            with open(mel_octave_recognition, 'rb') as file:
                mel_octave_recognition_model = pickle.load(file)
            Ypredict = mel_octave_recognition_model.predict(X)
            df_song['prediction'] = Ypredict
            print(df_song.groupby(['prediction'])['prediction'].count())
            df_song['octave'] = [octaves_names[i] for i in Ypredict]
            df_song['note_code'], df_song['note'] = note_identification(df_song)
            df_song.to_csv(output_folder + title+'.csv', index=False)
            x = 'Otra cosa' 
            print(df_song.groupby(['octave'])['octave'].count())
            print(df_song.groupby(['note_code'])['note_code'].count())
            print(df_song.groupby(['note'])['note'].count())
            
#### la vida sigue plop plop
