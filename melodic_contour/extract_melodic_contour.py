import pandas as pd
import numpy as np
import scipy

import librosa
import pytube

import json


####### GET top records
#top_2_idx = np.argsort(sp)[-40:]
#listo_freq = [freq[i] for i in top_2_idx]


#######################################################################
# Creates different chunks of data from a signal
# Includes half the past chunk
# Say, you have a 1s signal, you want windows of 0.25s 
# with this methodology you will have each window that includes half of the past window

def chunks(xs, Fs, n):
    # xs: signal
    # n: length of each window
    # len(return): n(even): 2n - 1
    #            : n(odd): 2n +1
    length = len(xs)/Fs
    num_windows = int(length/n)
    ms = int(len(xs)/num_windows)
    return list(xs[i-int(ms/2):i+int(ms/2)] for i in range(int(ms/2), len(xs), int(ms/2)))
#######################################################################

#######################################################################
# Sums amplitudes from repeated frequencies once the int() is applied

def suma_repetidos(freq_int, sp):
    sp_int = []
    frequ_int = []
    for i in range(1,len(freq_int)):
        frequ = int(freq_int[i-1])
        frequ_1 = int(freq_int[i])
        try:
            frequ_2 = int(freq_int[i+1])
        except:
            frequ_2 = int(freq_int[-1])
        if i == 1:
            frequ_int.append(frequ)
        if frequ == frequ_1:
            try:
                sp_new = sp_int[-1] + sp[i+1]
                sp_int.pop()
                sp_int.append(sp_new)
#                print("blop")
            except:
                sp_new = sp[i] + sp[i+1]
                sp_int.append(sp_new)
#                print("blopo")
            if frequ_int[-1] != frequ_1:
                frequ_int.append(frequ_1)
        else:
            sp_new = sp[i]
            sp_int.append(sp_new)
            frequ_int.append(frequ_1)
#            print("plopo")
    return frequ_int, sp_int

#######################################################################
# If a frequency is missing from a continous list of integers will be zero

def exist_or_zero(freq_new,sp,freq_final):
    sp_final = []
    maxxo = max(sp)
    for freq in freq_final:
        if freq in freq_new:
            spe = sp[list(freq_new).index(freq)]
            sp_final.append(spe/maxxo)
        else:
            sp_final.append(0)
    return sp_final

#######################################################################
# Get peaks and frequencies from a signal
# sp_sorted: values of peaks from most intense to less intense
# freq_sorted: frequencies corresponding to each peak from sp_sorted
# sp_final, freq_final: are for visualization purposes
# plt.plot(freq_final, sp_final)
# plt.plot(freq_sorted, sp_sorted, 'x')
# Will plot the fft of the signal and the corresponding values of the peaks

def extract_peaks_and_freqs(aud, Fs):
    sp = scipy.fft.fft(aud)
    freq = scipy.fft.fftfreq(len(aud),Fs)
    freq_sci= scipy.signal.find_peaks(sp)[0]
    sp_sci = scipy.signal.peak_prominences(sp, peaks=freq_sci)[0]
    freq_gaps = np.arange(0,6750,1)
    sp_final = exist_or_zero(freq_sci, sp_sci, freq_gaps)
    freq_final, _ = scipy.signal.find_peaks(sp_final)
    pikos = [sp_final[freq_final[i]] for i in range(0,len(freq_final))]
    sp_sorted = sorted(pikos, reverse= True)
    freq_sorted=[freq_final[pikos.index(pikoso)] for pikoso in sp_sorted]
    return sp_sorted, freq_sorted, sp_final, freq_final

#######################################################################
# Creates a DataFrame that compares the euclidean distance in a plane (freq, sp)
# from peak_1 to peak_2
# Each entry of the df relates to these distances
# The objective is to create a dataset that best describe the spectral composition of the signal
# This module should be applicable to more examples of sounds, such that you can expand the database
# from arbitrary audio files.

def final_data_collection(freq_sorted, pikos_sorted, n, m, note_played, indexo):
    df_final = pd.DataFrame({'index':[], 'peak_1': [], 'peak_2': [], 'Magnitude difference': [],'instrument': [], 'note_played': []})
    frequs_sp = []
    for i in range(0,len(freq_sorted[:n])):
        freq_1, sp_1 = freq_sorted[i], pikos_sorted[i]
        frequs_sp.append((freq_1, sp_1))
        for j in range(0,len(freq_sorted[:n])):
            freq_2, sp_2 = freq_sorted[j], pikos_sorted[j]
            if (freq_2, sp_2) in frequs_sp:
                continue
            else:
                distance = np.sqrt((freq_2 - freq_1)**2 + (sp_2 - sp_1)**2)
                df_iteration = pd.DataFrame({'index': indexo, 'peak_1':[(freq_1, sp_1)], 'peak_2':[(freq_2, sp_2)], 'Magnitude difference': [distance] ,'instrument': m, 'note_played': [note_played]})
                df_final = pd.concat((df_final,df_iteration), axis=0)
    return df_final

#######################################################################
# This dataset is to only consider the (freq,peak) relation
# n: number of peaks
# m: index of the instrument

def data_collection_only_peaks(freq_sorted, pikos_sorted, n, m, note_played, indexo):
#    df_final = pd.DataFrame({'index':[], 'peaks': [],'instrument': [], 'note_played': []})
    freq_peaks = []
    for i in range(0,n):
        freq_peaks.append((freq_sorted[i],pikos_sorted[i]))
    df_iteration = pd.DataFrame({'index': indexo, 'peaks':[freq_peaks],'instrument': m, 'note_played': [note_played]})
#    df_final = pd.concat((df_final,df_iteration), axis=0)
    return df_iteration

def hist_sum(freqs, sp, bins, sp_sum):
    min_max_freq = np.arange(200,2000,1)
    interval = int(len(min_max_freq)/bins)
    freq_ranges = [(min(min_max_freq[i*interval:(i+1)*interval]),max(min_max_freq[i*interval:(i+1)*interval])) for i in range(0,bins)]
#    print(freq_ranges)
    for kk in range(0,len(freqs)):
        freq = freqs[kk]
        for rangee in freq_ranges:
            ###### DUDA AQUI
            rangee_string = str(rangee)
            ###############
            ###### DUDA AQUI 2
            if freq in range(int(rangee[0]),int(rangee[1])):
                try:
                    sp_sum[rangee_string] += sp[kk]
                except:
                    sp_sum[rangee_string] = sp[kk]
            else:
                try:
                    sp_sum[rangee_string] += 0
                except:
                    sp_sum[rangee_string] = 0
            ###############
    return sp_sum

def main_frequencies_songs(examples, input_folder, output_folder, background):
    df_dummy = pd.DataFrame()
    df_final3 = pd.DataFrame()
    if background == 1:
        audio, Fs = librosa.load(input_folder+ 'song_similarity/foreground_signal.wav')
        dataset_name = examples+'_foreground'
    else:
        audio, Fs = librosa.load(input_folder + examples+ '.wav')
        dataset_name = examples
    length = audio.shape[0] / Fs
    bpms = int(librosa.feature.tempo(y=audio, sr=Fs)[0])
    print(f"length = {length}s")
    print(f"num of chunks = {length*(60/bpms)}")
    chonkos = chunks(audio, Fs, 60/bpms)
    for chunk in chonkos:
        try:
            sp_sorted, freq_sorted, sp_final, freq_final = extract_peaks_and_freqs(chunk, Fs)
            rms = librosa.feature.rms(y=chunk, frame_length=len(chunk), hop_length=int(len(chunk)+2))
            spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=Fs, n_fft=len(chunk), hop_length=int(len(chunk)+2))
            rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=Fs, n_fft=len(chunk), hop_length=int(len(chunk)+2))
            zcr = librosa.feature.zero_crossing_rate(chunk, frame_length=len(chunk), hop_length=int(len(chunk)+2))
            df_final_2 = pd.DataFrame({'10_freq': [list(freq_sorted)[:10]], '10_peak': [list(sp_sorted)[:10]], 'song_played': [examples]})
            df_final_2['rms']= rms[0,:]
            df_final_2['spec_cent']= spec_cent[0,:]
            df_final_2['rolloff']= rolloff[0,:]
            df_final_2['zcr']= zcr[0,:]
            df_final3 = pd.concat((df_final3,df_final_2), axis=0).reset_index(drop=True)
        except:
            continue
    df_dummy = pd.concat((df_dummy,df_final3), axis=0).reset_index(drop=True)
    print(df_dummy)
    print('plop')
    df_dummy.to_csv(output_folder + dataset_name +'.csv', index=False)
    df_dummy = pd.read_csv(output_folder + dataset_name+'.csv')
    df_final = pd.DataFrame()
    for kk in range(0,len(df_dummy)):
        note_played = df_dummy.iloc[kk]['song_played']
        chunk_rms = df_dummy.iloc[kk]['rms']
        chunk_spec_cent = df_dummy.iloc[kk]['spec_cent']
        chunk_rolloff= df_dummy.iloc[kk]['rolloff']
        chunk_zcr= df_dummy.iloc[kk]['zcr']
        sp_sum = dict()
        freqos = df_dummy.iloc[kk]['10_freq']
        freqos = list(freqos.replace('[','').replace(']','').split(', '))
        try:
            freqos = [float(freqo) for freqo in freqos]
        except:
            continue
        spos = df_dummy.iloc[kk]['10_peak']
        spos = spos.replace('[','').replace(']','').split(', ')
        spos = [float(spo) for spo in spos]
        sp_sum= hist_sum(freqos, spos, 90, sp_sum)
        df_final3 = pd.DataFrame({'freq_sp': [sp_sum], 'song_played': [examples], 'rms': [chunk_rms], 'spec_cent':[chunk_spec_cent],
                                  'rolloff': [chunk_rolloff], 'zcr': [chunk_zcr]})
        df_final = pd.concat((df_final,df_final3), axis=0).reset_index(drop=True)
    print(sp_sum)
    freqsp = dict(json.loads(str(df_final['freq_sp'].iloc[0]).replace("'",'"')))
    freqs = list(freqsp.keys())
    for entry in freqs:
        entry_dummy = []
        for ii in range(0,len(df_final)):
            dict_dummy = dict(json.loads(str(df_final['freq_sp'].iloc[ii]).replace("'",'"')))
            entry_dummy.append(dict_dummy[entry])
        df_final[entry] = entry_dummy
    df_final = df_final[['song_played',
                         '(200, 219)', '(220, 239)', '(240, 259)', '(260, 279)', '(280, 299)',
                         '(300, 319)', '(320, 339)', '(340, 359)', '(360, 379)', '(380, 399)',
                         '(400, 419)', '(420, 439)', '(440, 459)', '(460, 479)', '(480, 499)',
                         '(500, 519)', '(520, 539)', '(540, 559)', '(560, 579)', '(580, 599)',
                         '(600, 619)', '(620, 639)', '(640, 659)', '(660, 679)', '(680, 699)',
                         '(700, 719)', '(720, 739)', '(740, 759)', '(760, 779)', '(780, 799)',
                         '(800, 819)', '(820, 839)', '(840, 859)', '(860, 879)', '(880, 899)',
                         '(900, 919)', '(920, 939)', '(940, 959)', '(960, 979)', '(980, 999)',
                         '(1000, 1019)', '(1020, 1039)', '(1040, 1059)', '(1060, 1079)',
                         '(1080, 1099)', '(1100, 1119)', '(1120, 1139)', '(1140, 1159)',
                         '(1160, 1179)', '(1180, 1199)', '(1200, 1219)', '(1220, 1239)',
                         '(1240, 1259)', '(1260, 1279)', '(1280, 1299)', '(1300, 1319)',
                         '(1320, 1339)', '(1340, 1359)', '(1360, 1379)', '(1380, 1399)',
                         '(1400, 1419)', '(1420, 1439)', '(1440, 1459)', '(1460, 1479)',
                         '(1480, 1499)', '(1500, 1519)', '(1520, 1539)', '(1540, 1559)',
                         '(1560, 1579)', '(1580, 1599)', '(1600, 1619)', '(1620, 1639)',
                         '(1640, 1659)', '(1660, 1679)', '(1680, 1699)', '(1700, 1719)',
                         '(1720, 1739)', '(1740, 1759)', '(1760, 1779)', '(1780, 1799)',
                         '(1800, 1819)', '(1820, 1839)', '(1840, 1859)', '(1860, 1879)',
                         '(1880, 1899)', '(1900, 1919)', '(1920, 1939)', '(1940, 1959)',
                         '(1960, 1979)', '(1980, 1999)', 'rms', 'spec_cent',
                         'rolloff', 'zcr']]
    df_final.to_csv(output_folder + dataset_name +'.csv', index=False)
    print('se guardo')
#    df_final.to_csv(output_folder + note+'.csv',index=False)
    return df_final


