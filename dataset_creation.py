# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy
# for dataframe manipulation
import pandas as pd

####### GET top records
#top_2_idx = np.argsort(sp)[-40:]
#listo_freq = [freq[i] for i in top_2_idx]

#######################################################################
# Creates different chunks of data from a signal
# Includes half the past chunk
# Say, you have a 1s signal, you want windows of 0.25s 
# with this methodology you will have each window that includes half of the past window

def chunks(xs, n):
    # xs: signal
    # n: times you want to split the signal
    # len(return): n(even): 2n - 1
    #            : n(odd): 2n +1
    ms = int(len(xs)/n)
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
