import pandas as pd
import scipy.io.wavfile as wavfile
from dataset_creation import chunks, extract_peaks_and_freqs, final_data_collection

common_path = '/home/jacs/Documents/DataScience/Personal/song_similarity_audio/TinySOL/'
df_into = pd.read_csv(common_path+'TinySOL_metadata.csv')
df_into= df_into[df_into['Needed digital retuning']]
instruments = df_into['Instrument (in full)'].unique()

instruments

instruments_test = ['Trombone', 'Trumpet in C']

#audio_files = ['A-sharp-trumpet', 'B-trumpet', 'C-sharp-trumpet', 'D-sharp-trumpet', 'E-trumpet', 'F-sharp-trumpet', 'G-sharp-trumpet']

df_final = pd.DataFrame({'peak_1': [], 'peak_2': [], 'Magnitude difference': [],'instrument': [], 'note_played': []})

i=0

for woko in df_into[df_into['Instrument (in full)'] == 'Trumpet in C']['Path']:
    wavo = common_path + woko
    titulo = woko.split('/')[-1]
#    file_1 = file.format(audio_1)
#    wavo = path + file_1
    Fs, audio = wavfile.read(wavo)
    length = audio.shape[0] / Fs
    audio_chunks = chunks(audio,int(length)*2)
    print(f"length = {length}s")
    for aud in audio_chunks[2:-2]:
        length_2 = aud.shape[0] / Fs
#        print(Fs)
        # select left channel only
        try:
            aud = aud[:,0]
            print('plop')
        except:
            aud = aud[:]
            print('anti-plop')
        pikos_sorted, freq_sorted, sp_final, peaks  = extract_peaks_and_freqs(aud, Fs)
        print(i)
        i+=1
        df_final_2 = final_data_collection(freq_sorted, pikos_sorted, 10, 1, titulo).reset_index(drop=True)
        df_final = df_final.append(df_final_2).reset_index(drop=True)
#################################
#        plt.specgram(aud, Fs=Fs)
#        plt.xticks(time_cnk)    
#        plt.ylim(0,5000)
#        plt.title(titulo)
#        plt.show()
#        time = np.linspace(0., length_2, aud.shape[0])
#        plt.plot(time, aud)
#        plt.title('Original signal')
#        plt.show()
#        plt.plot(peaks, [sp_final[i] for i in peaks])
#        plt.plot(freq_sorted[:10], pikos_sorted[:10],'x')
#        plt.title('Final frequencies and intensities')
#        plt.show()
#        ss = np.fft.ifft(sp_final)
#        time = np.linspace(0., length_2, len(sp_final))
#        plt.plot(time, ss)
#        plt.title('Reconstruccion with data manipulation')
#        plt.show()
#################################
