import pandas as pd
import scipy.io.wavfile as wavfile
import re
import librosa
import os
from feature_extraction import chunks, extract_peaks_and_freqs, final_data_collection, data_collection_only_peaks, mel_freq_cepstrum, dataset_merge
from audio_from_link import delete_spaces, download_audio, remove_audio, from_mp4_to_wav #obtain_youtube_link

common_path = '/home/jacs/Documents/DataScience/Personal/'
input_path= 'song_similarity_audio/'
dummy_path = 'song_similarity/'


lapse = 10

# As an input there is going to be a list of youtube links that contain audio that can be
# usefull to expand and enrich the database of that particular instrument

# As an input we want a tupple, with the link as first position and instrument as second position

instruments = ['Bass Tuba','French Horn','Trombone','Trumpet in C','Accordion','Cello','Contrabass','Viola','Violin','Alto Saxophone','Bassoon','Clarinet in Bb','Flute','Oboe','Guitar','Electric guitar','Acoustic guitar','Bass guitar','Tapping','Steelpan','Percussion', 'Drum and bass', 'Wind instrument', 'Bowed string instrument', 'Keyboard', 'Single Voice Singing', 'Group Singing', 'Noise']

# bowed = 23

instruments_reduced = {'woodwind': ['Clarinet in Bb','Flute','Oboe', 'Bassoon', 'Alto Saxophone', 'Wind Instrument'],
                       'brass':['Bass Tuba','French Horn','Trombone', 'Trumpet in C'],
                       'bowed_string_instruments': ['Cello','Viola','Violin', 'Bowed string instrument'],
                       'plucked_string_instruments':['Guitar','Electric guitar','Acoustic guitar', 'Tapping', 'Bass guitar','Contrabass'],
                       'keyboard': ['Keyboard','Accordion'],
                       'percussion': ['Steelpan','Percussion', 'Drum and bass'],
                       'singing': ['Single Voice Singing', 'Group Singing'],
                       'noise': ['Noise']}

#instruments = ['Guitar','Electric guitar','Acoustic guitar','Bass guitar','Tapping','Steelpan','Percussion', 'Drum and bass', 'Wind instrument', 'Bowed string instrument']

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

def main_fourier(lapse):
    for kk in range(0,len(links_audio)):
        database_name = str(titles[kk])
        if not os.path.exists(common_path+input_path+instrument_folder+'/'+database_name+'.csv'):
            indexoo = 0
            range_1 = str(df_links['from'].iloc[kk])
            range_2 = str(df_links['to'].iloc[kk])
            print(range_1,range_2)
            linko = links_audio[kk]
            try:
                title_file = str(download_audio(linko))
                print(instrument,kk)#,title_file)
            except:
                print(instrument,kk,"titulo raro")
                continue
            #        df_final = pd.DataFrame({'index':[], 'peak_1': [], 'peak_2': [], 'Magnitude difference': [],'instrument': [], 'note_played': []})
            df_final = pd.DataFrame({'index':[], 'peaks': [], 'instrument': [], 'note_played': []})
            try:
#                audio, Fs = librosa.load(wavo)
                Fs, audio = wavfile.read(title_file+'.wav')
            except:
                from_mp4_to_wav(title_file+'.mp4',common_path+dummy_path)
#                audio, Fs = librosa.load(wavo)
                Fs, audio = wavfile.read(title_file+'.wav')
                remove_audio(title_file+'.mp4')
            length = audio.shape[0] / Fs
#            print(len(audio), Fs)
            audio = audio_partition(audio, Fs, range_1, range_2)
#            print(len(audio), Fs)
            audio_chunks = chunks(audio,lapse*2)
            print(f"length = {length}s")
            for aud in audio_chunks[2:-2]:
                length_2 = aud.shape[0] / Fs
                #        print(Fs)
                # select left channel only
                try:
                    aud = aud[:,0]
#                    print('plop')
                except:
                    aud = aud[:]
#                    print('anti-plop')
                try:
                    pikos_sorted, freq_sorted, sp_final, peaks  = extract_peaks_and_freqs(aud, Fs)
                    df_final_2 = data_collection_only_peaks(freq_sorted, pikos_sorted, 20, kk, title_file, indexoo).reset_index(drop=True)
                    df_final = pd.concat((df_final,df_final_2), axis=0).reset_index(drop=True)
                except:
                    print("error con dimensiones de los pikos")
                    continue
                indexoo += 1
            #            df_final_2 = final_data_collection(freq_sorted, pikos_sorted, 10, kk, title_file, indexoo).reset_index(drop=True)
            #        df_final = df_final.drop_duplicates().reset_index(drop=True)
            df_final.to_csv(common_path+input_path+instrument_folder+'/'+database_name+'.csv', index=False)
            remove_audio(title_file+'.wav')
        else:
            print(instrument,kk,"ya existe, we")

def main_cepstrum_dataset():
    for kk in range(0,len(links_audio)):
        try:
            database_name = str(titles[kk])
            if not os.path.exists(common_path+input_path+instrument_folder+'/'+database_name+'.csv'):
                indexoo = 0
                range_1 = str(df_links['from'].iloc[kk])
                range_2 = str(df_links['to'].iloc[kk])
                print(range_1,range_2)
                linko = links_audio[kk]
                try:
                    title_file = str(download_audio(linko))
                    print(instrument,kk)#,title_file)
                except:
                    print(instrument,kk,"titulo raro")
                    continue
            #        df_final = pd.DataFrame({'index':[], 'peak_1': [], 'peak_2': [], 'Magnitude difference': [],'instrument': [], 'note_played': []})
                df_final = pd.DataFrame()
                #            df_final = pd.DataFrame({'index':[], 'mfccs_envelope': [], 'instrument': [], 'note_played': []})
                try:
                    audio, Fs = librosa.load(title_file+'.wav')
                except:
                    from_mp4_to_wav(title_file+'.mp4',common_path+dummy_path)
                    audio, Fs = librosa.load(title_file+'.wav')
                    remove_audio(title_file+'.mp4')
                audio = audio_partition(audio, Fs, range_1, range_2)
                length = audio.shape[0] / Fs
                df_final_2 = mel_freq_cepstrum(audio, Fs, 13, pp, title_file)
                df_final = pd.concat((df_final,df_final_2), axis=0).reset_index(drop=True)
                #            df_final_2 = final_data_collection(freq_sorted, pikos_sorted, 10, kk, title_file, indexoo).reset_index(drop=True)
                #        df_final = df_final.drop_duplicates().reset_index(drop=True)
                df_final.to_csv(common_path+input_path+instrument_folder+'/'+database_name+'.csv', index=False)
                remove_audio(title_file+'.wav')
            else:
                print(instrument,kk,"ya existe, we")
            if kk > 10000:
                break
        except:
            continue


def instr_reduced(final_data_path):
    for instrument_final in instruments_reduced:
        df_final = pd.DataFrame()
        for instrument in innstrument_final:
            instrument_folder = re.sub(' ','_',str(instrument)).lower()
            df_instrument = pd.read_csv(final_data_path+'{}.csv'.format(instrument_folder))
            df_final = pd.concat((df_final, df_instrument), axis=0)
        df_final.to_csv(final_data_path+'{}.csv'.format(instrument_final))
            

model_run = 1
        
    
if __name__ == '__main__':
    for pp in range(0,len(instruments)):
        instrument = instruments[pp]
        print(instrument)
        instrument_folder = re.sub(' ','_',str(instrument)).lower()#+'/'
        df_links = pd.read_csv(common_path+input_path+instrument_folder+'/'+'{}_youtube_database_enrichment.csv'.format(instrument_folder))
        links_audio = list(df_links['youtube_links'])       
        #        print(instrument,len(links_audio))
        titles = list(df_links['title'])
        main_cepstrum_dataset()
        input_data_path = common_path + input_path+ instrument_folder
        final_data_path = common_path + dummy_path + 'instrument_identification/data/' #instrument_folder
        dataset_merge(input_data_path, instrument_folder, final_data_path)
        instr_reduced(final_data_path)
        main_fourier(lapse)
    
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
