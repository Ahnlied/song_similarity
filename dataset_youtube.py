import pandas as pd
import scipy.io.wavfile as wavfile
import re
from dataset_creation import chunks, extract_peaks_and_freqs, final_data_collection, data_collection_only_peaks
from get_audio_from_link import obtain_youtube_link, delete_spaces, download_audio, remove_audio, from_mp4_to_wav

common_path = '/home/jacs/Documents/DataScience/Personal/'

input_path= 'song_similarity_audio/'

dummy_path = 'song_similarity/'

# As an input there is going to be a list of youtube links that contain audio that can be
# usefull to expand and enrich the database of that particular instrument

# As an input we want a tupple, with the link as first position and instrument as second position

instruments = ['Bass Tuba','French Horn','Trombone','Trumpet in C','Accordion','Cello','Contrabass','Viola','Violin','Alto Saxophone','Bassoon','Clarinet in Bb','Flute','Oboe','Guitar','Electric guitar','Acoustic guitar','Bass guitar','Tapping','Steelpan','Percussion', 'Drum and bass']

def min_to_sec(number):
    number = str(number).split(':')
    print(number)
    number = int(number[0])*60 + int(number[1])
    return number

def audio_partition(audio, Fs, range_1, range_2):
    if range_1=='begin':
        range_1='0:00'
    if  range_2=='end':
        range_2_min = str(int((audio.shape[0]/Fs)/60))
        range_2_seg = str(int((audio.shape[0]/Fs)%60))
        range_2 = range_2_min+':'+range_2_seg
        print(range_2)
    length = audio.shape[0]/Fs
    range_1_index = int(min_to_sec(range_1)*Fs)
    range_2_index = int(min_to_sec(range_2)*Fs)
    audio_cut = audio[range_1_index:range_2_index]
    return audio_cut

def main():
    for kk in range(0,len(links_audio)):
        indexoo = 0
        range_1 = str(df_links['from'].iloc[kk])
        range_2 = str(df_links['to'].iloc[kk])
        print(range_1,range_2)
        linko = links_audio[kk]
        try:
            title_file = str(download_audio(linko))
        except:
            continue
        database_name = str(titles[kk])
#        df_final = pd.DataFrame({'index':[], 'peak_1': [], 'peak_2': [], 'Magnitude difference': [],'instrument': [], 'note_played': []})
        df_final = pd.DataFrame({'index':[], 'peaks': [], 'instrument': [], 'note_played': []})
        try:
            Fs, audio = wavfile.read(title_file+'.wav')
        except:
            from_mp4_to_wav(title_file+'.mp4',common_path+dummy_path)
            Fs, audio = wavfile.read(title_file+'.wav')
            remove_audio(title_file+'.mp4')
        length = audio.shape[0] / Fs
        audio = audio_partition(audio, Fs, range_1, range_2)
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
            try:
                pikos_sorted, freq_sorted, sp_final, peaks  = extract_peaks_and_freqs(aud, Fs)
            except:
                continue
#            df_final_2 = final_data_collection(freq_sorted, pikos_sorted, 10, kk, title_file, indexoo).reset_index(drop=True)
            df_final_2 = data_collection_only_peaks(freq_sorted, pikos_sorted, 10, kk, title_file, indexoo).reset_index(drop=True)
            df_final = pd.concat((df_final,df_final_2), axis=0).reset_index(drop=True)
            indexoo += 1
#        df_final = df_final.drop_duplicates().reset_index(drop=True)
        df_final.to_csv(common_path+input_path+instrument_folder+'/'+database_name+'.csv', index=False)
        remove_audio(title_file+'.wav')
        
if __name__ == '__main__':
    for kk in range(0,len(instruments)):
        instrument = instruments[kk]
        print(instrument)
        instrument_folder = re.sub(' ','_',str(instrument)).lower()#+'/'
        df_links = pd.read_csv(common_path+input_path+instrument_folder+'/'+'{}_youtube_database_enrichment.csv'.format(instrument_folder))
        links_audio = list(df_links['youtube_links'])
        titles = list(df_links['title'])
        main()
    
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
