import pandas as pd
import re
import os

def audioset_to_enrichment(df_eval):
    df_eval=df_eval.reset_index(drop=True)
    cols = list(df_eval.columns)
    enrichment = pd.DataFrame({'youtube_links':[],'title':[],'from':[],'to':[]})
    enrichment['youtube_links'] = df_eval[cols[0]].apply(lambda x: 'https://www.youtube.com/watch?v='+str(x))
    enrichment['from'] = df_eval[cols[1]].apply(lambda x: str(int(x/60))+':'+str(int(x%60)))
    enrichment['to']= df_eval[cols[2]].apply(lambda x: str(int(x/60))+':'+str(int(x%60)))
    return enrichment

common_path = '/home/jacs/Documents/DataScience/Personal/'
input_path = 'song_similarity_audio/TinySOL/'
output_path = 'song_similarity_audio/'

instruments_to_audioset = {'Bass Tuba': [], 
                           'French Horn': ['/m/0319l'], 
                           'Trombone': ['/m/07c6l'], 
                           'Trumpet in C': ['/m/07gql'], 
                           'Accordion': ['/m/0mkg'], 
                           'Cello':['/m/01xqw'], 
                           'Contrabass': ['/m/02fsn'], 
                           'Viola':[], 
                           'Violin':['/m/07y_7', '/m/02qmj0d'],
                           'Bowed string instrument': ['/m/0l14_3','/m/02qmj0d'],
                           'Alto Saxophone': ['/m/06ncr'], 
                           'Bassoon':[], 
                           'Clarinet in Bb':['/m/01wy6'], 
                           'Flute':['/m/0l14j_'], 
                           'Oboe':[],
                           'Wind instrument':['/m/085jw'],
                           'Guitar':['/m/0342h'], 
                           'Electric guitar':['/m/02sgy'], 
                           'Acoustic guitar':['/m/042v_gx'], 
                           'Bass guitar':['/m/018vs'], 
                           'Tapping':['/m/01glhc'],
                           'Steelpan':['/m/0l156b'], 
                           'Percussion':['/m/0l14md'], 
                           'Drum and bass':['/m/0283d']}

audioset = ['eval_segments', 'balanced_train_segments', 'unbalanced_train_segments']

for dataset in audioset:
    df_eval = pd.read_csv('/home/jacs/Documents/DataScience/Personal/song_similarity_audio/references/{}.csv'.format(dataset), header=2,skipinitialspace=True)
    df_eval['positive_labels'] = df_eval['positive_labels'].apply(lambda x: x.split(','))
    instruments = list(instruments_to_audioset.keys())    
    for kk in range(0,len(instruments)):
        instrument = instruments[kk]
        instrument_folder = re.sub(' ','_',str(instrument)).lower()#+'/'
        output_file = '{}_youtube_database_enrichment'.format(instrument_folder)
        if not os.path.exists(common_path+output_path+instrument_folder):
            os.makedirs(common_path+output_path+instrument_folder)
        eval1 = df_eval[df_eval['positive_labels'].apply(lambda x: len(set(instruments_to_audioset[instrument]) &  set(x)) > 0)]
        eval1 = audioset_to_enrichment(eval1).reset_index(drop=True)
        eval1['title'] = range(0,len(eval1))
        eval1['title'] =eval1['title'].apply(lambda x: 'database_{}_20_peaks_youtube_'.format(instrument_folder)+str(x))
        eval1.to_csv(common_path+output_path+instrument_folder+'/'+output_file+'.csv', index=False)
