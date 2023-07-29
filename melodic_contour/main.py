import pandas as pd

from extract_melodic_contour import main_frequencies_songs


if __name__ == '__main__':
    input_folder = '/home/jacs/Documents/DataScience/Personal/'
    output_folder = '/home/jacs/Documents/DataScience/Personal/song_similarity/melodic_contour/'
    df_links = pd.read_csv(input_folder+'song_similarity/' +'audio_model.csv')
    print(df_links)
    links_audio = list(df_links['youtube_links'])
    titles = list(df_links['title'])
    for kk in range(0,len(links_audio)):
        x = 'Algo shady'
        title = titles[kk]
        df_song = main_frequencies_songs(title, input_folder, output_folder)
        df_song.to_csv(output_folder + title+'.csv', index=False)

#### la vida sigue plop plop
