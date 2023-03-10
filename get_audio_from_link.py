import os
import ffmpy
from pytube import YouTube
from pytube import Search
import re
from youtube_search import YoutubeSearch

#inputdir = '/home/jacs/Documents/DataScience/Personal/song_similarity/'

######################################################
# To obtain youtube links of a search, the primarly objective was to find
# a query which collects good data sample from Youtube,
# but to find a generic search might be a bit tricky.

def obtain_youtube_link(busqueda, maxim_results):
    results = YoutubeSearch(busqueda, max_results=maxim_results).to_dict()
    linkos_kun = []
    for v in results:
        linkos_kun.append('https://www.youtube.com' + v['url_suffix'])
    return linkos_kun

######################################################
# To delete spaces from the title of a video and replace it with an underscore

def delete_spaces(selected_video):
    title2 = selected_video.title
    title2 = re.sub("[)'!?/(]","",title2)
    title2 = re.sub('"',"",title2)
    title2 = re.sub("(\s)","_"," ".join(re.sub("(-)|(&)|(\[)|(\])","",title2).split()))
    return title2


#######################################################
# To download audio from a Youtube link to then run the data collection

def download_audio(link):
    selected_video = YouTube(link)
    title = delete_spaces(selected_video)
    audio = selected_video.streams.filter(only_audio=True, file_extension='mp4').first()
    audio.download(filename=title+'.mp4')
    return title

#######################################################
# Change audio format
def from_mp4_to_wav(filename, inputdir):
    for filename in os.listdir(inputdir):
        actual_filename = filename[:-4]
        if(filename.endswith(".mp4")):
            os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(filename, actual_filename))
        else:
            continue
#######################################################

# To remove the audio from local file

def remove_audio(title):
    os.remove(title)
