##### COMMON PACKAGES
import pandas as pd
import pickle
import re
import librosa
import numpy as np
import os

import sys

#############################

###### MELODIC CONTOUR PACKAGES

import json

sys.path.insert(0, '/home/jacs/Documents/DataScience/Personal/song_similarity/melodic_contour')
from extract_melodic_contour import main_frequencies_songs

############################

##### INSTRUMENT RECOGNITION PACKAGES
import scipy.io.wavfile as wavfile
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, '/home/jacs/Documents/DataScience/Personal/song_similarity/instrument_identification/')
from feature_extraction import  mel_freq_cepstrum, dataset_merge #chunks, extract_peaks_and_freqs, final_data_collection, data_collection_only_peaks

############################

##### CHANGE ROUTE OF OUTPUTS
sys.path.insert(0, '/home/jacs/Documents/DataScience/Personal/song_similarity/')
import repet
from audio_from_link import delete_spaces, download_audio, remove_audio, from_mp4_to_wav #obtain_youtube_link

############################


with open("melodic_contour/main.py") as f:
    exec(f.read())

with open("instrument_identification/main.py") as f:
    exec(f.read())
