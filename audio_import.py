# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile


audio_files = ['trumpet-fsharp-natural-minor', 'trumpet-b-major']

for audio in audio_files:
    Fs, aud = wavfile.read(audio + '.wav')
    # select left channel only
    try:
        aud = aud[:,0]
        print('plop')
    except:
        aud = aud[:]
        print('anti-plop')
    print(aud.shape)
    # trim the first 125 seconds
    first = aud[:int(Fs*125)]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(first, Fs=Fs)
    plt.title(audio)
#    plt.show()
    plt.savefig('{}.png'.format(audio))
