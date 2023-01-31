# Feature extractoring and Preprocessing data
import librosa 
import pandas as pd
import numpy as np
import os
import csv

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Csv header with file name and all attributes to classification
header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

# Write header into csv
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Write each attribute of each song in data.csv
for g in genres:
    for filename in os.listdir(f'genres/{g}'):
        songname = f'genres/{g}/{filename}'
        # Load the audio as a waveform `y`
        # Store the sampling rate as `sr`
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y,sr)
        rms = librosa.feature.rms(y)
        spec_cent = librosa.feature.spectral_centroid(y, sr)
        spec_bw = librosa.feature.spectral_bandwidth(y, sr)
        rolloff = librosa.feature.spectral_rolloff(y, sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y,sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

# Print data
data = pd.read_csv('data.csv')
print(data.shape)
data = data.drop(['filename'],axis=1)
print(data)