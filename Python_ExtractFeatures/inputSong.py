# feature extractoring and preprocessing data
import sys
import librosa
import pandas as pd
import numpy as np
import os
import csv

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Receive song as input
filename = sys.argv[-1]

# Csv header with file name and all attributes to classification
header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header = header.split()

# Write headers into csv
file = open('testMySong1.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

# Extract all features from spectogram 
songname = f'{filename}'
y, sr = librosa.load(songname, mono=True, duration=30)
# Load the audio as a waveform `y`
# Store the sampling rate as `sr`
chroma_stft = librosa.feature.chroma_stft(y,sr)
rms = librosa.feature.rms(y)
spec_cent = librosa.feature.spectral_centroid(y,sr)
spec_bw = librosa.feature.spectral_bandwidth(y,sr)
rolloff = librosa.feature.spectral_rolloff(y,sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y, sr)
to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
for e in mfcc:
	to_append += f' {np.mean(e)}'

# Write in file
file = open('testMySong1.csv', 'a', newline='')
with file:
	writer = csv.writer(file)
	writer.writerow(to_append.split())

# Read and filter data
data = pd.read_csv('testMySong1.csv')
print(data)		
