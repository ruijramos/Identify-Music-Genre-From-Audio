# feature extractoring and preprocessing data
import sys
import librosa
import pandas as pd
import numpy as np
import os
import csv

#filter warnings
import warnings
warnings.filterwarnings('ignore')

#receive song as input
filename = sys.argv[-1]

#csv header with file name and all attributes to classification
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header = header.split()

#write headers into csv
file = open('testMySong.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

#extract all features from spectogram 
songname = f'{filename}'
y, sr = librosa.load(songname, mono=True, duration=30)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rms(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
for e in mfcc:
	to_append += f' {np.mean(e)}'

#write in file
file = open('testMySong.csv', 'a', newline='')
with file:
	writer = csv.writer(file)
	writer.writerow(to_append.split())

#read and filter data
data = pd.read_csv('testMySong.csv')
print(data)		
