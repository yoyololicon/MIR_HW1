from librosa import load
from librosa.feature import chroma_stft
from librosa.display import specshow
import matplotlib.pyplot as plt

y, sr = load('/media/ycy/86A4D88BA4D87F5D/DataSet/GTZAN/genres/blues/blues.00000.au')
stft = chroma_stft(y=y, sr=sr)

