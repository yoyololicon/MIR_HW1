from librosa import load
from librosa.feature import chroma_stft
from key_finding_baseline import audio_dir, audio_ext, label_dir, label_ext, test_genres, mirex_evaluate
from ks_key_finding import template
from scipy.stats import pearsonr
from prettytable import PrettyTable
import numpy as np
import os
import matplotlib.pyplot as plt