from librosa import load
from librosa.feature import chroma_stft, chroma_cens
from key_finding_baseline import mirex_evaluate
from ks_key_finding import template
from scipy.stats import pearsonr
from scipy.signal import medfilt, decimate, fftconvolve
from prettytable import PrettyTable
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/BPS_piano'
ref_prefix = 'REF_key_'

key = ['A', 'B-', 'B', 'C', 'D-', 'D', 'E-', 'E', 'F', 'G-', 'G', 'A-']
tmp = [k.lower() for k in key]
key += tmp
inv_key_map = dict(zip(key, range(24)))

key2 = ['A', 'A+', 'B', 'C', 'C+', 'D', 'D+', 'E', 'F', 'F+', 'G', 'G+']
tmp = [k.lower() for k in key2]
key2 += tmp
for k, i in zip(key2, range(24)):
    if k not in inv_key_map:
        inv_key_map[k] = i

if __name__ == '__main__':
    file_names = [".".join(f.split(".")[:-1]) for f in os.listdir(data_dir) if f[-4:] == '.wav']

    d = 10
    g = 100
    w = 641
    mean_filt = np.ones(w) / w
    overall_acc = []

    sym2num = np.vectorize(inv_key_map.get)
    evaluate_vec = np.vectorize(mirex_evaluate, otypes=[float])

    for f in file_names:
        label = np.loadtxt(os.path.join(data_dir, ref_prefix + f + '.txt'), dtype='str')
        t = sym2num(label[:, 1])

        data, sr = load(os.path.join(data_dir, f + '.wav'), sr=None)
        hop_size = int(sr / d)
        window_size = hop_size * 2

        chroma_a = chroma_stft(y=data, sr=sr, hop_length=hop_size, n_fft=window_size, base_c=False)
        chroma_a = np.apply_along_axis(fftconvolve, 1, chroma_a, mean_filt, 'same')

        if chroma_a.shape[1] > len(label) * d:
            chroma_a = chroma_a[:, :len(label) * d]
        elif chroma_a.shape[1] < len(label) * d:
            chroma_a = np.column_stack((chroma_a, np.zeros((12, len(label) * d - chroma_a.shape[1]))))

        # chroma_a = decimate(chroma_a[:, int(d/2):], d, axis=1)
        chroma_a = chroma_a.reshape(12, -1, d).mean(axis=2)
        chroma_a = np.log(1 + g * chroma_a)

        chroma_a = np.apply_along_axis(fftconvolve, 1, chroma_a, np.ones(w // (2 * d)) / (w // (2 * d)), 'same')

        prob = np.zeros((template.shape[0], chroma_a.shape[1]))
        for n in range(chroma_a.shape[1]):
            prob[:, n] = np.apply_along_axis(pearsonr, 1, template, chroma_a[:, n])[:, 0]

        y = np.argmax(prob, axis=0)
        y = medfilt(y, w // (4 * d) + 1)
        acc = evaluate_vec(y, t).tolist()

        print(f + '.wav', format(acc.count(1) / len(acc), '.6f'), format(np.mean(acc), '.6f'))
        overall_acc += acc

    print(format(overall_acc.count(1) / len(overall_acc), '.6f'), format(np.mean(overall_acc), '.6f'))
