from librosa import load
from librosa.feature import chroma_stft, chroma_cens, chroma_cqt
from key_finding_baseline import mirex_evaluate
from ks_key_finding import template
from scipy.stats import pearsonr
from scipy.signal import medfilt
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

    divider = 21
    prob = np.zeros(24)
    med_num = 20
    mean_filt = np.ones(med_num)/med_num
    overall_acc = []

    for f in file_names:
    #f = '22'
        label = np.loadtxt(os.path.join(data_dir, ref_prefix + f + '.txt'), dtype='str')
        label = label[:, 1]

        data, sr = load(os.path.join(data_dir, f + '.wav'), sr=None)
        hop_size = int(sr / divider)
        window_size = 4096

        chroma_c = chroma_stft(y=data, sr=sr, hop_length=hop_size, n_fft=window_size, base_c=False)

        if chroma_c.shape[1] > len(label) * divider:
            chroma_c = chroma_c[:, :len(label) * divider]
        elif chroma_c.shape[1] < len(label) * divider:
            chroma_c = np.column_stack((chroma_c, np.zeros((12, len(label) * divider - chroma_c.shape[1]))))

        chroma_c = np.mean(chroma_c.reshape((12, -1, divider)), axis=2)
        chroma_c = np.apply_along_axis(np.convolve, 1, chroma_c, mean_filt, 'same')

        #chroma_a = np.roll(chroma_c, -3, axis=0)

        y = []
        for i in range(chroma_c.shape[1]):
            for j in range(24):
                prob[j] = pearsonr(chroma_c[:, i], template[j])[0]
            y.append(np.argmax(prob))

        y = medfilt(y, 23)

        acc = []
        for i, (p, t) in enumerate(zip(y, label)):
            acc.append(mirex_evaluate(p, inv_key_map[t]))

        print(f + '.wav', format(acc.count(1)/len(acc), '.6f'), format(np.mean(acc), '.6f'))
        overall_acc += acc

    print(format(overall_acc.count(1)/len(overall_acc), '.6f'), format(np.mean(overall_acc), '.6f'))