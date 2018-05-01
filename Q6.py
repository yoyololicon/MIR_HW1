from librosa import load
from librosa.feature import chroma_stft
from utils import mirex_evaluate, ks_template, inv_key_map
from scipy.stats import pearsonr
from scipy.signal import medfilt, fftconvolve
import numpy as np
import os

data_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/BPS_piano'
ref_prefix = 'REF_key_'
predict_dir = 'predict_results/'

key_map = {v: k for k, v in inv_key_map.items()}

if __name__ == '__main__':
    file_names = [".".join(f.split(".")[:-1]) for f in os.listdir(data_dir) if f[-4:] == '.wav']
    file_names.sort(key=float)

    d = 10
    g = 100
    w = 641
    mean_filt = np.ones(w) / w
    mean_filt2 = np.ones(w // 2 // d + 1) / (w // 2 // d + 1)
    overall_acc = []

    sym2num = np.vectorize(inv_key_map.get)
    num2sym = np.vectorize(key_map.get, otypes=[np.str])
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
        chroma_a = chroma_a.reshape(12, len(label), d).mean(axis=2)
        chroma_a = np.log(1 + g * chroma_a)

        chroma_a = np.apply_along_axis(fftconvolve, 1, chroma_a, mean_filt2, 'same')

        prob = np.zeros((ks_template.shape[0], chroma_a.shape[1]))
        for n in range(chroma_a.shape[1]):
            prob[:, n] = np.apply_along_axis(pearsonr, 1, ks_template, chroma_a[:, n])[:, 0]

        y = np.argmax(prob, axis=0)
        y = medfilt(y, 9)
        acc = evaluate_vec(y, t).tolist()

        print(f + '.wav', format(acc.count(1) / len(acc), '.6f'), format(np.mean(acc), '.6f'))
        overall_acc += acc

        y_inv = num2sym(y)
        np.savetxt(os.path.join(predict_dir, 'PRED_key_' + f + '.txt'), np.column_stack((np.arange(len(y)), y_inv)),
                   delimiter='\t', fmt='%s')

    print(format(overall_acc.count(1) / len(overall_acc), '.6f'), format(np.mean(overall_acc), '.6f'))
