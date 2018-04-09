from librosa import load
from librosa.feature import chroma_stft
from key_finding_baseline import audio_dir, audio_ext, label_dir, label_ext, test_genres, simple_evaluate, \
    mirex_evaluate
from ks_key_finding import template
from scipy.stats import pearsonr
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    g = 100
    overall_acc = []
    genre_acc = []
    prob = np.zeros(24)
    for genre in test_genres:
        adir = os.path.join(audio_dir, genre)
        ldir = os.path.join(label_dir, genre)
        file_names = [".".join(f.split(".")[:-1]) for f in os.listdir(adir)]

        acc = []
        count = 0
        for f in file_names:
            with open(os.path.join(ldir, f + label_ext)) as label_file:
                t = int(label_file.readline())
                if t < 0:
                    continue
                count += 1
                data, sr = load(os.path.join(adir, f + audio_ext), sr=None)
                chroma = chroma_stft(y=data, sr=sr, n_fft=4096, base_c=False, norm=None)
                chroma = np.log(1 + g * np.abs(chroma))
                chroma = np.sum(chroma, axis=1)
                for i in range(24):
                    prob[i] = pearsonr(chroma, template[i])[0]
                weight = np.append(chroma, chroma)/chroma.max()
                y = np.argmax(prob * weight)

                #acc.append(simple_evaluate(y, t))
                acc.append(mirex_evaluate(y, t))

        genre_acc.append(np.mean(acc))
        print("After", count, "of samples, the accuracy for genre", genre, "is", genre_acc[-1])
        overall_acc += acc

    print("Overall accuracy:", np.mean(overall_acc))
    x = np.arange(len(test_genres))
    plt.bar(x, genre_acc, 1)
    plt.xticks(x, test_genres)
    plt.title("Accuracy for each genre")
    plt.show()
