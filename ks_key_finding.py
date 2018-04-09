from librosa import load
from librosa.feature import chroma_stft
from key_finding_baseline import audio_dir, audio_ext, label_dir, label_ext, test_genres, simple_evaluate, \
    mirex_evaluate
from scipy.stats import pearsonr
import numpy as np
import os
import matplotlib.pyplot as plt

major_template = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]])
minor_template = np.array([[6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])

template = major_template
for i in range(11):
    template = np.append(template, np.roll(major_template, i + 1), axis=0)
for i in range(12):
    template = np.append(template, np.roll(minor_template, i), axis=0)

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
                y = np.argmax(prob)

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
