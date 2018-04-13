from librosa import load
from librosa.feature import chroma_stft
from key_finding_baseline import audio_dir, audio_ext, label_dir, label_ext, test_genres, mirex_evaluate
from ks_key_finding import template
from scipy.stats import pearsonr
from prettytable import PrettyTable
import numpy as np
import os
import matplotlib.pyplot as plt

blues_major_template = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])/np.sqrt(6)
blues_minor_template = np.array([[1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]])/np.sqrt(6)
for i in range(12):
    template = np.append(template, np.roll(blues_major_template, i), axis=0)
for i in range(12):
    template = np.append(template, np.roll(blues_minor_template, i), axis=0)

if __name__ == '__main__':
    g = 1000
    overall_acc = []
    genre_acc = []
    prob = np.zeros(48)
    table = PrettyTable(["Genre", "Num of Same", "Num of Perfect Fifth", "Num of Relative Minor/Major",
                         "Num of Parallel Minor/Major", "Accuracy", "Mirex Accuracy"])
    for genre in test_genres:
        adir = os.path.join(audio_dir, genre)
        ldir = os.path.join(label_dir, genre)
        file_names = [".".join(f.split(".")[:-1]) for f in os.listdir(adir)]

        acc = []
        count = 0
        print("Running genre", genre, "...")
        for f in file_names:
            with open(os.path.join(ldir, f + label_ext)) as label_file:
                t = int(label_file.readline())
                if t < 0:
                    continue
                count += 1
                data, sr = load(os.path.join(adir, f + audio_ext), sr=None)
                chroma = chroma_stft(y=data, sr=sr, n_fft=4096, base_c=False, norm=None)
                chroma = np.log(1 + g * chroma)
                chroma = np.sum(chroma, axis=1)
                for i in range(48):
                    prob[i] = pearsonr(chroma, template[i])[0]
                weight = np.tile(chroma, 4)
                y = np.argmax(prob * weight) % 24

                acc.append(mirex_evaluate(y, t))

        table.add_row([genre, acc.count(1), acc.count(0.5), acc.count(0.3), acc.count(0.2),
                       format(acc.count(1) / count, '.6f'),
                       format(np.mean(acc), '.6f')])
        genre_acc.append([acc.count(1) / count, np.mean(acc)])
        print(count, "of samples.")
        overall_acc += acc

    table.add_row(
        ["All", overall_acc.count(1), overall_acc.count(0.5), overall_acc.count(0.3), overall_acc.count(0.2),
         format(overall_acc.count(1) / len(overall_acc), '.6f'),
         format(np.mean(overall_acc), '.6f')])
    print(table)
    x = np.arange(len(test_genres))
    genre_acc = np.array(genre_acc)
    plt.bar(x, genre_acc[:, 1], 1, label="mirex acc", alpha=0.5)
    plt.bar(x, genre_acc[:, 0], 1, label="acc", alpha=0.5)
    plt.xticks(x, test_genres)
    plt.title("Accuracy for each genre using my own method")
    plt.legend()
    plt.show()
