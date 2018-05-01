from librosa import load
from librosa.feature import chroma_stft
from utils import audio_dir, audio_ext, label_dir, label_ext, test_genres, mirex_evaluate, ks_template, last_5chars
from scipy.stats import pearsonr
from prettytable import PrettyTable
import numpy as np
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    g = 100
    overall_acc = []
    genre_acc = []

    table = PrettyTable(["Genre", "Same", "Perfect Fifth", "Relative Minor/Major",
                         "Parallel Minor/Major", "Accuracy", "MIREX Accuracy"])

    for genre in test_genres:
        adir = os.path.join(audio_dir, genre)
        ldir = os.path.join(label_dir, genre)
        file_names = [".".join(f.split(".")[:-1]) for f in os.listdir(adir)]
        file_names = sorted(file_names, key=last_5chars)

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
                chroma = chroma_stft(y=data, sr=sr, n_fft=4096, base_c=False)
                chroma = np.mean(chroma, axis=1)
                chroma = np.log(1 + g * chroma)

                prob = np.apply_along_axis(pearsonr, 1, ks_template, chroma)[:, 0]
                y = np.argmax(prob)

                acc.append(mirex_evaluate(y, t))
                print(f + "\t" + str(y))

        table.add_row([genre, acc.count(1), acc.count(0.5), acc.count(0.3), acc.count(0.2),
                       format(acc.count(1) / count, '.6f'),
                       format(np.mean(acc), '.6f')])
        genre_acc.append([acc.count(1) / count, np.mean(acc)])
        print(count, "of samples.")
        overall_acc += acc

    table.add_row(["All", overall_acc.count(1), overall_acc.count(0.5), overall_acc.count(0.3), overall_acc.count(0.2),
                   format(overall_acc.count(1) / len(overall_acc), '.6f'),
                   format(np.mean(overall_acc), '.6f')])
    print(table)
    x = np.arange(len(test_genres))
    genre_acc = np.array(genre_acc)
    plt.bar(x, genre_acc[:, 1], 1, label="mirex acc", alpha=0.5)
    plt.bar(x, genre_acc[:, 0], 1, label="acc", alpha=0.5)
    plt.xticks(x, test_genres)
    plt.title("Accuracy for each genre using KS profile")
    plt.legend()
    plt.show()
