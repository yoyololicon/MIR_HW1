from librosa import load
from librosa.feature import chroma_stft
from scipy.stats import pearsonr
from prettytable import PrettyTable
import numpy as np
import os
import matplotlib.pyplot as plt

audio_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/GTZAN/genres'
label_dir = '/media/ycy/86A4D88BA4D87F5D/DataSet/GTZAN/gtzan_key/genres'
test_genres = ['pop', 'blues', 'metal', 'hiphop', 'rock']
audio_ext = '.au'
label_ext = '.lerch.txt'

key = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
tmp = [k.lower() for k in key]
key += tmp
key_map = dict(zip(range(24), key))

# Copy from example code
# Generate major scale templates
major_template = np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]]) / np.sqrt(7.0)
# Generate minor scale templates
minor_template = np.array([[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]]) / np.sqrt(7.0)

# move the tonic to A, not C
# minor_template, major_template = np.roll(major_template, -3), np.roll(minor_template, -3)
template = major_template
for i in range(11):
    template = np.append(template, np.roll(major_template, i + 1), axis=0)
for i in range(12):
    template = np.append(template, np.roll(minor_template, i), axis=0)


def simple_evaluate(y, t):
    if y == t:
        return 1
    else:
        print("Wrong:", key_map[t], "is not", key_map[y])
        return 0


def mirex_evaluate(y, t):
    y_is_minor = y // 12
    t_is_minor = t // 12
    if y == t:
        return 1
    elif t_is_minor == y_is_minor and (t + 7) % 12 == y % 12:  # perfect fifth
        return 0.5
    elif y_is_minor > t_is_minor and (y + 3) % 12 == t % 12:  # relative minor
        return 0.3
    elif y_is_minor < t_is_minor and (t + 3) % 12 == y % 12:  # relative major
        return 0.3
    elif t_is_minor != y_is_minor and t % 12 == y % 12:  # parallel
        return 0.2
    else:
        # print("Wrong: ", key_map[t], "is not", key_map[y])
        return 0


if __name__ == '__main__':
    g = 1000
    overall_acc = []
    genre_acc = []

    table = PrettyTable(["Genre", "Same", "Perfect Fifth", "Relative Minor/Major",
                         "Parallel Minor/Major", "Accuracy", "MIREX Accuracy"])

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
                chroma = chroma_stft(y=data, sr=sr, n_fft=4096, base_c=False)
                chroma = np.mean(chroma, axis=1)
                chroma = np.log(1 + g * chroma)
                tonic = np.argmax(chroma)

                corr_major = pearsonr(chroma, template[tonic])[0]
                corr_minor = pearsonr(chroma, template[tonic + 12])[0]

                if corr_major > corr_minor:
                    y = tonic
                else:
                    y = tonic + 12

                acc.append(mirex_evaluate(y, t))

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
    plt.bar(x, genre_acc[:, 1], 1, label="mirex acc", alpha=0.6)
    plt.bar(x, genre_acc[:, 0], 1, label="acc", alpha=0.6)
    plt.xticks(x, test_genres)
    plt.title("Accuracy for each genre")
    plt.legend()
    plt.show()
