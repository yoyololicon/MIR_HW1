from librosa import load
from librosa.feature import chroma_stft
from scipy.stats import pearsonr
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
#minor_template, major_template = np.roll(major_template, -3), np.roll(minor_template, -3)
template = major_template
for i in range(11):
    template = np.append(template, np.roll(major_template, i + 1), axis=0)
for i in range(12):
    template = np.append(template, np.roll(minor_template, i), axis=0)


def simple_evaluate(y, t):
    if y == t:
        return 1
    else:
        print("Wrong: ", key_map[t], "is not", key_map[y])
        return 0


def mirex_evaluate(y, t):
    minory = y // 12
    minort = t // 12
    if y == t:
        return 1
    elif minort == minory and (t + 7) % 12 == y % 12:  # perfect fifth
        return 0.5
    elif minory > minort and (y + 3) % 12 == t % 12:  # relative minor
        return 0.3
    elif minory < minort and (t + 3) % 12 == y % 12:  # relative major
        return 0.3
    elif minort != minory and t % 12 == y % 12:  # parallel
        return 0.2
    else:
        print("Wrong: ", key_map[t], "is not", key_map[y])
        return 0


if __name__ == '__main__':
    g = 100
    overall_acc = []
    genre_acc = []
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
                tonic = np.argmax(chroma)

                corr_major = pearsonr(chroma, template[tonic])[0]
                corr_minor = pearsonr(chroma, template[tonic + 12])[0]

                if corr_major > corr_minor:
                    y = tonic
                else:
                    y = tonic + 12

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
