import numpy as np

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
bin_major_template = np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]]) / np.sqrt(7.0)
# Generate minor scale templates
bin_minor_template = np.array([[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]]) / np.sqrt(7.0)

bin_template = bin_major_template
for i in range(11):
    bin_template = np.append(bin_template, np.roll(bin_major_template, i + 1), axis=0)
for i in range(12):
    bin_template = np.append(bin_template, np.roll(bin_minor_template, i), axis=0)

ks_major_template = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]])
ks_minor_template = np.array([[6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
ks_major_template /= np.sqrt((ks_major_template ** 2).sum())
ks_minor_template /= np.sqrt((ks_minor_template ** 2).sum())

ks_template = ks_major_template
for i in range(11):
    ks_template = np.append(ks_template, np.roll(ks_major_template, i + 1), axis=0)
for i in range(12):
    ks_template = np.append(ks_template, np.roll(ks_minor_template, i), axis=0)

blues_major_template = np.array([[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]]) / np.sqrt(6)
blues_minor_template = np.array([[1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0]]) / np.sqrt(6)
blues_template = blues_major_template
for i in range(11):
    blues_template = np.append(blues_template, np.roll(blues_major_template, i + 1), axis=0)
for i in range(12):
    blues_template = np.append(blues_template, np.roll(blues_minor_template, i), axis=0)


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