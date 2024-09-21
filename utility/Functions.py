import scipy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os


def Note_Splitter(waveform, fs):
    idx_split = []
    # Normalize The Data
    waveform = np.divide(np.subtract(waveform, np.min(waveform)), np.subtract(np.max(waveform), np.min(waveform)))

    # Do Variance On Chunks
    split_waveform = np.array(np.split(waveform, 500))
    var_wavform = np.var(split_waveform, axis=1)

    # Find The Start Of A Pick
    for j in range(1, len(var_wavform) - 1):
        if var_wavform[j] < var_wavform[j + 1] / 1.8 and var_wavform[j + 1] > 1e-4:
            idx_split.append(j * split_waveform.shape[1])

    # Organize Splitting
    diff_org = np.diff(idx_split)
    to_delete = np.where(diff_org == split_waveform.shape[1], 1, 0)
    idx_split = np.delete(idx_split, np.where(to_delete))

    # import matplotlib.pyplot as plt
    # # plt.plot(var_wavform)
    # plt.plot(waveform)
    # for idx in idx_split:
    #     plt.axvline(idx, color='r')
    # plt.show()

    # Find Metre
    diff_metre = np.diff(idx_split)
    metre_matrix = np.empty([diff_metre.shape[0],diff_metre.shape[0]])
    for i in range(len(diff_metre)):
        for j in range(len(diff_metre)):
            metre_matrix[i,j] = diff_metre[j]/diff_metre[i]


    # Round Metre
    round_to = np.array([0.25, 0.5, 1, 2, 4])
    for i in range(metre_matrix.shape[0]):
        for j in range(metre_matrix.shape[1]):
            metre_matrix[i,j] = round_to[np.abs(round_to - metre_matrix[i,j]).argmin()]

    # Find Best Solution
    notes_metre = [0.5, 1, 2]
    metre_matrix_bol = np.empty([metre_matrix.shape[0],metre_matrix.shape[1]])
    for i in range(metre_matrix.shape[0]):
        for j in range(metre_matrix.shape[1]):
            is_in_number = np.where(metre_matrix[i][j] == notes_metre, False, True)
            metre_matrix_bol[i,j] = np.all(is_in_number)
    # Convert Bol To True/False
    metre_matrix_bol = np.where(metre_matrix_bol, False, True)
    # Check For Optimal Solution
    correct_metre = np.all(metre_matrix_bol, axis=1)
    correct_metre = np.where(correct_metre)
    correct_metre = metre_matrix[correct_metre[0][0]]

    # Calculate BPM
    bpm = np.average(fs/diff_metre*60*correct_metre)

    return idx_split, correct_metre, bpm


def Note_Identifier(waveform, fs, idx_spliter):
    note_frq = np.array([])
    for idx in idx_spliter:
        # Take Note Sample Out Of waveform
        start = 200
        note_waveform = waveform[idx + start:idx + start + 500]
        # Do FFT On Chunk
        N = note_waveform.size
        yf = scipy.fft.fft(note_waveform)
        yf = np.abs(2 / N * yf[0:N // 2])
        xf = scipy.fft.fftfreq(N, 1 / fs)[:N // 2]

        # Search For 5 Highest Points Of The FFT
        sorting = np.sort(yf)
        a = np.array([])
        for i in range(1, 20):
            a = np.append(a, np.multiply(np.where(yf == sorting[-i]), (xf[1] - xf[0])))
        a = np.sort(a)

        # Delete Close Points
        delete_where = np.where(np.diff(a)<100)
        a = np.delete(a, delete_where)

        # print(a)
        # import matplotlib.pyplot as plt
        # plt.plot(xf,yf)
        # plt.show()

        # Determine The Ratio Of The Highest Points
        diff_a = np.diff(a)
        max_diff = np.min(diff_a)
        arr = np.empty([diff_a.shape[0]])
        for i in range(diff_a.shape[0]):
            arr[i] = diff_a[i] / max_diff
        arr_round = np.round(arr)

        note_frq = np.append(note_frq, np.average(diff_a / arr_round))

    return note_frq


def Frq_To_Note(note_frq):
    notes_name = np.array([])
    for frq in note_frq:
        # Generate Frq Matrix (C2-B6)
        frq_data = np.array([])
        for i in np.arange(-33,27):
            frq_data = np.append(frq_data, 440*2**(i/12))

        # Generate Labels Array
        # notes = np.array(['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B'])
        notes = np.array(['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'])

        # Find The Nearest Frequency
        nearest_frq = np.argmin(np.abs(frq_data - frq))
        # Divade With Remainder
        octave, note = divmod(nearest_frq, 12)
        notes_name = np.append(notes_name, notes[note]+str(octave+2))
    return notes_name


def Load(path_to_data):
    dirs = os.listdir(path_to_data)

    data = np.empty([7*500,500])
    labels = []
    notes_list = []
    for i,note in enumerate(dirs):
        notes_list.append(note)
        file_names = os.listdir(os.path.join(path_to_data, note))
        for file in file_names:
            full_path = os.path.join(path_to_data, note, file)
            fs, file_data = scipy.io.wavfile.read(full_path)
            data[i,:] = file_data
            labels.append(i)

    # for i, notes in enumerate(dirs):
    #     files_names = os.listdir(os.path.join(path_to_data, notes))
    #     for file in files_names:
    #         file_name = os.path.join(path_to_data,notes_list[i],file)
    #         file_data, fs = sf.read(file_name, dtype='float32')
    #         data_all[i].append(file_data)

    data = np.array(data)
    labels = np.array(labels)
    labels = np.reshape(labels, (len(labels), 1))
    transformed = OneHotEncoder().fit_transform(labels)
    labels = transformed.toarray()
    return data, labels, fs


def make_spectogram(waveform, fs):
    f, t, Sxx = scipy.signal.spectrogram(waveform, fs=fs, nperseg=500, nfft=1024)
    return Sxx
