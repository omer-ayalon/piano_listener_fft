import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from utility import Functions


fig, axs = plt.subplots(11)

path_to_dataset = '01-data\\seperating_demos'
path_to_waveforms = os.listdir(path_to_dataset)
for i, path in enumerate(path_to_waveforms):
    path_to_waveforms[i] = os.path.join(path_to_dataset, path_to_waveforms[i])

for i in range(len(path_to_waveforms)):
    idx_split = []
    fs, waveform = scipy.io.wavfile.read(path_to_waveforms[i])
    axs[i].plot(waveform)

    split_waveform = np.array(np.split(waveform, 500))
    var_wavform = np.var(split_waveform, axis=1)

    for j in range(1, len(var_wavform) - 1):
        if var_wavform[j] < var_wavform[j + 1]/1.8 and var_wavform[j + 1] > 5e14:
            idx_split.append(j*split_waveform.shape[1])

    diff = np.diff(idx_split)
    to_delete = np.where(diff==split_waveform.shape[1],1,0)
    idx_split = np.delete(idx_split, np.where(to_delete))

    for j,idx in enumerate(idx_split):
        axs[i].axvline(idx, color='r')

plt.show()


