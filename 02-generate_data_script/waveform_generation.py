import numpy as np
import matplotlib.pyplot as plt
import random
# import soundfile as sf
import scipy
import os
import uuid
import numpy as np
import sys


keys = ['c4','d4','e4','f4','g4','a4','b4']
frq_dict = {'c4':261.63, 'd4':293.66, 'e4':329.63, 'f4':349.23, 'g4':392.00, 'a4':440, 'b4':493.88}
fs = 5000

save_dir = '06-train_note_recognition\\data\\1'

def generate_waveform(note_frq):
    time = 500
    time_dom = np.arange(0, time * (1 / fs), 1 / fs)

    noise_coefficient = np.random.choice(np.arange(0,20,0.01))
    noise = np.random.normal(0, np.random.choice(np.arange(0, 0.1, 0.001)), time)*noise_coefficient

    waveform = np.sin(2 * np.pi * note_frq * time_dom) * np.exp(-0.0004 * 2 * np.pi * note_frq * time_dom)
    waveform += np.sin(2 * 2 * np.pi * note_frq * time_dom) * np.exp(-0.0015 * 2 * np.pi * note_frq * time_dom) / 2
    waveform += np.sin(3 * 2 * np.pi * note_frq * time_dom) * np.exp(-0.0015 * 2 * np.pi * note_frq * time_dom) / 4
    waveform += np.sin(4 * 2 * np.pi * note_frq * time_dom) * np.exp(-0.0015 * 2 * np.pi * note_frq * time_dom) / 8
    waveform += np.sin(5 * 2 * np.pi * note_frq * time_dom) * np.exp(-0.0015 * 2 * np.pi * note_frq * time_dom) / 16
    waveform += np.sin(6 * 2 * np.pi * note_frq * time_dom) * np.exp(-0.0015 * 2 * np.pi * note_frq * time_dom) / 32
    waveform += waveform ** np.random.choice([1, 3, 5, 7, 9, 11, 13])

    waveform = waveform+noise

    # waveform = []
    # waveform = np.append(waveform, noise)
    # waveform = np.append(waveform, waveform1)
    waveform = waveform.astype('float32')

    return waveform

# def generate_noise():
#     time = 2000
#     noise = np.random.normal(0, np.random.choice(np.arange(0, 0.1, 0.001)), time)
#     return noise

fig1, axs1 = plt.subplots(7)
for i in range(7):
    note_frq = frq_dict[np.random.choice(keys)]
    for j in range(500):
        note_frq = frq_dict[keys[i]]
        waveform = generate_waveform(note_frq)

        scipy.io.wavfile.write(os.path.join(save_dir, keys[i], str(uuid.uuid1())) + '.wav', fs, waveform)

    axs1[i].plot(waveform)

plt.show()
