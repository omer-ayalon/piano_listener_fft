from utility import Functions
import scipy
import matplotlib.pyplot as plt
import numpy as np


# Load File
fs, waveform = scipy.io.wavfile.read('01-data\\seperating_demos\\output_1.wav')

# Split Notes
idx_spliter, metre, bmp = Functions.Note_Splitter(waveform, fs)
print('The Metre Of The Clip Is: ', metre)
print('The BPM Is: ', np.round(bmp))

# Note Identifier
notes_frq = Functions.Note_Identifier(waveform, fs, idx_spliter)
# print(notes_frq)

# Translate Frequency To Note
notes_name = Functions.Frq_To_Note(notes_frq)
print('The Notes Names Are: ', notes_name)

# Plotting
time_domain = np.arange(0, (1 / fs) * np.size(waveform), 1 / fs)
plt.plot(time_domain, waveform)
for idx in idx_spliter:
    plt.axvline(idx/fs, color='r')
plt.grid()
plt.title('Sound Waveform With Splitters')
plt.xlabel('Time [s]')
plt.ylabel('Magnitude')
plt.show()
