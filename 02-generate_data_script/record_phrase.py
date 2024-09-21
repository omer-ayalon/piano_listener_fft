import sounddevice as sd
import scipy
import uuid
import os


path_to_save = '01-data\\seperating_demos'

fs = 5000  # Sample rate
seconds = 15  # Duration of recording

for i in range(1):
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, device=2)
    print('start')
    sd.wait()  # Wait until recording is finished
    print('stop')
    scipy.io.wavfile.write(os.path.join(path_to_save,str(uuid.uuid1()))+'.wav', fs, recording)  # Save as WAV file