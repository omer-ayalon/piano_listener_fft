from utility import Functions
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os


# Path To Dataset
path_to_dataset = '06-train_note_recognition\\data\\1'
# Notes
notes = ['a4','b4','c4','d4','e4','f4','g4']
# Load Dataset
data,labels,fs = Functions.Load(path_to_dataset)

# Normal Data To 0-1
data = np.divide(np.subtract(data, np.min(data)), np.subtract(np.max(data), np.min(data)))

# import scipy
# import matplotlib.pyplot as plt
# f, t, Sxx = scipy.signal.spectrogram(data[0], fs=fs, nperseg=500, nfft=1024)
# # Sxx = np.squeeze(Sxx)
# print(Sxx.shape)
# # plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.plot(Sxx)
# plt.show()

# Make Spectograms
spectrograms = np.empty([data.shape[0],513,1])
for i,wave in enumerate(data):
    spectrograms[i] = Functions.make_spectogram(wave, fs)

# Split Dataset To Train And Test
# spectrograms = np.expand_dims(spectrograms,axis=3)
spectrograms_train, spectrograms_test, labels_train, labels_test = train_test_split(spectrograms, labels, test_size=0.4, random_state=42)
spectrograms_test, spectrograms_val, labels_test, labels_val = train_test_split(spectrograms_test, labels_test, test_size=0.5, random_state=42)
print(spectrograms_train.shape, spectrograms_test.shape, spectrograms_val.shape)
print(labels_train.shape, labels_test.shape, labels_val.shape)

# Model Structure
input_shape = (513,1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    # Downsample the input.
    # tf.keras.layers.Resizing(32,32),
    # Normalize.
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(notes), activation='softmax')
])
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'],
)

# Fit Model
epochs = 100
H = model.fit(spectrograms_train, labels_train, validation_data=(spectrograms_val, labels_val), batch_size=32, epochs=epochs, verbose=1)

save_dest = os.path.join('06-train_note_recognition', 'models', '1_500_500')
model.save(save_dest+'.hdf5')

print('[INFO] Evaluating newtwork...')
prediction = model.predict(spectrograms_test, batch_size=32)
print(labels_test.shape, prediction.shape)
print(classification_report(labels_test.argmax(axis=1), prediction.argmax(axis=1), target_names=notes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(save_dest+'.png')
plt.show()
