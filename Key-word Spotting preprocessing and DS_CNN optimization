
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import numpy as np
import argparse
import pandas as pd
import zlib
import os
import time



class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step, num_mel_bins=None,
                 lower_frequency=None, upper_frequency=None, num_coefficients=None,
                 mfcc=False):

        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients

        if mfcc is True:
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio),dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])
        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio,
                              frame_length=self.frame_length,
                              frame_step=self.frame_step,
                              fft_length=self.frame_length)
        spectrogram = tf.abs(stft)
        return spectrogram

    def get_mfcc(self, spectrogram):
        
        num_spectrogram_bins = spectrogram.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                                            self.num_mel_bins,
                                            num_spectrogram_bins,
                                            self.sampling_rate,  
                                            self.lower_frequency,
                                            self.upper_frequency
                                        )
        mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix,1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :self.num_coefficients]
        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])
        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfcc(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess,num_parallel_calls=4) # num of threads
        ds = ds.batch(32)
        ds = ds.cache()

        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds



class Model():
    def __init__(self,alpha, input_shape, kernel_size, strides):
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64*alpha, kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=input_shape[1:]),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size = [3,3], strides = [1,1], use_bias = False),
            tf.keras.layers.Conv2D(filters=64*alpha, kernel_size=[1, 1],strides=[1, 1], use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],use_bias=False),
            tf.keras.layers.Conv2D(filters=64*alpha, kernel_size=kernel_size, strides=[1, 1], use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=n_classes)]
            )
    
    def model(self):
        return self.model
    
    def compile_model(self, learning_rate):
        self.model.compile(optimizer = tf.optimizers.Adam(learning_rate = learning_rate),
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics = tf.keras.metrics.SparseCategoricalAccuracy())
        self.model.summary()
        
    def train_model(self, epochs, train, val, version): 
        
        self.callbacks = [tf.keras.callbacks.ModelCheckpoint(f'cp_bestmodel_2{version}',
                monitor="val_sparse_categorical_accuracy",
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch"),
                tf.keras.callbacks.LearningRateScheduler(scheduler)]
        
        self.model.fit(train, epochs=epochs, validation_data=val_ds, callbacks = self.callbacks)   
        
def scheduler(epoch,lr):
        if epoch%10!=0:
            return lr
        else:
            return lr * 0.05



parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True,
        help='model version', choices = ['a','b','c'])

args = parser.parse_args()

version = args.version



seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
zip_path = tf.keras.utils.get_file(
     origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
     fname='mini_speech_commands.zip',
     extract=True,
     cache_dir='.',
     cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')



d = {}
for file in os.listdir():
    if file.endswith('.txt'):
        content = []
        with open(file) as split:
            for line in split:
                content.append(line.replace('\n',''))
        d[file.split('.')[0]] = content



train_data = tf.convert_to_tensor(d['kws_train_split'])
val_data = tf.convert_to_tensor(d['kws_val_split'])
test_data =  tf.convert_to_tensor(d['kws_test_split'])

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS = LABELS[LABELS != 'README.md']
n_classes = len(LABELS)


MFCC = True
rate = 16000

if version == 'a':
    alpha = 1
    frame_length = int(rate * 40e-3)
    frame_step = int(rate * 15e-3) #to increase the accuracy we enlarge the overlapping 
    num_mel_bins = 40
    strides = [2,1]
    kernel_size = [1,1]
    
    
elif version == 'b':
    alpha = 0.7
    frame_length = int(rate * 40e-3)
    frame_step = int(rate * 15e-3) 
    num_mel_bins = 40
    strides = [2,1]
    kernel_size = [3,3]

elif version == 'c':
    alpha = 0.85
    frame_length = int(rate * 30e-3) #la frame length piu ampia riduce l'accuracy
    frame_step = int(rate * 25e-3) 
    num_mel_bins = 40
    strides = [2,1]
    kernel_size = [3,3]

    
signal_generator = SignalGenerator(LABELS, rate, frame_length, frame_step, num_mel_bins,
                                       lower_frequency=20, upper_frequency=4000, num_coefficients=10,
                                       mfcc=MFCC)

train_ds = signal_generator.make_dataset(train_data, True)
test_ds = signal_generator.make_dataset(test_data, False)
val_ds = signal_generator.make_dataset(val_data, False)


for x,y in train_ds.take(1):
    input_shape = x.shape
    break

tf.keras.backend.clear_session()

DS_CNN = Model(alpha,input_shape, kernel_size, strides)
DS_CNN.compile_model(learning_rate = 0.05)


history = DS_CNN.train_model(30, train_ds, val_ds, version)


converter = tf.lite.TFLiteConverter.from_saved_model(f'cp_bestmodel_2{version}')

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

filename = f'Group13_kws_{version}.tflite'
with open(filename, 'wb') as fp:
    fp.write(tflite_model)
    
time.sleep(0.5)


print(f'TFLite Model Size: {os.path.getsize(filename)/1000:.2f} Kb')

test_ds = test_ds.unbatch().batch(1)

interpreter = tf.lite.Interpreter(model_path = filename)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
accuracy = 0
count = 0
for x,y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_pred = np.argmax(y_pred)
    y_true = y_true.numpy().squeeze()
    accuracy += y_pred == y_true
    count +=1
accuracy /= count
print('Accuracy {:.2f}'.format(accuracy*100))
print()

