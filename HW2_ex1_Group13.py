import tensorflow as tf
import argparse 
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import numpy as np
import pandas as pd
import zlib
import os
import time

class WindowGenerator:
    def __init__(self, input_width, labels_opts, mean, std):

        self.input_width = input_width
        self.label_option = labels_opts

        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])  
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features): 
        inputs = features[:, :-6, :] 
        labels = features[:, -6:, :] 
        num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.input_width, num_labels])
        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)  
        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)
        return inputs, labels

    def make_dataset(self, data,
                     train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width + 6, 
            sequence_stride=1,
            batch_size=32)
        ds = ds.map(self.preprocess) 
        ds = ds.cache() 
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds
    
class multi_MAE(tf.keras.metrics.Metric):
    def __init__(self, name='multi_MAE', **kwargs):
        super(multi_MAE, self).__init__(name=name, **kwargs)
        self.local_mae = self.add_weight(name='local_mae', shape=(2,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.abs(y_true - y_pred)
        values = tf.reduce_mean(values, axis = 1)
        values = tf.reduce_mean(values, axis = 0)
        self.local_mae.assign_add(values)
        self.count.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self.local_mae, self.count)

    def reset_states(self):
        self.local_mae.assign(tf.zeros_like(self.local_mae))
        self.count.assign(tf.zeros_like(self.count))

class Model():
    def __init__(self,alpha):
        
        self.model = keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(6, 2)),
            tf.keras.layers.Dense(units=32*alpha, activation='relu'),
            tf.keras.layers.Dense(12,kernel_initializer=tf.initializers.glorot_normal()),
            tf.keras.layers.Reshape([6, 2])
        ])
    
    def model(self):
        return self.model
    
    def compile_model(self):
        self.model.compile(optimizer = tf.optimizers.Adam(),
                    loss = tf.keras.losses.MSE,
                    metrics = multi_MAE())
        self.model.summary()
        
    def train_model(self, epochs, train,val, input_shape): 
        self.model.build(input_shape)
        self.model.fit(train, epochs=epochs, validation_data=val_ds, callbacks = self.callbacks)
        
    def set_pruning_schedule(self, init_spars, fin_spars, version):
        pruning_params = {'pruning_schedule':
                          tfmot.sparsity.keras.PolynomialDecay(
                          initial_sparsity=init_spars,
                          final_sparsity=fin_spars,
                          begin_step=len(train_ds)*5,
                          end_step=len(train_ds)*15)}
        
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        self.model = prune_low_magnitude(self.model, **pruning_params)

        self.callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        
        return self.model

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, choices = ['a','b'])
args = parser.parse_args()

version = args.version

input_width = 6 
labels_opts = 2 

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# download the data using keras from the link
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

# recover the path of the csv
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_index = [2, 5]
columns = df.columns[column_index]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n * 0.7)]
test_data = data[int(n * 0.7):int(n * 0.9)]
val_data = data[int(n * 0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

generator = WindowGenerator(input_width, labels_opts, mean, std)

train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

for x,y in train_ds.take(1):
    input_shape = x.shape
    break

if version =='a':
    alpha = 1
    init_sparsity = 0.5
    final_sparsity = 0.85
else:
    alpha = 0.75
    init_sparsity = 0.6
    final_sparsity = 0.9
    
tf.keras.backend.clear_session()

MLP = Model(alpha)
MLP.set_pruning_schedule(init_sparsity,final_sparsity,version) #initial and final sparsity
MLP.compile_model()

history = MLP.train_model(20, train_ds, val_ds, input_shape)
test_loss, test_mae = MLP.model.evaluate(test_ds)
print(test_mae)


prun_aware_model = tfmot.sparsity.keras.strip_pruning(MLP.model)
converter = tf.lite.TFLiteConverter.from_keras_model(prun_aware_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

filename = f'Group13_th_{version}.tflite.zlib'

with open(filename, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)
    
time.sleep(0.5)


print(f'{os.path.getsize(filename)/1000:.2f} Kb')

with open(filename,'rb') as fp:
    tfile = fp.read()
    tfile = zlib.decompress(tfile)

test_ds = test_ds.unbatch().batch(1)

interpreter = tf.lite.Interpreter(model_content = tfile)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
mae = 0
count = 0

for x,y_true in test_ds:
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    y_pred = y_pred.squeeze()
    y_true = y_true.numpy().squeeze()
    mae += np.abs(y_true-y_pred)
    avg_mae = mae.mean(axis=0)
    count +=1
avg_mae /= count

print(avg_mae)

