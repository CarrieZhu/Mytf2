import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np
import pathlib
import shutil
import tempfile
#%%
logdir = pathlib.Path(tempfile.mkdtemp())/'tensorboard_logs'
shutil.rmtree(logdir, ignore_errors=True)
#%%
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

#%%
FEATURES = 28

ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type='GZIP')
#%%
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    print(features, label)
    return features, label

packed_ds = ds.batch(10000).map(pack_row).unbatch()
#%%
# packed_ds.batch(1000).take(2): deal packed_ds to batches which has 1000 samples in one batch and then take 2 batches
for features,label in packed_ds.batch(1000).take(3):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)
#%%
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE #20

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

for line in train_ds.take(1):
    print(type(line))
    print('------------------')

# =============================================================================
# # dataset.shuffle(n):打乱数据集，n越大越乱
# # dataset.repeat(n)：使用数据集n次（重复n-1次）
# # dataset.batch(N)：分批，batchsize=N
# # dataset.unbatch()：解分批
# # dataset.skip(n)：跳过前n个
# # dataset.cache(): use the Dataset.cache method to ensure that the loader doesn't need to re-read the data from the file on each epoch
# 
# #                  When caching to a file, the cached data will persist across runs. Even the first iteration through the data will read from the cache file. 
# #                  Changing the input pipeline(include .cache()) before the call to .cache() will have no effect until the cache file is removed or the filename is changed.
# 
# 
# dataset = tf.data.Dataset.from_tensor_slices({
#     "feat":np.array([[1., 0.], [2., 0.], [3. ,0.]]),
#     "label":np.random.random(size=(3,3))})
# 
# dataset = dataset.skip(1).batch(1).cache()
# for line in dataset:
#     print(line['feat'],"****",line['label'])
#     print('------------------')
# =============================================================================


#%%
# Training procedure
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False)

# =============================================================================
# tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None)

# staircase=False:
# def decayed_learning_rate(step):
#   return initial_learning_rate / (1 + decay_rate * step / decay_step)
# =============================================================================
def get_optimizers():
    return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize=(8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('learninf Rate')

#%%
def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name),
        ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizers()
        
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(
                          from_logits=True, name='binary_crossentropy'),
                      'accuracy'])
    model.summary()
    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH, #  When passing an infinitely repeating dataset, you must specify the `steps_per_epoch` argument.
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)
    return history
        

# =============================================================================
# 'input_shape=':
# Build the model first by :1 or 2 or 3
#     1、calling `build()` ,
#     2、calling `fit()` with some data,
#     3、specify an `input_shape` argument in the first layer(s) for automatic build.
# =============================================================================

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
    ])

size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')


plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

#%%
small_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
    ])
size_histories['small'] = compile_and_fit(small_model, 'sizes/Small')

medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])
size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")

#%%
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")


#%%
# View in TensorBoard
print(logdir)
display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px")
# tensorboard --logdir C:\Users\zhuky2\AppData\Local\Temp\tmp0k9_71c_\temsorboard_logs/sizes


#%%
# Strategies to prevent overfitting

shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

# Add weight regularization
# =============================================================================
# In tf.keras.layers.Dense():
# kernel_regularizer = tensorflow.keras.regularizers.l2(0.001):
# l2(0.001) means that every coefficient in the weight matrix of the layer will add 0.001 * weight_coefficient_value**2 to the total loss of the network.
# =============================================================================

l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001), 
                 input_shape=(FEATURES,),),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(8, activation='elu',),
    layers.Dense(1)
    ])
regularizer_histories['l2'] = compile_and_fit(l2_model, 'regularizers/l2')

#%%
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
#%%
# regularization loss
# =============================================================================
# model.evaluate(test): Returns the loss value & metrics values for the model in test mode.
# model(test): Returns the output of the model
# =============================================================================

result = l2_model(features)
# loss = BinaryCrossentropy + regularization loss
# BinaryCrossentropy: from compile, decided by input data
# regularization loss: sum of regularization loss of every layer weights, dependent only on  model and weights
regularization_loss = tf.add_n(l2_model.losses) 
# =============================================================================
# Sequential (inherits from)-> Layer, Model -> Module
# Method of tf.Module:
#     Module(x)
# Attributes of tf.keras.layers.Layer:
#     name, compute_dtype, dtype_policy, non_trainable_weights, input_spec, activity_regularizer, supports_masking...
#     l2_model.losses: list of losses in every regularized layer
#     l2_model.metrics
#     l2_model.output
#     l2_model.input
#     l2_model.weights
#     ...
# Attributes of tf.keras.Model:
#     metrics_names: Returns the model's display labels for all outputs.
#     Note: metrics_names are available only after a keras.Model has been trained/evaluated on actual data.
# =============================================================================

#%%
# Add dropout
# =============================================================================
# 1\Dropout, applied to a layer, consists of randomly "dropping out" (i.e. set to zero) a number of output features of the layer during training.
# 2\The "dropout rate": [0.2, 0.5]. 
# 3\At test time, no units are dropped out, and instead the layer's output values are scaled down by a factor equal to the dropout rate, so as to balance for the fact that more units are active than at training time.
# 4\tf.keras.layers.Dropout layer gets applied to the output of layer right before.
# =============================================================================

dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
    ])
regularizer_histories['dropout'] = compile_and_fit(dropout_model, 'regularization/dropout')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])


#%%
# Combined L2 + dropout
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")