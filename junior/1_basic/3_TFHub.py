# tf.keras:一个 Tensorflow 中用于构建和训练模型的高级API
# TensorFlow Hub: 一个用于迁移学习的库和平台

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#%%
# 下载 IMDB 数据集 网络电影数据库（Internet Movie Database）的 IMDB 数据集（IMDB dataset）
# 将训练集分割成 60% 和 40%，从而最终我们将得到 15,000 个训练样本
# 10,000 个验证样本以及 25,000 个测试样本。
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

print(train_examples_batch)
print(train_labels_batch)

#%%
# 构建模型
'''
针对此示例我们将使用 TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1 的一种预训练文本嵌入（text embedding）模型 。

为了达到本教程的目的还有其他三种预训练模型可供测试：

google/tf2-preview/gnews-swivel-20dim-with-oov/1 ——类似 google/tf2-preview/gnews-swivel-20dim/1，但 2.5%的词汇转换为未登录词桶（OOV buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
google/tf2-preview/nnlm-en-dim50/1 ——一个拥有约 1M 词汇量且维度为 50 的更大的模型。
google/tf2-preview/nnlm-en-dim128/1 ——拥有约 1M 词汇量且维度为128的更大的模型。
'''
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

print(hub_layer(train_examples_batch[:3]))

#%%
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

# %%
# 损失函数与优化器
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              metrics=['accuracy']
              )
# logits: value before activation.
# from_logits=False: y_pred in [0,1]
# from_logits=True: y_pred=logits not in [0,1], more numerically stable.


#%%
# 训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20, 
                    validation_data=validation_data.batch(512),
                    verbose=1)


#%% 
# 评估模型
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print('%s: %.3f' % (name, value))

# model.metrics_names: ['loss', 'accuracy']
# results: [0.3190848760458888, 0.8586]





















