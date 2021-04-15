import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

#%%
# 下载 IMDB 数据集
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))
#%%
# 探索数据
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(train_data[0], decode_review(train_data[0]))

#%%
# 准备数据
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',#在序列的起始还是结尾补（默认为pre）
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=256)
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])
#%%
# 构建模型
vocab_sie = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_sie, 16))

model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# =============================================================================
# keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length)
# vocab_size:字典大小
# embedding_dim:本层的输出大小，也就是生成的embedding的维数
# input_length:输入数据的维数，因为输入数据会做padding处理，所以一般是定义的max_length

# 定义矩阵(vocab_size * embedding_dim), 输出(batch_size, max_length, embedding_dim), 将词表表示的句子转化为embedding
# batch_size*max_length*embedding_dim->batch_size*embedding_dim
# =============================================================================
#%%
# 损失函数与优化器
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
# 创建一个验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train= train_labels[10000:]

#%%
# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#%%
# 评估模型
results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

#%%
# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表

history_dict = history.history
print(history_dict.keys()) #dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

#%%
import matplotlib.pyplot as plt
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validayion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
