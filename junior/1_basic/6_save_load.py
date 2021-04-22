import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

#%%
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
print(train_images.shape)
train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0
print(train_images.shape)
#%%
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
        ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()
#%%
# 在训练期间保存模型（以 checkpoints 形式保存）
# Checkpoint 回调用法
# tf.keras.callbacks.ModelCheckpoint 允许在训练的过程中和结束时回调保存的模型。
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

print(checkpoint_path, checkpoint_dir)

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# 使用新的回调训练模型
model.fit(train_images, 
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback]) # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）
# 是防止过时使用，可以忽略。

#%%
# 仅恢复模型的权重时，必须具有与原始模型具有相同网络结构的模型。
# 由于模型具有相同的结构，您可以共享权重，尽管它是模型的不同实例。

# 创建一个基本模型实例
model = create_model()

# 评估模型
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# 加载权重
model.load_weights(checkpoint_path)

# 重新评估模型
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%
# checkpoint 回调选项
# 回调提供了几个选项，为 checkpoint 提供唯一名称并调整 checkpoint 频率。

# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)
# 创建一个新的模型实例
model = create_model()

# 使用 `checkpoint_path` 格式保存权重
model.save_weights(checkpoint_path.format(epoch=0))

# 使用新的回调训练模型
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

#%%
# 如果要进行测试，请重置模型并加载最新的 checkpoint ：

# 获取目录下最新的模型：tf.train.latest_checkpoint
lastest = tf.train.latest_checkpoint(checkpoint_dir)
print(lastest) # 'training_2/cp-0050.ckpt'
# 注意: 默认的 tensorflow 格式仅保存最近的5个 checkpoint。

# 创建一个新的模型实例
model = create_model()

# 加载以前保存的权重
model.load_weights(lastest)

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%
# 这些文件是什么？
# =============================================================================
# 上述代码将权重存储到 checkpoint—— 格式化文件的集合中，这些文件仅包含二进制格式的训练权重。 Checkpoints 包含：
#     一个或多个包含模型权重的分片。
#     索引文件，指示哪些权重存储在哪个分片中。
# =============================================================================

# 手动保存权重
# 使用 Model.save_weights 方法手动保存它们同样简单。
# 默认情况下， tf.keras 和 save_weights 特别使用:
#     TensorFlow checkpoints 格式 .ckpt 扩展名和 ( 保存在 HDF5 扩展名为 .h5 保存并序列化模型 )

# 保存权重
model.save_weights('./checkpoints/my_checkpoint')

# 创建模型实例
model= create_model()

# 恢复权重
model.load_weights('./checkpoints/my_checkpoint')

# 评估模型
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%
# 保存整个模型
# =============================================================================
# 调用 model.save 将保存模型的结构，权重和训练配置保存在单个文件/文件夹中。
# 导出模型，以便在不访问原始 Python 代码*的情况下使用它。
# 因为优化器状态（optimizer-state）已经恢复，您可以从中断的位置恢复训练。

# 整个模型可以以两种不同的文件格式（SavedModel 和 HDF5）进行保存。
# 需要注意的是 TensorFlow 的 SavedModel 格式是 TF2.x. 中的默认文件格式。
# 但是，模型仍可以以 HDF5 格式保存。

# 保存完整模型会非常有用——您可以在 TensorFlow.js（Saved Model, HDF5）加载它们，
# 然后在 web 浏览器中训练和运行它们，或者使用 TensorFlow Lite 将它们转换为在移动设备上运行（Saved Model, HDF5）

# *自定义对象（例如，子类化模型或层）在保存和加载时需要特别注意。
# 请参阅下面的保存自定义对象部分*自定义对象（例如，子类化模型或层）在保存和加载时需要特别注意。
# =============================================================================


# SavedModel 格式
# SavedModel 格式是序列化模型的另一种方法。
# 以这种格式保存的模型，可以使用 tf.keras.models.load_model 还原，并且模型与 TensorFlow Serving 兼容。

model = create_model()
model.fit(train_images, train_labels, epochs=5)

model.save('saved_model')

#%%
# 从保存的模型重新加载新的 Keras 模型
new_model= tf.keras.models.load_model('saved_model')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('restored model, accuracy:{:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)

#%%
# HDF5 格式
# Keras使用 HDF5 标准提供了一种基本的保存格式。

model = create_model()
model.fit(train_images, train_labels, epochs=5)

# 将整个模型保存为 HDF5 文件。
# '.h5' 扩展名指示应将模型保存到 HDF5。
model.save('my_model.h5')

#%%
# 重新创建完全相同的模型，包括其权重和优化程序
new_model = tf.keras.models.load_model('my_model.h5')

new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('restored model, accuracy:{:5.2f}%'.format(100*acc))

# Keras 无法保存 v1.x 优化器（来自 tf.compat.v1.train），因为它们与检查点不兼容。
# 对于 v1.x 优化器，您需要在加载-失去优化器的状态后，重新编译模型。

#%%
# =============================================================================
# 保存自定义对象
# 如果使用的是 SavedModel 格式，则可以跳过此部分。HDF5 和 SavedModel 之间的主要区别在于，HDF5 使用对象配置保存模型结构，而 SavedModel 保存执行图。
# 因此，SavedModel 能够保存自定义对象，例如子类化模型和自定义层，而无需原始代码。
# 
# 要将自定义对象保存到 HDF5，必须执行以下操作:
# 
# 在对象中定义一个 get_config 方法，以及可选的 from_config 类方法。
# get_config(self) 返回重新创建对象所需的参数的 JSON 可序列化字典。
# from_config(cls, config) 使用从 get_config 返回的 config 来创建一个新对象。
# 默认情况下，此函数将使用 config 作为初始化 kwargs（return cls(**config)）。
# 加载模型时，将对象传递给 custom_objects 参数。参数必须是将字符串类名称映射到 Python 类的字典。
# 例如，tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})
# =============================================================================



















































