import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
model.summary()
#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)


# =============================================================================
# tf.keras.layers.Flatten(
#     data_format=None, **kwargs
# )
# =============================================================================

# =============================================================================
# tf.layers.Dense(
#     units, activation=None, use_bias=True, kernel_initializer=None,
#     bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
#     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#     bias_constraint=None, trainable=True, name=None, **kwargs
# )
# units: Integer or Long, dimensionality of the output space.
# activation	: Activation function (callable). Set it to None to maintain a linear activation.
# use_bias:	Boolean, whether the layer uses a bias.
# =============================================================================
