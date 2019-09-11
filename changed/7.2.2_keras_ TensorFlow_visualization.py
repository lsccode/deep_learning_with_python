# 代码清单 7-7　使用了 TensorBoard 的文本分类模型

import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

# 作为特征的单词个数
max_features = 100 # 2000

#在这么多单词之后截断文本（这些单词都属于前 max_features个最常见的单词）
max_len = 50

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128,
                           input_length=max_len,
                           name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

from keras.utils import plot_model

plot_model(model, to_file='model.png')
plot_model(model, show_shapes=True, to_file='model-more.png')

# $ mkdir my_log_dir

# 代码清单 7-9　使用一个 TensorBoard 回调函数来训练模型

# x_train = x_train.astype('float32')
# y_train = y_train.astype('float32')

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=1,
        embeddings_freq=1,
        embeddings_data=x_train[:100].astype("float32")
    )
]
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)

# $ tensorboard --logdir=my_log_dir