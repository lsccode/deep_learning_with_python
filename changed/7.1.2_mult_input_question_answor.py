from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')
# 将输入嵌入长度为 64 的向量
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
#利用 LSTM 将向量编码为单个向量
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,),dtype='int32',name='question')
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question],axis=-1)

answer = layers.Dense(answer_vocabulary_size,activation='softmax')(concatenated)
model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

# Feeding data to a multi-input model
import numpy as np
import keras

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size,
                         size=(num_samples, max_length))

question = np.random.randint(1, question_vocabulary_size,size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
answers = keras.utils.to_categorical(answers, answer_vocabulary_size)

model.fit([text, question], answers, epochs=10, batch_size=128)

model.fit({'text': text, 'question': question}, answers,epochs=10, batch_size=128)