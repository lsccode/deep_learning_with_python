# keras.callbacks.ModelCheckpoint
# keras.callbacks.EarlyStopping
# keras.callbacks.LearningRateScheduler
# keras.callbacks.ReduceLROnPlateau
# keras.callbacks.CSVLogger

import keras
from keras.models import Model

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='acc',
        patience=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='my_model.h5',
        monitor='val_loss',
        save_best_only=True,
    )
]

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# model.fit(x, y,
#           epochs=10,
#           batch_size=32,
#           callbacks=callbacks_list,
#           validation_data=(x_val, y_val))


#ReduceLROnPlateau 回调函数

# callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,)]
# model.fit(x, y,epochs=10,batch_size=32,callbacks=callbacks_list,validation_data=(x_val, y_val))
#
import keras
import numpy as np


class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):

        #在训练之前由父模型调用，告诉回调函数是哪个模型在调用它
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]
        self.activations_model = keras.models.Model(model.input,
                                                    layer_outputs) #模型实例，返回每层的激活

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')

        # 获取验证数据的第一个输入样本
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)

        #将数组保存到硬盘
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()