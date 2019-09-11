from keras import layers
from keras.models import Model
import numpy as np
from keras import Input

x= Input(shape=(300, 300, 3))
# 每个分支都有相同的步幅值（2），这对于保持所有分支输出具有相同尺寸是很有必要的，这样你才能将它们连接在一起
branch_a = layers.Conv2D(128, 1,activation='relu', strides=2)(x)

# 在这个分支中，空间卷积层用到了步幅
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

# 在这个分支中，平均池化层用到了步幅
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)


branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

# 将分支输出连接在一起，得到模块输出
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

model = Model(x,output)
model.summary()

from keras import layers

x = np.random.randint(1, 10000,size=(10, 3, 300, 300))
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)

y = layers.add([y, x])
from keras import layers

x = np.random.randint(1, 10000,size=(10, 3, 300, 300))
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)

y = layers.add([y, residual])