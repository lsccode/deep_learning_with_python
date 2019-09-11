from keras import layers
from keras import Input
from keras.models import Model
import numpy as np

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)

x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)

x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input,[age_prediction, income_prediction, gender_prediction])

# Listing 7.4 Compilation options of a multi-output model: multiple losses
# model.compile(optimizer='rmsprop',loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])

# Equivalent (possible only if you give names to the output layers)
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse',
#                     'income': 'categorical_crossentropy',
#                     'gender': 'binary_crossentropy'})


#age_targets, income_targets, and gender_targets are assumed to be Numpy arrays.

# Listing 7.5 Compilation options of a multi-output model: loss weighting
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])

#Equivalent (possible only if you give names to the output layers)
# model.compile(optimizer='rmsprop',
#               loss={'age': 'mse',
#                     'income': 'categorical_crossentropy',
#                     'gender': 'binary_crossentropy'},
#               loss_weights={'age': 0.25,
#                             'income': 1.,
#                             'gender': 10.})

# Listing 7.6 Feeding data to a multi-output model

#age_targets, income_targets, and gender_targets are assumed to be Numpy arrays.

num_samples = 100
max_length = 100
posts = np.random.randint(1, vocabulary_size,size=(num_samples, max_length))

min_age = 15
max_age = 60

age_targets = np.random.randint(min_age, max_age,size=(num_samples))
income_targets = np.random.randint(num_income_groups,size=(num_samples,num_income_groups))
gender_targets = np.random.randint(2,size=(num_samples))


model.fit(posts, [age_targets, income_targets, gender_targets],
          epochs=10, batch_size=10)
#
# #Equivalent (possible only if you give names to the output layers)
# model.fit(posts, {'age': age_targets,
#                   'income': income_targets,
#                   'gender': gender_targets},
#           epochs=10, batch_size=64)

