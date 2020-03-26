import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


# downlosd the minist 
# Z shape (60000 28*28), y shape(10000, )
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)


# Another way to build your CNN
model = Sequential()

# Conv lyer1 output shape(32,28,28)
model.add(Convolution2D(
    nb_filter = 32,
    nb_row = 5,
    nb_col = 5,
    border_mode = 'same',   #padding method
    input_shape = (1, # channel
                28, 28), # height width 
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32,14,14)
model.add(MaxPooling2D(
    pool_size = (2, 2),
    strides = (2, 2),
    border_mode = 'same',   #padding method

))

# Conv layer2 output shape(64,14,14)

model.add(Convolution2D(64, 5, 5, border_mode = 'same'))
model.add(Activation('relu'))

# Pooling layer 2(max pooling) output shape (64,7,7)
model.add(MaxPooling2D(pool_size = (2, 2), border_mode = 'same'))

# Fully connected layer1 input shape(64*7*7) = 3136, output shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer2 input shape(10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define optimizer
adam = Adam(lr = 1e-4)

#  Add more metrics to get more results u want to see
model.compile(
    optimizer = adam,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
)

# training
print("Training~~~~~")
# Another way to train
model.fit(X_train, Y_train, epochs = 1, batch_size = 32)


# testing
print("\nTesting~~~~~")
loss, accuracy = model.evaluate(X_test, Y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)