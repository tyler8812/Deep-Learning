import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# downlosd the minist 
# Z shape (60000 28*28), y shape(10000, )
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1)/255. # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255. # normalize
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

# Another way to build your neural network
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])



# Another way to define optimizer

rmsprop = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.0)

# Add more metrics to get more results u want to see
model.compile(
    optimizer = rmsprop,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
)


# training
print("Training~~~~~")
# Another way to train
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)


# testing
print("\nTesting~~~~~")
loss, accuracy = model.evaluate(X_test, Y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)