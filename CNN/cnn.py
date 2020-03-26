import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import matplotlib.pyplot as plt


batch_size = 64
num_classes = 10
epochs = 30

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', input_shape=(32,32,3)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu'))
    model.add(Dense(84, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=0.001), 
                  metrics=['accuracy'])
    return model

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#show train img
print(x_train[0].shape) #32*32*3
plt.imshow(x_train[0])
plt.savefig('train.png')
plt.clf()
#normalize img
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#set model
model = build_model()
#model = load_model('lenet.h5')
model.summary()
#start training
history = model.fit(x_train[:40000], y_train[:40000],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_train[40000:], y_train[40000:]),
                        shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predict = model.predict(x_test)
model.save('lenet.h5')

# print(predict.shape)
# print(predict[0])

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('acc.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')


