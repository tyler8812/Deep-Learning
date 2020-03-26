import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
# Y = W*X+B
Y = 0.5 * X + 2 +np.random.normal(0, 0.05, (200, ))

# plot data
# plt.scatter(X, Y)
# plt.show()

X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# build neural network from 1st layer to last layer
model = Sequential()
# 第一個layer
model.add(Dense(output_dim = 1, input_dim = 1))
# 第二個layerif needed，不用再定義input_dim因為為上一個的output_dim
# model.add(Dense(output_dim = 1))



# choose loss function, and optimizing method
# mse = mean square error
# sgd = stochastic gradient decent

model.compile(loss='mse', optimizer = 'sgd')


# training
print("Training~~~~~")
# iteration 301 times
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if(step % 100 == 0):
        print("train cost[%d]:" %(step), cost)



# testing
print("\nTesting~~~~~")
cost = model.evaluate(X_test, Y_test, batch_size=40)
print("test cost:",cost)
# 第一個layer
W, b = model.layers[0].get_weights()
print("Weights:", W, "\nbiases", b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

