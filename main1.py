from Bearclass import *
from lstmclass import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import random
import numpy as np

n_batch = 100
n_step = 200
n_input = 10
n_output = 10
n_cell = 100
lr = 0.006
n_train = 8000
bear = Bear()
bear.savedata()
data, target = bear.data, bear.target
lb = LabelBinarizer()
target = lb.fit_transform(target)
# print(target, ...)
# print(lb.inverse_transform(target))
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.333)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
total_train_batch, total_test_batch = X_train.shape[0], X_test.shape[0]
print(total_train_batch, ...)
print(X_train.shape, y_test.shape, ...)

lstm = LSTM(n_batch, n_step, n_input, n_output, n_cell)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(n_train):
        term = random.sample(range(total_train_batch), n_batch)
        sess.run(lstm.train_op, {lstm.x: X_train[term][
                 :, :, np.newaxis].reshape(n_batch, n_step, n_input), lstm.y: y_train[term]})
        if i % 200 == 0:
            acc = np.ones((int(total_test_batch / n_batch), n_batch))
            for j in range(int(total_test_batch / n_batch)):
                acc[j] = (sess.run(lstm.acc,
                                   {lstm.x: X_test[j * n_batch: (j + 1) * n_batch][
                                       :, :, np.newaxis].reshape(n_batch, n_step, n_input),
                                    lstm.y: y_test[j * n_batch: (j + 1) * n_batch]}))
            print(i, acc.mean(), ...)
