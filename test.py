from Bearclass import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

bear = Bear()
data,target=bear.data,bear.target
lb = LabelBinarizer()
target = lb.fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.333)
print(y_test)
sort = np.argsort(y_test.argmax(axis=1))
y_p = y_test[sort]
x_p = X_test[sort]
print(y_p[:,:])