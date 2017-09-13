from Bearclass import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report

bear = Bear()
data1, target = bear.data, bear.target
lb = LabelEncoder()
target = lb.fit_transform(target)
print(target, ...)
X_train, X_test, y_train, y_test = train_test_split(
    data1, target, test_size=0.333)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svc1 = svm.SVC()
svc1.fit(X_train,y_train)
y_pred = svc1.predict(X_train)
print(classification_report(y_train,y_pred))

data2 = scio.loadmat('wavedata.mat')['wavedata']
print(sum(data2[0]))
X_train, X_test, y_train, y_test = train_test_split(
    data2, target, test_size=0.333)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

svc2 = svm.SVC()
svc2.fit(X_train,y_train)
y_pred = svc2.predict(X_train)
print(classification_report(y_train,y_pred))

# 使用原始数据时，经过归一化处理效果更好 76% 48%
# 使用小波能量特征时，最后不经过归一化，效果更好 88% 66%