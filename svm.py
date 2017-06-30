from Bearclass import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report

bear = Bear()
data, target = bear.data, bear.target
lb = LabelEncoder()
target = lb.fit_transform(target)
print(target, ...)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.333)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svc = svm.SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_train)
print(classification_report(y_train,y_pred))