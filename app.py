from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from preprocess_data import preprocess 

features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print(accuracy_score(pred, labels_test))