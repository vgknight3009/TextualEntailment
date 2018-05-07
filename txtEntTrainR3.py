import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

train = pd.read_csv('trainRTE3.csv')
test = pd.read_csv('testRTE3.csv')


def modifyDF(dataF):

	dataF['F1'] = (dataF['F1'] >=0.70).astype(int)		# fine .65 or 0.70
	dataF['F3'] = (dataF['F3'] >=0.50).astype(int)		# fine 0.50
	dataF['F6'] = (dataF['F6'] >=0.45).astype(int)		# fine 0.45, 0.50
	dataF['F14'] = (dataF['F14'] >=0.50).astype(int)	## fine 0.50

	dataF = dataF.drop(['F21'],axis=1)		## fine
	#dataF = dataF.drop(['F20'],axis=1)
	dataF = dataF.drop(['F19'],axis=1)
	dataF = dataF.drop(['F18'],axis=1)
	return dataF
	

train = modifyDF(train)
test = modifyDF(test)

X_train = train.drop(['Entailment'],axis=1)
Y_train = train['Entailment']
#print(X_train.shape)

X_test = test.drop(['Entailment'],axis=1)
Y_test = test['Entailment']
#print(X_test.shape)

print('\nLogistic')
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)

Y_pred = logreg.predict(X_test)
print('Accuracy: ',round(accuracy_score(Y_test, Y_pred),3))
print('Precision: ',round(precision_score(Y_test, Y_pred, average="macro"),3))
print('Recall: ',round(recall_score(Y_test, Y_pred, average="macro"),3))
print('F1_Score: ',round(f1_score(Y_test, Y_pred, average="macro"),3))

print(pd.crosstab(Y_test, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


coeff_df = pd.DataFrame(X_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
#print(coeff_df.sort_values(by='Correlation', ascending=False))

#print('\nPerceptron')
#perceptron = Perceptron()
#perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(X_test)
#print('Accuracy: ',round(accuracy_score(Y_test, Y_pred),2))
#print('Precision: ',round(precision_score(Y_test, Y_pred, average="macro"),2))
#print('Recall: ',round(recall_score(Y_test, Y_pred, average="macro"),2))
#print('F1_Score: ',round(f1_score(Y_test, Y_pred, average="macro"),2))

# print('\nNaive')
# gaussian = GaussianNB()
# gaussian.fit(X_train,Y_train)
# Y_pred = gaussian.predict(X_test)
# print('Accuracy: ',round(accuracy_score(Y_test, Y_pred),2))
# print('Precision: ',round(precision_score(Y_test, Y_pred, average="macro"),2))
# print('Recall: ',round(recall_score(Y_test, Y_pred, average="macro"),2))
# print('F1_Score: ',round(f1_score(Y_test, Y_pred, average="macro"),2))

# print('\nRandom')
# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# print('Accuracy: ',round(accuracy_score(Y_test, Y_pred),4))
# print('Precision: ',round(precision_score(Y_test, Y_pred, average="macro"),4))
# print('Recall: ',round(recall_score(Y_test, Y_pred, average="macro"),4))
# print('F1_Score: ',round(f1_score(Y_test, Y_pred, average="macro"),4))

print('\nSVM')
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print('Accuracy: ',round(accuracy_score(Y_test, Y_pred),3))
print('Precision: ',round(precision_score(Y_test, Y_pred, average="macro"),3))
print('Recall: ',round(recall_score(Y_test, Y_pred, average="macro"),3))
print('F1_Score: ',round(f1_score(Y_test, Y_pred, average="macro"),3))

print(pd.crosstab(Y_test, Y_pred, rownames=['True'], colnames=['Predicted'], margins=True))