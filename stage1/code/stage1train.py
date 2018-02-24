%matplotlib inline
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f = open('surname10.txt', 'r')
surname_list = f.read().split('\n')
data=pd.read_csv('data.csv', delimiter=',')
data['preWord'].fillna('null', inplace=True)
data['postWord'].fillna('null', inplace=True)
data['isCommon']=data['word'].apply(lambda s:int(any(x.lower() in s.lower() for x in surname_list)))
data['wordlen']=data['endPos']-data['startPos']
data['isCap']=data['word'].apply(lambda s: int(all(x[0].isupper() for x in s.split())))
data['preisCap']=data['preWord'].apply(lambda s: int(all(x[0].isupper() for x in s.split())))
data['postisCap']=data['postWord'].apply(lambda s: int(all(x[0].isupper() for x in s.split())))
data['preisCommon']=data['preWord'].apply(lambda s:int(any(x.lower() in s.lower() for x in surname_list)))
data['postisCommon']=data['postWord'].apply(lambda s:int(any(x.lower() in s.lower() for x in surname_list)))
data1=data[(data['docID']<=200) & (data['isCap']==1)]
data2=data[(data['docID']<=200) & (data['isCap']==0)&(data['label']==1)]
train=data1.append(data2, ignore_index=True)
data3=data[(data['docID']>200) & (data['docID']<=300)& (data['isCap']==1)]
data4=data[(data['docID']>200) & (data['docID']<=300)& (data['isCap']==0)&(data['label']==1)]
test=data3.append(data4, ignore_index=True)

x=train[['isCommon','wordlen','startPos','bag','preisCap','postisCap','preisCommon','postisCommon']].values
y=train[['label']].values
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(x, y)
y = np.array(y.ravel()).astype(int)
precision = 0
recall = 0
for train_index, test_index in skf.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # clf = LinearRegression()
    # clf.fit(X_train, y_train)
    # predsLM=np.where(clf.predict(X_test)>0.2,1,0)
    # precision+=precision_score(y_test, predsLM)
    # recall+=recall_score(y_test, predsLM)

    # clf=LogisticRegression()
    # clf.fit(X_train, y_train)
    # probLG=clf.predict_proba(X_test)
    # predsLG=np.where(probLG[:,1]>0.2,1,0)
    # precision+=precision_score(y_test, predsLG)
    # recall+=recall_score(y_test, predsLG)

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    probRF = clf.predict_proba(X_test)
    predsRF = np.where(probRF[:, 1] > 0.2, 1, 0)
    precision += precision_score(y_test, predsRF)
    recall += recall_score(y_test, predsRF)
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # probDT=clf.predict_proba(X_test)
    # predsDT=np.where(probDT[:,1]>0.2,1,0)
    # precision+=precision_score(y_test, predsDT)
    # recall+=recall_score(y_test, predsDT)
    # clf=svm.SVC()
    # clf.fit(X_train, y_train)
    # probSV=clf.predict_proba(X_test)
    # predsSV=np.where(probSV[:,1]>0.2,1,0)
    # precision+=precision_score(y_test, predsSV)
    # recall+=recall_score(y_test, predsSV)
precision = precision / 2
recall = recall / 2
print(precision, recall)