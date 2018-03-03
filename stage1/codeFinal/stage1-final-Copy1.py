
# coding: utf-8



get_ipython().magic(u'matplotlib inline')
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
from nltk.corpus import stopwords


f = open('surname10.txt', 'r')
surname_list = f.read().split('\n')
f = open('cityname.txt', 'r')
cityname_list = f.read().split('\n')
f = open('frequentword.txt', 'r')
frequent_list = f.read().split('\n')
f = open('country.txt', 'r')
country_list =f.read().split('\n')
# f = open('blacklist.txt', 'r')
# blacklist =f.read().split('\n')

#print surname_list
data=pd.read_csv('data_shuffled.csv', delimiter=',')
data['preWord'].fillna('null', inplace=True)
data['postWord'].fillna('null', inplace=True)

data['isCommon']=data['word'].apply(lambda s:int(any(x.lower() in s.lower() for x in surname_list)))
data['isCity']=data['word'].apply(lambda s:int(any(x.lower() in s.lower() for x in cityname_list)))
data['isFrequentword']=data['word'].apply(lambda s:int(any(x.lower() in s.lower().split() for x in frequent_list)))
# data['isStopwords']=data['word'].apply(lambda s:int(any(x.lower() in s.lower() for x in stopwords_list)))
data['isCountry']=data['word'].apply(lambda s:int(any(x.lower() in s.lower() for x in country_list)))
# data['isBlacked']=data['word'].apply(lambda s:int(any(x.lower() in s.lower() for x in blacklist)))

data['wordlen']=data['endPos']-data['startPos']
data['isCap']=data['word'].apply(lambda s: int(all(x[0].isupper() for x in s.split())))
data['preisCap']=data['preWord'].apply(lambda s: int(all(x[0].isupper() for x in s.split())))
data['postisCap']=data['postWord'].apply(lambda s: int(all(x[0].isupper() for x in s.split())))
data['preisCommon']=data['preWord'].apply(lambda s:int(any(x.lower() in s.lower() for x in surname_list)))
data['postisCommon']=data['postWord'].apply(lambda s:int(any(x.lower() in s.lower() for x in surname_list)))
data['preisFrequentword']=data['preWord'].apply(lambda s:int(any(x.lower() in s.lower().split() for x in frequent_list)))
data['postisFrequentword']=data['postWord'].apply(lambda s:int(any(x.lower() in s.lower().split() for x in frequent_list)))


# In[ ]:


data = data[(data['bag'] != 4)]
print data[(data['docID'] <= 200)].shape
print data[(data['docID'] > 200) & (data['docID'] <= 300)].shape
datanew = data[(data['isCap']==1)]
datanew_append = data[(data['isCap'] == 0) & (data['label'] == 1)]
datanew = datanew.append(datanew_append)
print datanew[(datanew['docID'] <= 200)].shape
print datanew[(datanew['docID'] > 200) & (datanew['docID'] <= 300)].shape


# In[ ]:


# print data.head(100)
# datanew = data[(data['isCap']==1)]
# datanew_append = data[(data['isCap'] == 0) & (data['label'] == 1)]
# datanew = datanew.append(datanew_append)

# set partial word
datanew['isPartial'] = 0
# datanew_pos = datanew[datanew['label'] == 1]
for i,row in datanew.iterrows():
    flag = 0
    datacmp = datanew[(datanew['docID'] == row['docID']) & (datanew['label'] == 1)]
    for j, row1 in datacmp.iterrows():
        if len(row['word'].strip()) != len(row1['word'].strip()) and row['word'].strip() in row1['word'].strip() and row['label'] == 0 and row['docID'] == row1['docID']:
#             print str(row['docID']) + " / " + row['word'] + " / " +  row1['word'] + " / " + str(row1['docID'])
            flag = 1
            break
    if flag == 1:
#         datanew.set_value(i, 'label', 1)
        datanew.set_value(i, 'isPartial', 1)
datanew['isPartial'].fillna(0, inplace=True)


# datanew_rmp = datanew[(datanew['isPartial'] == 0) & (datanew['isFrequentword'] == 0)]
# train = datanew_rmp[(datanew_rmp['docID']<=200)]

# datanew_rmp = datanew[(datanew['isFrequentword'] == 0)]
# train = datanew[(datanew['docID']>=0) & (datanew['docID']<=300)]


# In[ ]:



# datanew_rmp = datanew[(datanew['isFrequentword'] == 0) & (datanew['isPartial'] == 0)]
# print datanew_rmp[(datanew_rmp['docID'] <= 200)].shape
# print datanew_rmp[(datanew_rmp['docID'] > 200) & (datanew['docID'] <= 300)].shape
datanew_rmp = datanew
train = datanew_rmp[(datanew_rmp['docID'] <= 200) ]
test = datanew_rmp[(datanew_rmp['docID']<=300) & (datanew_rmp['docID']>200)]
train.shape


# In[ ]:


datanew_rmp


# In[ ]:


pd.set_option('display.max_rows', 1000)
datanew[(datanew['docID']<=200) & (datanew['isFrequentword'] == 0) & (datanew['isPartial'] == 1)]


# In[ ]:



import random
# x=train[['isCommon','wordlen','startPos','bag','preisCap','postisCap','preisCommon','postisCommon']]
# y=train[['label']]

# xt = pd.DataFrame(X_test)
# xt['prob'] = p
# xt
# train['random'] = [random.sample([0,1,2],1)[0] for i in range(train.shape[0])]
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(x, y)
# y = np.array(y.ravel()).astype(int)
precision = 0
recall = 0
# for num in [0,1,2]:
X_train = train[['isCommon','wordlen','startPos','bag','preisCap','postisCap','preisCommon','postisCommon','isCity','isCountry', 'docID','isFrequentword', 'preisFrequentword', 'postisFrequentword']]
y_train = train['label']
X_test = test[['isCommon','wordlen','startPos','bag','preisCap','postisCap','preisCommon','postisCommon','isCity', 'isCountry', 'docID', 'isFrequentword', 'preisFrequentword', 'postisFrequentword']]
y_test = test['label']
word = test['word']
label = test['label']
#     X_train, X_test = x.iloc[train_index,:], x.iloc[test_index,:]
#     y_train, y_test = y[train_index], y[test_index]
    
# clf = LinearRegression()
# clf.fit(X_train, y_train)
# predsLM=np.where(clf.predict(X_test)>0.34,1,0)
# precision+=precision_score(y_test, predsLM)
# recall+=recall_score(y_test, predsLM)

# clf=LogisticRegression()
# clf.fit(X_train, y_train)
# probLG=clf.predict_proba(X_test)
# predsLG=np.where(probLG[:,1]>0.3,1,0)
# precision+=precision_score(y_test, predsLG)
# recall+=recall_score(y_test, predsLG)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
 #     print clf
probRF = clf.predict_proba(X_test)
predsRF = np.where(probRF[:, 1] > 0.31, 1, 0)
p = [i[1] for i in probRF]
X_test['pred'] = np.array(p)
X_test['word'] = word
X_test['label'] = label
precision += precision_score(y_test, predsRF)
recall += recall_score(y_test, predsRF)
    
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# probDT=clf.predict_proba(X_test)
# predsDT=np.where(probDT[:,1]>0.7,1,0)
# #predsDT
# precision+=precision_score(y_test, predsDT)
# recall+=recall_score(y_test, predsDT)
    
#     clf=svm.SVC(kernel = 'linear', probability=True)
#     clf.fit(X_train, y_train)
#     probSV=clf.predict_proba(X_test)
#     predsSV=np.where(probSV[:,1]>0.3,1,0)
#     precision+=precision_score(y_test, predsSV)
#     recall+=recall_score(y_test, predsSV)
    
precision = precision
recall = recall
f1 = 2*precision*recall/(precision+recall)
print(precision, recall, f1)


# In[ ]:


pd.set_option('display.max_rows', 1000)
FP = X_test[['word','docID']][(X_test['pred'] > 0.4) & (X_test['label'] == 0)]
# print FP.shape
FP


# In[ ]:


print(FP.to_csv(sep='\t', index=False))


# In[ ]:


f1 = 2*precision*recall/(precision+recall)
print(precision, recall, f1)

