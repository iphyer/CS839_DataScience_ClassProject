train['random'] = [random.sample([0,1,2],1)[0] for i in range(train.shape[0])]
precision = 0
recall = 0
for num in [0,1,2]:
    X_train = train[['isCommon','wordlen','startPos','bag','preisCap','postisCap','preisCommon','postisCommon','isCity','isCountry', 'docID', 'isFrequentword', 'preisFrequentword', 'postisFrequentword']][train['random'] != num]
    y_train = train['label'][train['random'] != num]
    X_test = train[['isCommon','wordlen','startPos','bag','preisCap','postisCap','preisCommon','postisCommon','isCity', 'isCountry', 'docID', 'isFrequentword', 'preisFrequentword', 'postisFrequentword']][train['random'] == num]
    y_test = train['label'][train['random'] == num]
    word = train['word'][train['random'] == num]
    label = train['label'][train['random'] == num]
#     X_train, X_test = x.iloc[train_index,:], x.iloc[test_index,:]
#     y_train, y_test = y[train_index], y[test_index]
    
#     clf = LinearRegression()
#     clf.fit(X_train, y_train)
#     predsLM=np.where(clf.predict(X_test)>0.34,1,0)
#     precision+=precision_score(y_test, predsLM)
#     recall+=recall_score(y_test, predsLM)

#     clf=LogisticRegression()
#     clf.fit(X_train, y_train)
#     probLG=clf.predict_proba(X_test)
#     predsLG=np.where(probLG[:,1]>0.3,1,0)
#     precision+=precision_score(y_test, predsLG)
#     recall+=recall_score(y_test, predsLG)

    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    probRF = clf.predict_proba(X_test)
    predsRF = np.where(probRF[:, 1] > 0.45, 1, 0)
    p = [i[1] for i in probRF]
    X_test['pred'] = np.array(p)
    X_test['word'] = word
    X_test['label'] = label
    precision += precision_score(y_test, predsRF)
    recall += recall_score(y_test, predsRF)
    
#     clf = tree.DecisionTreeClassifier()
#     clf.fit(X_train, y_train)
#     probDT=clf.predict_proba(X_test)
#     predsDT=np.where(probDT[:,1]>0.2,1,0)
#     #predsDT
#     precision+=precision_score(y_test, predsDT)
#     recall+=recall_score(y_test, predsDT)
    
#     clf=svm.SVC(kernel = 'linear', probability=True)
#     clf.fit(X_train, y_train)
#     probSV=clf.predict_proba(X_test)
#     predsSV=np.where(probSV[:,1]>0.3,1,0)
#     precision+=precision_score(y_test, predsSV)
#     recall+=recall_score(y_test, predsSV)
    
precision = precision / 3
recall = recall / 3
f1 = 2*precision*recall/(precision+recall)
print(precision, recall, f1)