import xgboost as xgb
import os
from dataset import AutoSpeechDataset
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
import random
from sklearn.model_selection import train_test_split
import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import mixture
from meta import extract_mfcc
from meta import pad_seq
from meta import get_meta_features
def get_data(path):
    #dirs=['espeak-starcraft-words']
    dirs = os.listdir(path)
    meta_features=[]
    for d in dirs:
        print(d)
        p = os.path.join(path,d)
        D = AutoSpeechDataset(os.path.join(p,d+".data"))
        metadata = D.get_metadata()
        D.read_dataset()
        output_dim = D.get_class_num()
        D_train = D.get_train()
        D_test = D.get_test()
        train_data, train_labels = D_train
        test_data = D_test
        train_features = extract_mfcc(train_data,d)
        test_features = extract_mfcc(D_test,d)
        print(d)
        print("extracted mfcc")
        train_max_len = max([len(_) for _ in train_features])
        test_max_len = max([len(_) for _ in test_features])
        max_len = max(train_max_len, test_max_len)  # for CNN variants we need max sequence length in advance
        print("max len : ",max_len)
        train_features = pad_seq(train_features, max_len)  # padding at the beginning of the input
        test_features = pad_seq(test_features, max_len)
        x_data = np.concatenate((train_features,test_features),axis=0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(x_data.shape)
        x_data = torch.from_numpy(x_data).float().to(device)
        data = get_meta_features(x_data,d)
        meta_features.append(data)
    return meta_features





df = pd.read_csv("./data_256.csv", header=0)#,delimiter=" ")
idx = df[(df.name == "music_genre")].index
#print(idx)
dfa = df
#df = df.drop(idx)
#models = range(0,16)
#print(df)

X = df[['mean','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','skew_1','skew_2','skew_3','skew_4','skew_5','skew_6','skew_7','skew_8','skew_9','skew_10','skew_11','skew_12','skew_13','kurtosis_1','kurtosis_2','kurtosis_3','kurtosis_4','kurtosis_5','kurtosis_6','kurtosis_7','kurtosis_8','kurtosis_9','kurtosis_10','kurtosis_11','kurtosis_12','kurtosis_13']]


#X = df[['mean','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13']]

y = df[['name']]
#print(y.loc[200]['name'])
#l = [y.loc[i]['name'] for i in range (len(y))]
idx = df.index
l = [y.loc[i]['name'] for i in idx]
#print(l)
print(list(set(l)))
u = list(set(l))
print(u)
labels = {u[i]:i for i in range(len(u))}
print(labels)
#y = np.squeeze(np.eye(len(labels), dtype=int)[labels[y['name']]])
'''p=[]
for i in range (len(y)):
   p.append(labels[y.loc[i]['name']])
#print(y)
print(p)
one_hot = np.squeeze(np.eye(len(labels), dtype=int)[p])'''


#X,y = random.shuffle(X,y)
#df.set_index('name')

#X_train = X[5:]
#y_train = y[:-4]
#y_train = models

#X_test = X[:5]
#y_test = y[-4:]'''

X_train, X_test,y_train , y_test = train_test_split(X,y, test_size=.4, random_state=42, shuffle=True)

#print(X_train)
#print(y_train)

#print(X_test)
#print(y_test)

#RBF SVM
'''rbf_svm = SVC(gamma=2, C=1)i
rbf_svm.fit(X_train, y_train)
scor =  rbf_svm.score(X_test, y_test)
preds=rbf_svm.predict(X_test)
s = preds
print(preds)
print("Score of rbf svm:",scor)
pickle.dump(rbf_svm, open("./model_rbf.pickle.dat", "wb"))


#GP Classifier
gp = GaussianProcessClassifier(1.0 * RBF(1.0))
gp.fit(X_train, y_train)
scor =  gp.score(X_test, y_test)
preds=gp.predict(X_test)
print(preds)
print("Score of GP:",scor)
pickle.dump(gp, open("./model_gp.pickle.dat", "wb"))'''



xg_reg = xgb.XGBClassifier(n_estimators = 10)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
print(preds)
accuracy = accuracy_score(y_test, preds)
print("Accuracy of xgb: %f" % (accuracy))
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_test)
test = lb.transform(y_test)
pred = lb.transform(preds)
print("ROC-AUC score of xgb : %f "% (roc_auc_score(test,pred)))
pickle.dump(xg_reg, open("./model_xgboost.pickle.dat", "wb"))
print(xg_reg.feature_importances_)
feat_imp = pd.DataFrame({'importance':xg_reg.feature_importances_})
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
print(feat_imp)
#print(xg_reg.booster().get_score(importance_type="gain"))



'''gmm = mixture.GaussianMixture(n_components=16, covariance_type='full').fit(X_train)
labels = gmm.predict(X_test)
print(labels)
for l in labels:
    print(df.loc[l+5,'name'])'''


f = get_data("../sample_data/challenge/test")
test_frames=[]
#for meta_feature in f:
    #test_frames.append(pd.DataFrame.from_dict([meta_feature]))
test_df = pd.DataFrame.from_dict(f)

#test_df=pd.concatenate(test_frames)
#test_df = dfa[(dfa.name == "music_genre")]
print(test_df)
X_v = test_df[['mean','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','skew_1','skew_2','skew_3','skew_4','skew_5','skew_6','skew_7','skew_8','skew_9','skew_10','skew_11','skew_12','skew_13','kurtosis_1','kurtosis_2','kurtosis_3','kurtosis_4','kurtosis_5','kurtosis_6','kurtosis_7','kurtosis_8','kurtosis_9','kurtosis_10','kurtosis_11','kurtosis_12','kurtosis_13']]

#X_v = test_df[['mean','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13']]
y_v = test_df[['name']]
print(y_v)
preds = xg_reg.predict(X_v)
print(preds)
