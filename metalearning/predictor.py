import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'ingestion'))
sys.path.append(os.path.join(os.getcwd(), 'scoring'))
import xgboost as xgb
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
import matplotlib.pyplot as plt
#sys.path.append('../')
import extract_best_model
import run_local_test
def get_data(path):
    #dirs=['espeak-starcraft-words']
    dirs = os.listdir(path)
    meta_features=[]
    #dirs=['data03']
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
        data = get_meta_features(x_data,d,max_len)
        meta_features.append(data)
    return meta_features





df = pd.read_csv("./ingestion/data_256.csv", header=0)#,delimiter=" ")
idx = df[(df.name == "flickr")].index

dfa = df
df = df.drop(idx)


X = df[['mean','seq_length','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','skew_1','skew_2','skew_3','skew_4','skew_5','skew_6','skew_7','skew_8','skew_9','skew_10','skew_11','skew_12','skew_13','kurtosis_1','kurtosis_2','kurtosis_3','kurtosis_4','kurtosis_5','kurtosis_6','kurtosis_7','kurtosis_8','kurtosis_9','kurtosis_10','kurtosis_11','kurtosis_12','kurtosis_13']]


y = df[['name']]
#print(y.loc[200]['name'])
#l = [y.loc[i]['name'] for i in range (len(y))]
idx = df.index
l = [y.loc[i]['name'] for i in idx]
#print(l)
print(list(set(l)))
u = list(set(l))
print(u)
labels = {i:u[i] for i in range(len(u))}
print(labels)
#y = np.squeeze(np.eye(len(labels), dtype=int)[labels[y['name']]])

X_train, X_test,y_train , y_test = train_test_split(X,y, test_size=.4, random_state=42, shuffle=True)



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
#print(preds)
accuracy = accuracy_score(y_test, preds)
print("Accuracy of xgb: %f" % (accuracy))
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)
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
print('class mappings :  {}'.format(xg_reg.classes_))
feat_imp.plot.barh(title='XGBoost Classifier',fontsize=34,figsize=(7,7))
plt.xlabel('Feature Importance', fontsize=10)
plt.ylabel('Meta Feature', fontsize=10)
plt.yticks(fontsize=6)
plt.xticks(fontsize=8)
plt.savefig('feature_imp.png')

'''gmm = mixture.GaussianMixture(n_components=16, covariance_type='full').fit(X_train)
labels = gmm.predict(X_test)
print(labels)
for l in labels:
    print(df.loc[l+5,'name'])'''


f = get_data("../../jinu_hpo/AutoDL/sample_data/challenge/test/")
test_frames=[]
#for meta_feature in f:
    #test_frames.append(pd.DataFrame.from_dict([meta_feature]))
test_df = pd.DataFrame.from_dict(f)


print(test_df)
X_v = test_df[['mean','seq_length','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','skew_1','skew_2','skew_3','skew_4','skew_5','skew_6','skew_7','skew_8','skew_9','skew_10','skew_11','skew_12','skew_13','kurtosis_1','kurtosis_2','kurtosis_3','kurtosis_4','kurtosis_5','kurtosis_6','kurtosis_7','kurtosis_8','kurtosis_9','kurtosis_10','kurtosis_11','kurtosis_12','kurtosis_13']]

y_v = test_df[['name']]
print(y_v)
preds = xg_reg.predict(X_v)
print(preds)
best_models = extract_best_model.get_incumbent()




for index,row in y_v.iterrows():
    dataset = row['name']
    print(dataset)
    idx = test_df.loc[(test_df.name == dataset)]
    x=idx
    X_v = x[['mean','seq_length','var','skew','kurtosis','mean_1','mean_2','mean_3','mean_4','mean_5','mean_6','mean_7','mean_8','mean_9','mean_10','mean_11','mean_12','mean_13','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13','skew_1','skew_2','skew_3','skew_4','skew_5','skew_6','skew_7','skew_8','skew_9','skew_10','skew_11','skew_12','skew_13','kurtosis_1','kurtosis_2','kurtosis_3','kurtosis_4','kurtosis_5','kurtosis_6','kurtosis_7','kurtosis_8','kurtosis_9','kurtosis_10','kurtosis_11','kurtosis_12','kurtosis_13']]
    #y_v = x.loc['name']
    print(X_v)
    pred = xg_reg.predict(X_v)
    prob = xg_reg.predict_proba(X_v)
    prob = prob.squeeze(axis = 0).tolist()
    print(prob)
    import heapq
    print(heapq.nlargest(2,prob)[1])
    second_best = prob.index(heapq.nlargest(2,prob)[1])
    print("second best : {}".format(xg_reg.classes_[second_best]))
    print("prediction {} ".format(pred[0]))
    print('dataset {} is similar to dataset {} '.format(dataset,pred[0]))
    print('running model for dataset {} '.format(dataset))
    data = extract_best_model.get_incumbent()
    #data = best_models
    for d in data:
        print('checking if {} data matches {}'.format(d['dataset'],dataset))
        if d['dataset'] == pred[0]:
            print(d['config'])
            print(d['model'])
            cfg = d
            cfg['dataset_dir']="./data/{}".format(dataset)
            cfg['dataset']=dataset
            import bohb_epoch
            alc = bohb_epoch.execute_run(cfg,d['config'],1200)
            print('ALC score for dataset {} running model {} is {} '.format(dataset, d['model'],alc))
            break




