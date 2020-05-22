import os
from scipy import stats
from dataset import AutoSpeechDataset
import librosa
from tensorflow.python.keras.preprocessing import sequence
import tensorflow as tf
import numpy as np
import cupy as cp
import torch
import minpy
from mxnet import nd
from numpy import percentile
from sklearn.cluster import KMeans
print(torch.version.cuda)
import csv
from sklearn.model_selection import train_test_split
import torch.utils.data

def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='pre')


def extract_mfcc(data,name,sr=16000):
    results = []
    length =[]
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=13)
        r = r.transpose()
        length.append(r.shape[0])
        results.append(r)
    l = remove_outliers(length)
    results = [x for x in results if x.shape[0] in l]
    return results


def remove_outliers(data):
    # calculate summary statistics
    print("removing outliers function ")
    print("individual lengths :", data)
    data_mean, data_std = np.mean(data), np.std(data)
    # identify outliers
    cut_off = data_std * 2
    lower, upper = data_mean - cut_off, data_mean + cut_off
    print("lower: ",lower)
    print("upper: ",upper)
    print("max : ", max(data))
    # identify outliers
    outliers = [x for x in data if x < lower or x > upper]
    p = len(outliers)/len(data)
    print('Identified outliers: %d' % len(outliers))
    print('% of outliers : ',p)
    print(outliers)
    d = max(data)*len(data)
    #print(data)
    # remove outliers
    if p < 0.1 and d > 32: 
        outliers_removed = [x for x in data if x >= lower and x <= upper]
        #print(outliers_removed)
        print("outliers removed!")
        return  outliers_removed
    else:
        return data

def get_data_loader(train_features, test_features,batch_size=64):
    train_X = torch.Tensor(train_features)
    test_X = torch.Tensor(test_features)

    train_dataset = torch.utils.data.TensorDataset(train_X)
    test_dataset = torch.utils.data.TensorDataset(test_X)

    test_dataset,test_holdout_dataset = train_test_split(test_dataset, test_size=.4, random_state=42)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=4,
                                                       shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size,num_workers=4,
                                                      shuffle=False )

    val_loader = torch.utils.data.DataLoader(dataset=test_holdout_dataset,  batch_size=batch_size,num_workers=4,
                                                      shuffle=False )
    return train_loader,test_loader,val_loader



def get_meta_features(x_data, d):
     print("inside meta features function")
     print("size of each batch :", x_data.shape)
     mean = torch.mean(x_data,0,False)
     diffs = x_data - mean
     var = torch.mean(torch.pow(diffs, 2.0),0,False)
     std = torch.pow(var, 0.5)
     zscores = diffs / std
     skews = torch.mean(torch.pow(zscores, 3.0),0,False)
     kurtosis = torch.mean(torch.pow(zscores, 4.0),0,False) - 3.0
     mean_var = torch.mean(var,0,False)
     mean_var[torch.isnan(mean_var)]=0
     mean_kurtosis = torch.mean(kurtosis,0,False)
     mean_kurtosis[torch.isnan(mean_kurtosis)]=0
     mean_skew = torch.mean(skews,0,False)
     mean_skew[torch.isnan(mean_skew)]=0
     mean_mean=torch.mean(mean,0,False)
     mean_mean[torch.isnan(mean_mean)]=0
     
     data={}

     data['name']=d
     for i in range(1,14):
         st = "mean_"+str(i)
         data[st]=mean_mean[i-1].item()
     for i in range(1,14):
         st = "var_"+str(i)
         data[st]=mean_var[i-1].item()
     for i in range(1,14):
         st = "skew_"+str(i)
         data[st]=mean_skew[i-1].item()
     for i in range(1,14):
         st = "kurtosis_"+str(i)
         data[st]=mean_kurtosis[i-1].item()
     print('stats for mfcc dimension :')
     print("mean: ",mean_mean)
     print("shape of mean : ",mean.shape)
     print("variance: ",mean_var)
     print("skewness: ",mean_skew)
     print("kurtosis: ",mean_kurtosis)
     print("after using cuda: ")
     print(type(x_data))
     mean = torch.mean(x_data)
     diffs = x_data - mean
     var = torch.mean(torch.pow(diffs, 2.0))
     std = torch.pow(var, 0.5)
     zscores = diffs / std
     skews = torch.mean(torch.pow(zscores, 3.0))
     kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0
     data['mean']=mean.item()
     data['var']=var.item()
     data['skew']=skews.item()
     data['kurtosis']=kurtosis.item()
     print('stats accross data :')
     print("mean: ",mean.item())
     print("variance: ",var.item())
     print("skewness: ",skews.item())
     print("kurtosis: ",kurtosis.item())
     return data



print("ajhkqjekqje")
