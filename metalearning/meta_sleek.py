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
	
	
	
def calc_covariance(x):
    x = cp.asarray(x)
    N = x.shape[2]
    m1 = x - x.sum(2,keepdims=1)/N
    out = np.einsum('ijk,ilk->ijl',m1,m1)  / (N - 1)
    return out

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
    '''import matplotlib.pyplot as plt
    print("dataset : ",d)
    plt.hist(length,bins=50)
    plt.xlim((0,max(length)))
    plt.savefig("train_before_"+name+".png")'''
    #l = remove_outliers(length)
    #results = [x for x in results if x.shape[0] in l]
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



def get_meta_features(x_data, d,max_len):
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
     data['seq_length']=max_len
     print('stats accross data :')
     print("mean: ",mean.item())
     print("variance: ",var.item())
     print("skewness: ",skews.item())
     print("kurtosis: ",kurtosis.item())
     return data

def calc_stats(path):
    dirs = os.listdir(path)
    print(dirs)
    #dirs=["spokenlanguage"]
    #dirs = ['flickr','urbansound']
    if "test" in dirs:
        dirs.remove("test")
    if "starcraft" in dirs:
        dirs.remove("starcraft")
    if "flickr" in dirs:
        dirs.remove("flickr")
    if "oldmes" in dirs:
        dirs.remove("oldmes")
    if "spokenlanguage" in dirs:
        dirs.remove("spokenlanguage")
    if "latest" in dirs:
        dirs.remove("latest")
    '''if "env-sound" in dirs:
        dirs.remove("env-sound")
    if "music_genre" in dirs:
        dirs.remove("music_genre")'''
	
    #data_to_write=[]
    #dirs = ['music_env_speech']
    row = 0
    col = 0

    heading = ['name','seq_length','mean','var','skew','kurtosis']#,'sampling_rate','num_train_samples','num_test_samples','output_dim']
    l = ['mean','var','skew','kurtosis']
    for word in l:
        c =0
        while c < 13:
            c+=1
            w=word+"_"+str(c)
            heading.append(w)
    print(heading)
    create_csv_file(heading)
    
    
    for d in dirs:
        print(d)
        p = os.path.join(path,d)
        D = AutoSpeechDataset(os.path.join(p,d+".data"))
        metadata = D.get_metadata()
        D.read_dataset()
        output_dim = D.get_class_num()
        D_train = D.get_train()
        D_test = D.get_test()
        '''num_test_samples = metadata['test_num']
        num_train_samples = metadata['train_num']
        sampling_rate = 16000
        for k,v in metadata.items():
            if k == 'sampling_rate':
                sampling_rate = metadata['sampling_rate']'''
        
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
        
        import matplotlib.pyplot as plt
        train_len = [len(_) for _ in train_features]
        plt.hist(train_len,bins=range(0,len(train_features)))
        plt.xlim((0,max_len))
        plt.savefig("train_after_"+d+".png")
        train_features = pad_seq(train_features, max_len)  # padding at the beginning of the input
        test_features = pad_seq(test_features, max_len)
        
        print(test_features.shape)
        print(train_features.shape)
        
        '''x_data = np.concatenate((train_features,test_features),axis=0)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_data = torch.from_numpy(x_data).float().to(device)'''
        process(train_features, test_features,d,max_len,heading,batch_size=256)
        
        '''train_loader,test_loader,val_loader = get_data_loader(train_features, test_features, batch_size=64)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        for x in train_loader:
            x= torch.stack(x)
            x_data = x.float().to(device)
            data = get_meta_features(x_data,d)        
            #data_to_write.append(data)
            write_csv_file(data,heading)

        for x in test_loader:
            x= torch.stack(x)
            x_data = x.float().to(device)
            data = get_meta_features(x_data,d)                                                                                                                                                                     
            write_csv_file(data,heading)
            #data_to_write.append(data)

        for x in val_loader:
            x= torch.stack(x)
            x_data = x.float().to(device)
            data = get_meta_features(x_data,d)
            write_csv_file(data,heading)
            #data_to_write.append(data)'''
def process(train_features, test_features, d, max_len,heading , batch_size=64):
    x_data = np.concatenate((train_features,test_features),axis=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_data = torch.from_numpy(x_data).float().to(device)
    idx = torch.randperm(x_data.nelement())
    t = x_data.view(-1)[idx].view(x_data.size())#shuffle
    batches = torch.split(x_data,batch_size)
    print("batches :", len(batches))
    for x_data in batches:
        data = get_meta_features(x_data,d,max_len) 
        write_csv_file([data],heading)
    #return data_to_write

def model(d):
    import json

    with open('models.json') as json_file:
         data = json.load(json_file)
         return data['models'][d]
         '''for p in data['model_configs']:
             if p['id'] == data['models'][d]:
                 lr = p['lr']'''


def write_csv_file(data_to_write,heading):
    row = 0
    print(heading)
    print(data_to_write)
    dd=[]
    while row < len(data_to_write):
        col = 0
        line=[]
        while col < 58:#increase this if u want to add label model_id
            print(data_to_write[row][heading[col]])
            line.append(str(data_to_write[row][heading[col]]))
            col+=1
        row+=1
        dd.append(line)
    with open('data_256.csv', mode='a', newline='') as data_file:
        writer = csv.writer(data_file)
        writer.writerows(dd)



def create_csv_file(heading):
    with open('data_256.csv', mode='w+') as data_file:
            writer = csv.writer(data_file)
            writer.writerow(heading)
'''
row = 0
col = 0

heading = ['name','mean','var','skew','kurtosis']#,'sampling_rate','num_train_samples','num_test_samples','output_dim']
l = ['mean','var','skew','kurtosis']
for word in l:
    c =0
    while c < 13:
        c+=1
        w=word+"_"+str(c)
        heading.append(w)
#heading.append("model_id")
print(heading)'''


#dd=[]
#dd.append(heading)

#if os.path.exists('data.csv'):
#    os.remove("data.csv")
#create_csv_file(heading)

#data_to_write = calc_stats("../sample_data/challenge/test")
#write_csv_file(data_to_write, dd, heading)
#dd=[]# dont write heading twice
#data_to_write =
calc_stats("../sample_data/challenge")
#write_csv_file(data_to_write, dd, heading)
#data_to_write = calc_stats("../sample_data/challenge/latest")
#write_csv_file(data_to_write, dd, heading)


print("ajhkqjekqje")
