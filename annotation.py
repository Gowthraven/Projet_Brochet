import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import random

def init(foldername,datatype,filename):

    df = pd.DataFrame(columns=['class_id', 'file', "reid_split", 'segmentation_split'])
    df["file"]=[f for f in listdir(foldername) if isfile(join(foldername, f)) and f.endswith(".JPG")]
    df["reid_split"]=[datatype]*len(df)
    df.to_csv(filename+".csv", sep=',',index=False)
    print("Done")

def split(filename,train_test_val=(0.4,0.2,0.4)):
    df = pd.read_csv(filename+".csv")
    df=df.sort_values(by=['class_id'])
    class_id=np.unique(df["class_id"])
    random.shuffle(class_id)
    train, validate, test = np.split(class_id,[int(.4*len(class_id)), int(.6*len(class_id))])
    print(train)
    print(validate)
    print(test)
    for i in range(len(df)):
        if np.isin(df.at[i,"class_id"],train):
            df.at[i,"segmentation_split"]="training"
        if np.isin(df.at[i,"class_id"],test):
            df.at[i,"segmentation_split"]="testing"
        if np.isin(df.at[i,"class_id"],validate):
            df.at[i,"segmentation_split"]="validation"
            
    df.to_csv(filename+".csv", sep=',',index=False)


if __name__ == '__main__':

    #init("../CAMPAGNE2-101114/","database","annotation2")
    split("annotation12")