import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import random

def init(foldername):

    df = pd.DataFrame(columns=['class_id', 'file', 'segmentation_split'])
    df["file"]=[f for f in listdir(foldername) if isfile(join(foldername, f)) and f.endswith(".JPG")]
    df.to_csv(foldername+"/annotation.csv", sep=',',index=False)

def split(foldername,train_test_val=(0.4,0.2,0.4)):
    df = pd.read_csv(foldername+"/annotation.csv")
    df=df.sort_values(by=['class_id'])
    class_id=np.unique(df["class_id"])
    random.shuffle(class_id)
    train, validate, test = np.split(class_id,[int(.4*len(class_id)), int(.6*len(class_id))])
    for i in range(len(df)):
        if np.isin(df.at[i,"class_id"],train):
            df.at[i,"segmentation_split"]="training"
        if np.isin(df.at[i,"class_id"],test):
            df.at[i,"segmentation_split"]="testing"
        if np.isin(df.at[i,"class_id"],validate):
            df.at[i,"segmentation_split"]="validation"
            
    df.to_csv(foldername+"/annotation.csv", sep=',',index=False)


if __name__ == '__main__':
    foldername=""
    init(foldername)
    #split(foldername)