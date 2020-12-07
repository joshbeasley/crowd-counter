import numpy as np
import cv2
import os
import random
import pandas as pd
import sys

class ImageDataLoader():
    def __init__(self, data_path, num_classes=10):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        #num_classes: total number of classes into which the crowd count is divided (default: 10 as used in the paper)
        self.data_path = data_path
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename))]
        self.data_files.sort()
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = range(0,self.num_samples)
        self.num_classes = num_classes
        self.count_class_hist = np.zeros(self.num_classes)        
        self.preload_data() #load input images and grount truth into memory                
            
    
    def get_classifier_weights(self):
        #since the dataset is imbalanced, classifier weights are used to ensure balance.
        #this function returns weights for each class based on the number of samples available for each class
        wts = self.count_class_hist
        wts = 1-wts/(sum(wts));
        wts = wts/sum(wts);
        return wts
        
    def preload_data(self):
        print('Pre-loading the data. This may take a while...')
        idx = 0
        for fname in self.data_files:            
            img = self.read_image(fname)
            
            blob = {}
            blob['data']=img
            blob['fname'] = fname                                
            
            self.blob_list[idx] = blob
            idx = idx+1
            if idx % 100 == 0:                               
                print('Loaded ', idx , '/' , self.num_samples)
        print('Completed loading ' ,idx, 'files')    
                    
    def __iter__(self):
        files = self.data_files
        id_list = self.id_list
       
        
        for idx in id_list:
            blob = self.blob_list[idx]    
            blob['idx'] = idx             
            yield blob
        
    def get_num_samples(self):
        return self.num_samples
                
    def read_image(self,fname):
        img = cv2.imread(os.path.join(self.data_path,fname),0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = int((ht/4)*4)
        wd_1 = int((wd/4)*4)
        img = cv2.resize(img,(wd_1,ht_1))
        img = img.reshape((1,1,img.shape[0],img.shape[1]))

        return img
        
            
        