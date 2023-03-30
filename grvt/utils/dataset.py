import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from imageio import imread
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

class DatasetGrocery(torch.utils.data.Dataset):
    def __init__(self,csv_file_path,transform,p_folder="/root/grocery/GroceryStoreDataset/dataset/"):
        csv=pd.read_csv(p_folder+csv_file_path,sep=',',header=None,names=["filepath","flabel","clabel"])
        self.p_folder=p_folder
        self.transform=transform
        self.img_list=csv.filepath.tolist()
        self.flabel_list=csv.flabel.tolist()
        self.clabel_list=csv.clabel.tolist()
        print("len of dataset:{}".format(len(self.img_list)))
        # print(self.img_list)
        # print(self.flabel_list)
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        flabel=self.flabel_list[index]
        clabel=self.clabel_list[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        # print(img)
        # print(flabel)
        return img,flabel
    def __len__(self):
        return len(self.flabel_list)
class DatasetGroceryCoarse(torch.utils.data.Dataset):
    def __init__(self,csv_file_path,transform,p_folder="/root/grocery/GroceryStoreDataset/dataset/"):
        csv=pd.read_csv(p_folder+csv_file_path,sep=',',header=None,names=["filepath","flabel","clabel"])
        self.p_folder=p_folder
        self.transform=transform
        self.img_list=csv.filepath.tolist()
        self.flabel_list=csv.flabel.tolist()
        self.clabel_list=csv.clabel.tolist()
        print("len of dataset:{}".format(len(self.img_list)))
        # print(self.img_list)
        # print(self.flabel_list)
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        flabel=self.flabel_list[index]
        clabel=self.clabel_list[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        # print(img)
        # print(flabel)
        return img,clabel
    def __len__(self):
        return len(self.clabel_list)

class DatasetGrocerySWAP(torch.utils.data.Dataset):
    def __init__(self,csv_file_path,transform,train=True,p_folder="/root/grocery/GroceryStoreDataset/dataset/"):
        csv=pd.read_csv(p_folder+csv_file_path,sep=',',header=None,names=["filepath","flabel","clabel"])
        self.p_folder=p_folder
        self.train=train
        self.transform=transform
        self.img_list=csv.filepath.tolist()
        self.flabel_list=csv.flabel.tolist()
        self.clabel_list=csv.clabel.tolist()
        print("len of dataset:{}".format(len(self.img_list)))
        # print(self.img_list)
        # print(self.flabel_list)
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        flabel=self.flabel_list[index]
        clabel=self.clabel_list[index]
        img=Image.open(img).convert('RGB')
        # swap image
        if(self.train and random.choice([True, False])):
            img=swap(img,(4,4))
            
        img=self.transform(img)
        # print(img)
        # print(flabel)
        return img,flabel
    def __len__(self):
        return len(self.flabel_list)

class DatasetGrocerySplit(torch.utils.data.Dataset):
    def __init__(self,transform,train=True,p_folder="/root/grocery/GroceryStoreDataset/dataset/",split=0):
        if(train):
            txt=p_folder+"split/train"+str(split)+".csv"
        else:
            txt=p_folder+"split/test"+str(split)+".csv"
        csv=pd.read_csv(txt,sep=',',header=None,names=["filepath","flabel","clabel"])
        self.p_folder=p_folder
        self.transform=transform
        self.img_list=csv.filepath.tolist()
        self.flabel_list=csv.flabel.tolist()
        self.clabel_list=csv.clabel.tolist()
        print("len of dataset:{}".format(len(self.img_list)))
        # print(self.img_list)
        # print(self.flabel_list)
    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        flabel=self.flabel_list[index]
        clabel=self.clabel_list[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        # print(img)
        # print(flabel)
        return img,flabel
    def __len__(self):
        return len(self.flabel_list)

class DatasetFreiburg(torch.utils.data.Dataset):
    def __init__(self,transform,train=True,split=0):
        path="/root/grocery/Freiburg/"
        self.p_folder=path+"images/"
        self.img_list=[]
        self.label_list=[]
        if(train):
            txt=path+"splits/train"+str(split)+".txt"
        else:
            txt=path+"splits/test"+str(split)+".txt"

        # for txt in txts:
        csv=pd.read_csv(txt,delim_whitespace=True,header=None,names=["filepath","label"])
        self.img_list=csv.filepath.tolist()
        self.label_list=csv.label.tolist()
        self.transform=transform
        print("len of dataset:{}".format(len(self.img_list)))

    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        label=self.label_list[index]
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.label_list)


class DatasetFreiburgSWAP(torch.utils.data.Dataset):
    def __init__(self,transform,train=True,split=0):
        path="/root/grocery/Freiburg/"
        self.p_folder=path+"images/"
        self.img_list=[]
        self.label_list=[]
        self.train=train
        if(train):
            txt=path+"splits/train"+str(split)+".txt"
        else:
            txt=path+"splits/test"+str(split)+".txt"

        # for txt in txts:
        csv=pd.read_csv(txt,delim_whitespace=True,header=None,names=["filepath","label"])
        self.img_list=csv.filepath.tolist()
        self.label_list=csv.label.tolist()
        self.transform=transform
        print("len of swap split_{} dataset:{}".format(split,len(self.img_list)))

    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        label=self.label_list[index]
        img=Image.open(img).convert('RGB')
        if(self.train and random.choice([True, False])):
            img=swap(img,(7,7))
        img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.label_list)

class DatasetProducts10k(torch.utils.data.Dataset):
    def __init__(self,transform,train=True,split=0):
        path="/root/grocery/Product10k/"
        self.p_folder=path
        self.img_list=[]
        self.label_list=[]
        self.train=train
        if(train):
            
            self.p_folder=path+"train/"
            txt=path+"train.csv"
            csv=pd.read_csv(txt,sep=",",header=0,names=["name","classes","group"])
            self.img_list=csv.name.tolist()
            self.label_list=csv.classes.tolist()
        else:
            self.p_folder=path+"test/"
            txt=path+"test.csv"
            csv=pd.read_csv(txt,sep=",",header=0,names=["name"])
            self.img_list=csv.name.tolist()
        self.transform=transform
        print("len of swap split_{} dataset:{}".format(split,len(self.img_list)))

    def __getitem__(self,index):
        #print("FILE:{}".format(self.imgs[index]))
        img=self.p_folder+self.img_list[index]
        #img_name=self.img_list[index]
        
        img=Image.open(img).convert('RGB')
        img=self.transform(img)
        if(self.train):
            label=self.label_list[index]
            return img,label
        else:
            return img, index
    def __len__(self):
        return len(self.img_list)


def swap(img, crop):
    def crop_image(image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    widthcut, highcut = img.size
    img = img.crop((10, 10, widthcut-10, highcut-10))
    images = crop_image(img, crop)
    pro = 5
    if pro >= 5:          
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 1
        RAN = 2
        for i in range(crop[1] * crop[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
            if count_x == crop[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                random.shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
        random_im = []
        for line in tmpy:
            random_im.extend(line)
        
        # random.shuffle(images)
        width, high = img.size
        iw = int(width / crop[0])
        ih = int(high / crop[1])
        toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
        x = 0
        y = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            x += 1
            if x == crop[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    return toImage

class CUB():
    def __init__(self, root="/root/grocery/CUB_200_2011", is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_img = [imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]
        if not self.is_train:
            self.test_img = [imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]
    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)