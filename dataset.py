import numpy as np
from torchvision import transforms
import common_dataset as cd
import json_process as jp
import torch.utils.data as Data
import FI_dataset
import os
import torch
import math
import json


resize_transforms={
'train': transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

}

pretrain_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
}

pretrain_transforms_test = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    'test': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}

resnet_transforms = {
    'train': transforms.Compose([
        transforms.Resize(600),
        transforms.RandomResizedCrop(446),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(600),
        transforms.TenCrop(446),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ]),
    'test': transforms.Compose([
        transforms.Resize(600),
        transforms.TenCrop(446),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ])
}


def read_text(path):
    imagery_dict={}
    imagery_dict = jp.encode_json_data(path)

    return imagery_dict

def read_label(path):
    image_dict={}
    with open(path) as f:
        for line in f:
            current=line.split()
            data = np.array([(float(current[1]) - 1) / 10.0,
                              (float(current[2]) - 1) / 5.0], dtype=np.float32)
            if(math.isnan(data[0])):
                continue
            image_dict[current[0]]=data
    return image_dict

def generate_AllData_dataset(path,mode):

    image_path=os.path.join(path,"Dataset")
    return cd.dict_dataset(read_text(os.path.join(path,"AllData.txt")),
                           read_label(os.path.join(path,"AllData_label.txt")),
                           image_path,mode,resize_transforms,load_mode=False)

def generate_Person_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    return cd.dict_dataset(read_text(os.path.join(path,"Person.txt")),
                           read_label(os.path.join(path,"Person_label.txt")),
                           image_path,mode,resize_transforms,load_mode=False)

def generate_Animal_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    return cd.dict_dataset(read_text(os.path.join(path,"Animal.txt")),
                           read_label(os.path.join(path,"Animal_label.txt")),
                           image_path,mode,resize_transforms,load_mode=False)

def generate_Plant_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    return cd.dict_dataset(read_text(os.path.join(path,"Plant_label.txt")),
                           read_label(os.path.join(path,"Plant_label.txt")),
                           image_path,mode,resize_transforms,load_mode=False)

def generate_Environment_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    return cd.dict_dataset(read_text(os.path.join(path,"Environment.txt")),
                           read_label(os.path.join(path,"Environment_label.txt")),
                           image_path,mode,resize_transforms,load_mode=False)

def generate_Mix_dataset(path,mode):
    image_path=os.path.join(path,"Dataset")
    return cd.dict_dataset(read_text(os.path.join(path,"Mix.txt")),
                           read_label(os.path.join(path,"Mix_label.txt")),
                           image_path,mode,resize_transforms,load_mode=False)

def param_dict_producer(path,dataset,batch_size,epochs):
    if(dataset=="AllData"):
        generate_function=generate_AllData_dataset
    if(dataset=="Person"):
        generate_function=generate_Person_dataset
    if(dataset=="Animal"):
        generate_function=generate_Animal_dataset
    if(dataset=="Plant"):
        generate_function=generate_Plant_dataset
    if(dataset=="Environment"):
        generate_function=generate_Environment_dataset
    if(dataset=="Mix"):
        generate_function=generate_Mix_dataset

    param_dict={}
    param_dict["train_loader"]=Data.DataLoader(generate_function(path,"train"),batch_size=batch_size,shuffle=False,num_workers=8)
    param_dict["test_loader"]=Data.DataLoader(generate_function(path,"test"),batch_size=2,shuffle=False,num_workers=32)
    param_dict["val_loader"]=Data.DataLoader(generate_function(path,"val"),batch_size=2,shuffle=False,num_workers=32)
    param_dict["epochs"]=epochs
    return param_dict
