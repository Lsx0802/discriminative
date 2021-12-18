# coding=utf-8
import numpy as np

#An example of centroid distance, based on pytorch1.9

def cal_Cmass(data):
    '''
    input:data(ndarray):数据样本
    output:mass(ndarray):数据样本质心
    '''
    Cmass = np.mean(data,axis=0)
    return Cmass

def distance(x, y, p=2):
    '''
    input:x(ndarray):第一个样本的坐标
          y(ndarray):第二个样本的坐标
          p(int):等于1时为曼哈顿距离，等于2时为欧氏距离
    output:distance(float):x到y的距离
    '''
    dis2 = np.sum(np.abs(x-y)**p) # 计算
    dis = np.power(dis2,1/p)
    return dis

def mean_list(data,Cmass):
    '''
    input:data(ndarray):数据样本
          Cmass(ndarray):数据样本质心
    output:dis_list(list):样本到质心距离平均值
    '''
    dis_list = []
    for i in range(len(data)):       # 遍历data数据，与质心cmass求距离
        dis_list.append(distance(Cmass,data[i][:]))
    dis_list = np.mean(dis_list)      # 排序
    return dis_list


import torch
from torch.autograd import Variable
import numpy as np
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from LeNet5 import LeNet5_2,LeNet5
import matplotlib.patheffects as PathEffects
import os
from sklearn.manifold import TSNE
import SimpleITK as sitk
from Mydataset import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1

test_datapath = r'C:\Users\hello\PycharmProjects\HCC\2D3Dfusion\dataset2D/test/DMQ/Axial/'
test_txtpath = r'C:\Users\hello\PycharmProjects\HCC\2D3Dfusion\dataset2D/test.txt'
train_datapath = r'C:\Users\hello\PycharmProjects\HCC\2D3Dfusion\dataset2D/train/DMQ/Axial/'
train_txtpath = r'C:\Users\hello\PycharmProjects\HCC\2D3Dfusion\dataset2D/train.txt'


transforms_ = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])

test_data = MyDataset(txt=test_txtpath, transform=transforms_, path=test_datapath)

# 将数据集导入DataLoader，进行shuffle以及选取batch_size
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
#######################


model_weight_path = "leNet5_2D_4.pkl"
# model_weight_path = "leNet5_2D_original3.pkl"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

# option1
model = LeNet5_2()
nodel=model.load_state_dict(torch.load(model_weight_path))
model=model.to(device)

feature=[]
label=[]
model.eval()

for data, target in test_data_loader:
    data= data.to(device)
    target=target.to(device)
    data, target = Variable(data, volatile=True), Variable(target)
    x ,output= model(data)
    feature.extend(output.detach().cpu().numpy())
    label.extend(target.detach().cpu().numpy())

feature0=[]
feature1=[]
for i in range(len(label)):
    if label[i]==0:
        feature0.append(feature[i])
    else:
        feature1.append(feature[i])

cmass0 = cal_Cmass(feature0)
cmass1 = cal_Cmass(feature1)

list0 = mean_list(feature0,cmass0)
list1 = mean_list(feature1,cmass1)
print(list0)
print(list1)

list3=mean_list(feature0,cmass1)
list4=mean_list(feature1,cmass0)
print(list3)
print(list4)
