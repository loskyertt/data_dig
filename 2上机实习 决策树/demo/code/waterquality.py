#-*- coding: utf-8 -*- 

# 代码9-1

import numpy as np
import os,re
from PIL import Image

# 图像切割及特征提取
path = './demo/data/images/'  # 图片所在路径
# 自定义获取图片名称函数
def getImgNames(path=path):
    '''
    获取指定路径中所有图片的名称
    :param path: 指定的路径
    :return: 名称列表
    '''
    filenames = os.listdir(path)
    imgNames = []
    for i in filenames:
        if re.findall('^\d_\d+\.jpg$', i) != []:
            imgNames.append(i)
    return imgNames

# 自定义获取三阶颜色矩函数
def Var(data=None):
    '''
    获取给定像素值矩阵的三阶颜色矩
    :param data: 给定的像素值矩阵
    :return: 对应的三阶颜色矩
    '''
    x = np.mean((data-data.mean())**3)
return np.sign(x)*abs(x)**(1/3)

# 批量处理图片数据
imgNames = getImgNames(path=path)  # 获取所有图片名称
n = len(imgNames)        # 图片张数
data = np.zeros([n, 9])  # 用来装样本自变量
labels = np.zeros([n])   # 用来放样本标签

for i in range(n):
    img = Image.open(path+imgNames[i])  # 读取图片
    M,N = img.size  # 图片像素的尺寸
    img = img.crop((M/2-50,N/2-50,M/2+50,N/2+50))    # 图片切割
    r,g,b = img.split()     # 将图片分割成三通道
    rd = np.asarray(r)/255  # 转化成数组数据
    gd = np.asarray(g)/255
    bd = np.asarray(b)/255

    data[i,0] = rd.mean()  # 一阶颜色矩
    data[i,1] = gd.mean()
    data[i,2] = bd.mean()

    data[i,3] = rd.std()   # 二阶颜色矩
    data[i,4] = gd.std()
    data[i,5] = bd.std()

    data[i,6] = Var(rd)    # 三阶颜色矩
    data[i,7] = Var(gd)
    data[i,8] = Var(bd)

    labels[i] = imgNames[i][0]  # 样本标签


	
# 代码9-2	
	
from sklearn.model_selection import train_test_split
# 数据拆分,训练集、测试集
data_tr,data_te,label_tr,label_te = train_test_split(data,labels,test_size=0.2,
                                                     random_state=10)

from sklearn.tree import DecisionTreeClassifier
# 模型训练
model = DecisionTreeClassifier(random_state=5).fit(data_tr, label_tr) 



# 代码9-3

# 水质评价
from sklearn.metrics import confusion_matrix
pre_te = model.predict(data_te)
# 混淆矩阵
cm_te = confusion_matrix(label_te,pre_te)
print(cm_te)
from sklearn.metrics import accuracy_score
# 准确率
print(accuracy_score(label_te,pre_te))
