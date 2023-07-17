# 作者:     wxf

# 开发时间: 2022/6/10 15:48
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import csv
import math
import random
import xlrd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

data = []
ReadMyCsv(data, "关系对.csv")
print(len(data))

Allcirc = []
counter1 = 0
while counter1 < len(data): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(Allcirc):  #遍历AllDisease
        if data[counter1][1] != Allcirc[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif data[counter1][1] == Allcirc[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        Allcirc.append(data[counter1][1])
    counter1 = counter1 + 1
print('len(Allcirc)', len(Allcirc))
# storFile(Allcirc, 'Allcirc.csv')
# 构建AllDRUG
Allmi = []
counter1 = 0
while counter1 < len(data): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(Allmi):  #遍历AllDisease
        if data[counter1][0] != Allmi[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif data[counter1][0] == Allmi[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            break
    if flag == 0:
        Allmi.append(data[counter1][0])
    counter1 = counter1 + 1
print('len(Allmi)', len(Allmi))
# storFile(Allmi,'Allmi.csv')

# 挑选正负样本
# 挑选正/负例
import random
counter1 = 0    # 在疾病中随机选择
counter2 = 0    # 在rna中随机选择
counterP = 0    # 正样本数量
counterN = 0    # 负样本数量
PositiveSample = []     # rna - disease 对
# 若正例为全部的RNA-Disease对
PositiveSample = data
print('PositiveSample)', len(PositiveSample))
# storFile(PositiveSample, 'PositiveSample.csv')
# # 负样本为全部的disease-drug（313*593）中随机抽取，未在内LncDisease即为负样本
NegativeSample = []
counterN = 0
while counterN < len(PositiveSample):                         # 当正负样本任一小于10时执行循环，10用来测试，应与正样本数目相同，len(PositiveSample)！！！！！！！！！！！！！！！！！！！！
    counterD = random.randint(0, len(Allcirc)-1)
    counterR = random.randint(0, len(Allmi)-1)     # 随机选出一个疾病rna对
    DiseaseAndRnaPair = []
    DiseaseAndRnaPair.append(Allmi[counterR])
    DiseaseAndRnaPair.append(Allcirc[counterD])
    flag1 = 0
    counter = 0
    while counter < len(data):
        if DiseaseAndRnaPair == data[counter]:
            flag1 = 1
            break
        counter = counter + 1
    if flag1 == 1:
        continue
    flag2 = 0
    counter1 = 0
    while counter1 < len(NegativeSample):
        if DiseaseAndRnaPair == NegativeSample[counter1]:
            flag2 = 1
            break
        counter1 = counter1 + 1
    if flag2 == 1:
        continue
    if (flag1 == 0 & flag2 == 0):
        NegativePair = []
        NegativePair.append(Allmi[counterR])
        NegativePair.append(Allcirc[counterD])
        NegativeSample.append(NegativePair)
        counterN = counterN + 1
print('len(NegativeSample)', len(NegativeSample))
storFile(NegativeSample, '关系对-负样本.csv')

