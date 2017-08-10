# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:18:00 2017

@author: wencheng
"""

# -*- coding:utf-8 -*-
'''
手写数字识别
'''
import csv
from sklearn import neighbors
#导入训练数据和测试数据
def loadData(filename1,filename2,trainDataSet,trainTargetSet,testDataSet):
    with open(filename1,'r') as csvfile1:
        lines1 = csv.reader(csvfile1)
        dataSet = list(lines1)
        for x in range(1,len(dataSet)):
            temp = []
            dataSet[x][0] = int(dataSet[x][0])
            trainTargetSet.append(dataSet[x][0])
            for y in range(1,785):
                #if dataSet[x][y] != 0:
                    #dataSet[x][y] = 1
                dataSet[x][y] = int(dataSet[x][y])
                temp.append(dataSet[x][y])
            trainDataSet.append(temp)
    with open(filename2,'r') as csvfile2:  
        lines2 = csv.reader(csvfile2)
        dataSet2 = list(lines2)
        for x in range(1,len(dataSet2)):
            temp = []
            for y in range(784):
                #if dataSet2[x][y] != 0:
                    #dataSet2[x][y] = 1
                dataSet2[x][y] = int(dataSet2[x][y])
                temp.append(dataSet2[x][y])
            testDataSet.append(temp)
    return trainDataSet,trainTargetSet,testDataSet

#将结果保存为csv文件用于在kaggle网站提交
def saveResult(result): 
#结果保存的路径 
    with open(r'C:\Users\wencheng\Desktop\DigitR\result2.csv','w',newline='') as myFile:      
        myWriter=csv.writer(myFile)
        x=0 
        #myWriter.writerow(["ImageId","Label"])
        for i in result:  
            x += 1
            tmp=[x]  
            tmp.append(i)  
            myWriter.writerow(tmp)


def main():
    trainDataSet = []
    trainTargetSet = []
    testDataSet = []   
    print("开始加载数据")
    #训练数据和测试数据的路径
    loadData(r'C:\Users\wencheng\Desktop\DigitR\train.csv', r'D:\digit\test.csv', trainDataSet, trainTargetSet, testDataSet)
    knn = neighbors.KNeighborsClassifier()
    print("数据加载完毕，开始训练模型")
    knn.fit(trainDataSet,trainTargetSet)
    print("模型训练完毕，开始预测")
    prediction = knn.predict(testDataSet)
    print("预测结果:", prediction)
    print("打印完毕，开始保存")
    saveResult(prediction)
    print("保存完毕")


if __name__ == '__main__':
    main()