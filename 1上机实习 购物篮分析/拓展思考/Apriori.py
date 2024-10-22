import numpy

# 加载数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

'''创建集合C1即对dataSet去重排序'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset表示冻结的set 集合，元素无改变把它当字典的 key 来使用
    # return C1
    return map(frozenset, C1)

dataset = loadDataSet()
C1 = createC1(dataset)
print(dataset)