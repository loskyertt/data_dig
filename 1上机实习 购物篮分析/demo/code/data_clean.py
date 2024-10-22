# -*- coding: utf-8 -*-

# 代码8-5 数据转换

import pandas as pd

def data_translation() -> list:
    data = pd.read_csv("1上机实习 购物篮分析/demo/data/GoodsOrder.csv")
    # 根据id对“Goods”列合并，并使用“，”将各商品隔开
    data['Goods'] = data['Goods'].apply(lambda x:','+x)
    data = data.groupby('id').sum().reset_index()

    # 对合并的商品列转换数据格式
    data['Goods'] = data['Goods'].apply(lambda x :[x[1:]])
    data_list = list(data['Goods'])

    # 分割商品名为每个元素
    translation_list = []
    for i in data_list:
        p = i[0].split(',')
        translation_list.append(p)

    return translation_list

if __name__ == "__main__":
    data_list = data_translation()
    print(data_list[0: 5])
