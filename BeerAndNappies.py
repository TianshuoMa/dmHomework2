import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

#打开文件
def openFile(filename):
    data = pd.read_csv(filename)
    return data

#五数概括
def checkFiveNumbers(data, attr):
    fiveNumbers = {}
    fiveNumbers["Min"] = data[attr].min()
    fiveNumbers["Q1"] = data[attr].quantile(q=0.25)
    fiveNumbers["Median"] = data[attr].median()
    fiveNumbers["Q3"] = data[attr].quantile(q=0.75)
    fiveNumbers["Max"] = data[attr].max()
    return fiveNumbers

def processData(data):
    # 清理数据，滤除缺失数据与删除重复数据
    # 去除某些属性，将DataFrame转化为List
    print("数据清理前：", data.shape)
    data.dropna(how = "any", inplace = True)
    print("滤除缺失数据：", data.shape)
    data = data.drop_duplicates()
    print("删除重复数据：", data.shape)
    normData = data.drop(["Unnamed: 0","description","designation","country","region_1","region_2","winery"], axis = 1)
    normData1 = normData.drop(normData[normData["variety"] != "Pinot Noir"].index)
    normData2 = normData.drop(normData[normData["variety"] != "Chardonnay"].index)
    normData3 = normData.drop(normData[normData["variety"] != "Cabernet Sauvignon"].index)
    normData4 = normData.drop(normData[normData["variety"] != "Red Blend"].index)
    normData = pd.concat([normData1,normData2,normData3,normData4])
    normData.to_csv("normData.csv",index=False,sep=',')
    wineList = normData.values.tolist()
    return wineList, normData

def replaceFiveNumbers(wineList, fiveNumbers, index):
    # 对List的数值属性进行分类，1~4
    for i in range(len(wineList)):
        temp = wineList[i][index]
        if temp >= fiveNumbers['Min'] and temp < fiveNumbers['Q1']:
            wineList[i][index] = str(index) + '4'
        elif temp >= fiveNumbers['Q1'] and temp < fiveNumbers['Median']:
            wineList[i][index] = str(index) + '3'
        elif temp >= fiveNumbers['Median'] and temp < fiveNumbers['Q3']:
            wineList[i][index] = str(index) + '2'
        elif temp >= fiveNumbers['Q3'] and temp <= fiveNumbers['Max']:
            wineList[i][index] = str(index) + '1'
    return wineList

def checkFrequency(data, attr):
    return data[attr].value_counts().to_dict()

def plot_metrics_relationship(rule_matrix, col1, col2):
    fit = np.polyfit(rule_matrix[col1], rule_matrix[col2], 1)
    fit_funt = np.poly1d(fit)
    plt.plot(rule_matrix[col1], rule_matrix[col2], 'yo', rule_matrix[col1], 
    fit_funt(rule_matrix[col1]))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('{} vs {}'.format(col1, col2))
    plt.show()

def generateFrequentItemsets(transact_items_matrix, rule_type, min_support):
    start_time = 0
    total_time = 0
    if rule_type == "apriori":
        frequent_itemsets=apriori(trans_encoder_matrix, min_support, use_colnames=True)
        total_time = time.time() - start_time
    elif rule_type == "fpgrowth":
        frequent_itemsets=fpgrowth(trans_encoder_matrix, min_support, use_colnames=True)
        total_time = time.time() - start_time
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets.to_csv('frequent_itemset_{}.csv'.format(rule_type))
    print("执行{}算法，结果已保存至frequent_itemset_{}.csv".format(rule_type, rule_type))
    # 过滤单个商品
    # frequent_itemsets[frequent_itemsets['length']>1]
    return frequent_itemsets, total_time

if __name__=="__main__":
    filename = 'winemag-data_first150k.csv'
    data = openFile(filename)

    wineList, normData = processData(data)
    # 输出列表每列为:points|price|province|variety
    # 对数值属性使用五数概括分级
    points5Numbers = checkFiveNumbers(normData, "points")
    price5Numbers = checkFiveNumbers(normData, "price")
    print("points的五数概括：", points5Numbers)
    print("price的五数概括：", price5Numbers)
    wineList = replaceFiveNumbers(wineList, points5Numbers, 0)
    wineList = replaceFiveNumbers(wineList, price5Numbers, 1)
    print(wineList[:5])
    # The following instructions transform the dataset into the required format 
    trans_encoder = TransactionEncoder() # Instanciate the encoder
    trans_encoder_matrix = trans_encoder.fit(wineList).transform(wineList)
    trans_encoder_matrix = pd.DataFrame(trans_encoder_matrix, columns = trans_encoder.columns_)
    print("\n")
    print(trans_encoder_matrix.head())

    # Apriori算法
    itemsets_apriori, time_apriori = generateFrequentItemsets(trans_encoder_matrix, "apriori", min_support=0.1)
    apriori_rules_lift = association_rules(itemsets_apriori, metric="lift", min_threshold=1)
    apriori_rules_lift = apriori_rules_lift.sort_values(by='lift', ascending=False)
    apriori_rules_lift.to_csv("apriori_rules_lift.csv")
    apriori_rules_confidence = association_rules(itemsets_apriori, metric="confidence", min_threshold=0.2)
    apriori_rules_confidence = apriori_rules_confidence.sort_values(by='confidence', ascending=False)
    apriori_rules_confidence.to_csv("apriori_rules_confidence.csv")
    plot_metrics_relationship(apriori_rules_lift, col1='lift', col2='confidence')

    # Fp Growth算法
    itemsets_fpgrowth, time_fpgrowth =  generateFrequentItemsets(trans_encoder_matrix, "fpgrowth", min_support=0.1)
    fpgrowth_rules_lift = association_rules(itemsets_fpgrowth, metric="lift", min_threshold=1)
    fpgrowth_rules_lift = fpgrowth_rules_lift.sort_values(by='lift', ascending=False)
    fpgrowth_rules_lift.to_csv("fpgrowth_rules_lift.csv")
    fpgrowth_rules_confidence = association_rules(itemsets_fpgrowth, metric="confidence", min_threshold=0.2)
    fpgrowth_rules_confidence = fpgrowth_rules_confidence.sort_values(by='confidence', ascending=False)
    fpgrowth_rules_confidence.to_csv("fpgrowth_rules_confidence.csv")
    plot_metrics_relationship(fpgrowth_rules_lift, col1='lift', col2='confidence')