import re
import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import data_clean

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def save_topic_xlsx(df, labels):
    df = data_clean(df)
    for label in labels:
        df_label = df.loc[df['label']==label]
        df_label.to_excel('./data/附件3.xlsx')

def get_ORG_PER(df):
    pass


def comput_tfidf(df):
    df_texts = np.array(df['留言详情'])
    df_texts = [" ".join(text) for text in df_texts]

    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    tfidf=transformer.fit_transform(vectorizer.fit_transform(df_texts))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    
    sort = np.argsort(tfidf.toarray())#[:, 43501:43741] # 将二维数组中每一行按升序排序，并提取每一行的最后5个(即数值最大的五个)
    weight = np.sort(tfidf.toarray())#[:, 43501:43741]
    
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语

    # weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    # data_tfidf = []
    
    # for i in tqdm(range(len(sort))):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
    #     data = []
    #     for j in sort[i]:
    #         data.append(word[j])
    #     data_tfidf.append(data)
    
    # wf = open('./data/tfidf附件3.txt', 'w', encoding='utf-8')
    # for index, data in enumerate(data_tfidf):
    #     text = ','.join(data)
    #     wf.write("第%d类"%index+'\t'+text)
    #     wf.write('\n')
    # wf.close()

    # SSE = []  # 存放每次结果的误差平方和 
    # classsumlist = []
    # num_k = 500
    # for k in range(1,num_k):  
    #     estimator = KMeans(n_clusters=k)  # 构造聚类器  
    #     estimator.fit(weight)
    #     SSE.append(estimator.inertia_)
    #     #计算类间距离和
    #     classsum = 0.0
    #     centers = estimator.cluster_centers_.tolist()
    #     for center1 in centers:
    #         for center2 in centers:
    #             classsum = classsum + abs(center1[0] - center2[0])
    #     classsumlist.append(classsum) 
    # X = range(1,num_k)
    # plt.xlabel('k')  
    # plt.ylabel('SSE')  
    # plt.plot(X,SSE)
    # plt.plot(X,classsumlist)
    # plt.savefig('./images/SSE.png')
    # plt.close()
    return weight

def comput_sse():
    pass

def get_timestamp(date):
    return datetime.datetime.strptime(date, "%Y/%m/%d %H:%M:%S").timestamp()

def k_means(weight,clusters, title):
    clf=KMeans(n_clusters=clusters)
    y=clf.fit(weight)
 
    #每个样本所属的簇
    label = []               
    i = 1
    while i <= len(clf.labels_):
        label.append(clf.labels_[i-1])
        i = i + 1
        
    y_pred = clf.labels_
    pca = PCA(n_components=2)             #输出两维
    newData = pca.fit_transform(weight)   #载入N维
    
    xs, ys = newData[:, 0], newData[:, 1]
    
    df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=title)) 
    groups = df.groupby('label')
    
    fig, ax = plt.subplots(figsize=(8, 5)) # set size
    ax.margins(0.02)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='.', linestyle='', ms=10, mec='none')
     
    plt.savefig('./images/cluster_result_%d.png'%clusters)
    plt.close()
    return df

def caculate_cluster():
    # 计算tf-idf
    df_1 = pd.read_excel('./data/附件3.xlsx')
    df_1 = data_clean(df_1)
    weight = comput_tfidf(df_1)

    title = df_1['留言编号'].to_list()
    clusters = 1500
    df = k_means(weight,clusters, title)
    df = df.drop(['x', 'y'], axis=1)
    df.to_excel('./data/cluster_result.xlsx', index=None)

def plot_cluster_dis(x, y):
    plt.bar(x, y)
    plt.savefig('./images/class.png')
    plt.close()

def time2integr(time_str):
    time1 = time_str.split(' ')[0]
    time2 = time_str.split(' ')[-1]
    try:
        time2 = ''.join(time2.split(':'))
        time1 = ''.join(time1.split('/'))
    finally:
        time1 = ''.join(time1.split('-'))
    time = int(time1 + time2)
    return time

def compute_hot_value():
    data_df = pd.read_excel('./data/cluster_result.xlsx')
    origin_data_df = pd.read_excel('./data/附件3.xlsx')

    # print(origin_data_df.head())

    label_list = data_df['label'].unique()
    index_list = data_df['title'].to_list()

    # print(data_df.loc[data_df['label'] == 927]['title'].to_list())
    # plot_cluster_dis(list(data_df.index), data_df.to_list())
    
    datasets = dict()
    for label in label_list:
        datasets[label] = data_df.loc[data_df['label'] == label]['title'].to_list()

    # print(datasets)

    hot_topic_value = dict()
    for label, index in datasets.items():
        num_index = len(index)   # x_4
        time_ = [] # x_1
        time_cross = 0 # x_1
        num_good = 0 # x_2
        num_inver = 0 # x_3
        for i in index:
            num_good += origin_data_df.loc[origin_data_df['留言编号'] == i]['点赞数'].values[0]
            num_inver += origin_data_df.loc[origin_data_df['留言编号'] == i]['反对数'].values[0]
            time_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['留言时间'].values[0])
        time_ = [time2integr(str(n)) for n in time_]
        if len(time_) == 1:
            time_cross = 0
        else:
            time_cross = (max(time_) - min(time_)) // 8640000000
        hot_value = 0.1*time_cross + 0.2*num_good + 0.1*num_inver + 0.6*num_index
        hot_topic_value[label] = round(hot_value, 3)

    # 热点问题留言明细表
    hot_topic = sorted(datasets.items(), key=lambda d:d[0], reverse=False)
    data_table = []
    print('热点问题留言明细表.....')
    for q_id, statement_id in tqdm(hot_topic):
        for i in statement_id:
            data_ = []
            data_.append(q_id)
            data_.append(i)
            data_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['留言用户'].values[0])
            data_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['留言主题'].values[0])
            data_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['留言时间'].values[0])
            data_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['留言详情'].values[0].strip())
            data_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['点赞数'].values[0])
            data_.append(origin_data_df.loc[origin_data_df['留言编号'] == i]['反对数'].values[0])
            data_table.append(data_)
    
    # print(hot_topic)
    hot_topic_table_columns = ['问题ID', '留言编号', '留言用户', '留言主题', '留言时间', '留言详情', '点赞数', '反对数']
    data_table_df = pd.DataFrame(data_table, columns=hot_topic_table_columns)
    data_table_df.to_excel("./data/热点问题留言明细表.xlsx", index=None)

    # 取排名前5的热点问题

    # 字典按值排序
    hot_columns = ['热度排名', '问题ID', '热度指数', '时间范围', '地点/人群', '问题描述']
    hot_topic_sort = sorted(hot_topic_value.items(), key=lambda d:d[1], reverse=True)
    hot_5data = []
    rank = 1
    for q_id, hot_value in hot_topic_sort[:5]:
        data_ = []
        data_.append(rank)
        data_.append(q_id)
        data_.append(hot_value)
        data_.append('t')  # 这些值由人工填写，下一步可以考虑自动摘要。
        data_.append('t')
        data_.append('t')
        hot_5data.append(data_)
        rank += 1
    hot_5data_table = pd.DataFrame(hot_5data, columns=hot_columns)
    hot_5data_table.to_excel('./data/热点问题表.xlsx', index=None)




if __name__ == "__main__":
    caculate_cluster()
    compute_hot_value()