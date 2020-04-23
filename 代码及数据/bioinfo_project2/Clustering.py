from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

file_path = "差异基因.csv"
#print(type(data))
for k in range(3,7):
    #数据读入，预处理
    data = pd.read_table(file_path, header=0, index_col=0, sep=",")
    #去掉Gene.symbol，Gene.title列
    data.drop(['Gene.symbol','Gene.title'], axis=1,inplace=True)
    column_list = list(data)
    #数据标准化
    data_zs = (data-data.mean())/data.std()
    #设置Kmeans最大迭代次数
    interation = 500

    #调用Kmeans模型
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=interation)
    model.fit(data_zs)

    #打印聚类结果
    r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
    r = pd.concat([r2, r1], axis = 1) #横向连接(0是纵向), 得到聚类中心对应的类别下的数目
    r.columns = list(data.columns) + [u'类别数目'] #重命名表头
    print(r)
    #详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
    r.columns = list(data.columns) + [u'聚类类别'] #重命名表头

    #TSNE降维及可视化
    tsne = TSNE()
    tsne.fit_transform(data_zs) #进行数据降维
    tsne = pd.DataFrame(tsne.embedding_, index = data_zs.index) #转换数据格式
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    #不同类别用不同颜色和样式绘图
    plt.figure(k-2)
    if k==3:
        d = tsne[r[u'聚类类别'] == 0]
        plt.plot(d[0], d[1], 'r.')
        d = tsne[r[u'聚类类别'] == 1]
        plt.plot(d[0], d[1], 'go')
        d = tsne[r[u'聚类类别'] == 2]
        plt.plot(d[0], d[1], 'b*')
        plt.show()

    elif k==4:
        d = tsne[r[u'聚类类别'] == 0]
        plt.plot(d[0], d[1], 'r.')
        d = tsne[r[u'聚类类别'] == 1]
        plt.plot(d[0], d[1], 'go')
        d = tsne[r[u'聚类类别'] == 2]
        plt.plot(d[0], d[1], 'b*')
        d = tsne[r[u'聚类类别'] == 3]
        plt.plot(d[0], d[1], 'kv')
        plt.show()

    elif k==5:
        d = tsne[r[u'聚类类别'] == 0]
        plt.plot(d[0], d[1], 'r.')
        d = tsne[r[u'聚类类别'] == 1]
        plt.plot(d[0], d[1], 'go')
        d = tsne[r[u'聚类类别'] == 2]
        plt.plot(d[0], d[1], 'b*')
        d = tsne[r[u'聚类类别'] == 3]
        plt.plot(d[0], d[1], 'kv')
        d = tsne[r[u'聚类类别'] == 4]
        plt.plot(d[0], d[1], 'y+')
        plt.show()

    elif k==6:
        d = tsne[r[u'聚类类别'] == 0]
        plt.plot(d[0], d[1], 'r.')
        d = tsne[r[u'聚类类别'] == 1]
        plt.plot(d[0], d[1], 'go')
        d = tsne[r[u'聚类类别'] == 2]
        plt.plot(d[0], d[1], 'b*')
        d = tsne[r[u'聚类类别'] == 3]
        plt.plot(d[0], d[1], 'kv')
        d = tsne[r[u'聚类类别'] == 4]
        plt.plot(d[0], d[1], 'y+')
        d = tsne[r[u'聚类类别'] == 5]
        plt.plot(d[0], d[1], 'cd')
        plt.show()

    # 打印平均轮廓系数
    s = silhouette_score(np.array(data_zs), model.labels_)
    print("When cluster= {}\nThe silhouette_score= {}".format(k, s))
