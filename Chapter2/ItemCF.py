import tool
import math

def itemSimilartiy(train):
    retW = dict()
    N = dict()
    print("建立相似度矩阵...")
    userNum = len(train)
    usern = 1
    for items in train.values():
        print("\r%d/%d" % (usern, userNum), end="")
        usern += 1
        for i in items:
            if i not in N:
                N[i] = 0
                retW[i] = dict()
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                if j not in retW[i]:
                    retW[i][j] = 0
                retW[i][j] += 1
    print()
    for i, row in retW.items():
        for j in row.keys():
            retW[i][j] /= math.sqrt(N[i] * N[j])
    return {i: sorted(its.items(), key=lambda x: x[1], reverse=True) for i, its in retW.items()}

def recommendation(train, user, W, K, topItem):
    rank = dict()
    if user not in train or user not in W:
        return rank
    ru = train[user]
    # 根据train集的item找最相似的item，然后排序
    for i in ru:
        for j, pi in W[i][:K]:
            if j in ru:
                continue
            if j not in rank:
                rank[j] = 0
            rank[j] += pi * 1
    retrank = list(sorted(rank.items(), key=lambda x: x[1], reverse=True))[:topItem]
    return retrank

def itemCF(K, topItem, dataScale):
    print("加载数据...")
    train, test = tool.splitData(dataScale, 8, 0)
    rankList = dict()
    W = itemSimilartiy(train)
    print("进行推荐...")
    # 对test集的user进行推荐
    for user in test.keys():
        rankList[user] = recommendation(train, user, W, K, topItem)
    print("进行评估...")
    eva = tool.Evaluator(train, test, rankList, topN=K, dataScale=dataScale)
    eva.show()

itemCF(10, 10, 0.2)
"""
dataScale: 0.2 topN: 10 precision: 0.007485250737463127 recall: 0.016302601991647927 coverage: 0.6437234687689509 popularity: 3.436129421519991
dataScale: 0.5 topN: 10 precision: 0.05238235294117647 recall: 0.05352044955975599 coverage: 0.2254566210045662 popularity: 6.5938219398442754
dataScale: 1 topN: 10 precision: 0.20636612021857922 recall: 0.1108192969070955 coverage: 0.17234744365035734 popularity: 7.258922340084761
"""