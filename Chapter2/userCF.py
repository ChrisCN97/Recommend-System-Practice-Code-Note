import math
import tool
import time

def userSimilarity(train):
    item_users = dict()
    print("生成物品-用户倒排表...")
    for user, items in train.items():
        for i in items:
            item_users[i] = item_users.get(i, set())
            item_users[i].add(user)
    C = dict()
    N = dict()
    print("建立相似度矩阵...")
    item_usersNum = len(item_users)
    usern = 1
    for users in item_users.values():
        print("\r%d/%d" % (usern, item_usersNum), end="")
        usern += 1
        for u in users:
            N[u] = N.get(u, 0) + 1
            for v in users:
                if u != v:
                    C[u] = C.get(u, dict())
                    C[u][v] = C[u].get(v, 0)
                    C[u][v] += 1
    W = dict()
    print("\n计算用户兴趣相似度...")
    for u, row in C.items():
        for v, cuv in row.items():
            W[u] = W.get(u, dict())
            W[u][v] = W[u].get(v, 0)
            W[u][v] = cuv / math.sqrt(N[u]*N[v])
    return W

def recommend(user, train, W, topN):
    rank = dict()
    interacted_items = train[user]
    if user not in W:
        return rank
    for v, wuv in sorted(W[user].items(), key=lambda d: d[1], reverse=True)[:topN]:
        for vItem in train[v]:
            if vItem not in interacted_items:
                rank[vItem] = rank.get(vItem, 0.0) + wuv * 1
    return rank

def userCF():
    print("加载数据...")
    train, test = tool.splitData(8, 0, 0.5)
    rankList = dict()
    W = userSimilarity(train)
    print("进行推荐...")
    topN = 20
    for user in train.keys():
        rankList[user] = recommend(user, train, W, topN)
    eva = tool.Evaluator(train, test, rankList, topN)
    eva.show()

def test():
    data = [12]
    print(data[:3])

userCF()
# test()

# topN: 3 recall: 0.008504478422147833 precision: 0.004368435728227531 coverage: 0.7762917933130699 popularity: 4.151294907813066
# topN: 10 recall: 0.03675205938263782 precision: 0.004129331475473195 coverage: 0.9418923030118649 popularity: 4.10443959415352
# topN: 20 recall: 0.09428674742361237 precision: 0.0038995027479717353 coverage: 0.9911314984709481 popularity: 4.067470309262547