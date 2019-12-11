import math
import tool

def userSimilarity(train):
    item_users = dict()
    print("生成物品-用户倒排表...")
    for user, items in train.items():
        for i in items:
            if i not in item_users:
                item_users[i] = set()
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
            if u not in N:
                N[u] = 0
            N[u] += 1
            for v in users:
                if u != v:
                    if u not in C:
                        C[u] = dict()
                    if v not in C[u]:
                        C[u][v] = 0
                    C[u][v] += 1 / math.log(1 + len(users))
    print("\n计算用户兴趣相似度...")
    for u, row in C.items():
        for v, cuv in row.items():
            C[u][v] = cuv / math.sqrt(N[u]*N[v])
    sorted_user_sim = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in C.items()}
    return sorted_user_sim

# 注意相似用户和推荐物品都是取排名靠前的item，不然就太多了
def recommend(user, train, W, userN, itemN):
    rank = dict()
    if user not in W or user not in train:
        return rank
    interacted_items = train[user]
    for v, wuv in W[user][:userN]:
        for vItem in train[v]:
            if vItem not in interacted_items:
                if vItem not in rank:
                    rank[vItem] = 0
                rank[vItem] += wuv * 1
    recs = list(sorted(rank.items(), key=lambda x: x[1], reverse=True))[:itemN]
    return recs

def userCF(userN, itemN, dataScale):
    print("加载数据...")
    train, test = tool.splitData(8, 0, dataScale)
    rankList = dict()
    W = userSimilarity(train)
    print("进行推荐...")
    # 对test集的user进行推荐
    for user in test.keys():
        rankList[user] = recommend(user, train, W, userN, itemN)
    print("进行评估...")
    eva = tool.Evaluator(train, test, rankList, userN, dataScale)
    eva.show()

# userN, itemN, dataScale
# 取前N个相关用户，取前N个推荐物品，数据取dataScale%
userCF(80, 10, 0.5)
#dataScale: 0.1 topN: 8 precision: 0.003925417075564278 recall: 0.012327773749093546 coverage: 0.7647241165530069 popularity: 4.132750171483042
#dataScale: 0.2 topN: 8 precision: 0.009711165756679031 recall: 0.021062146892655367 coverage: 0.7249124854142357 popularity: 4.988955540712396
#dataScale: 0.2 topN: 80 precision: 0.016363636363636365 recall: 0.035796610169491525 coverage: 0.23561882626380012 popularity: 5.793893530988032
#dataScale: 0.5 topN: 80 precision: 0.05416666666666667 recall: 0.05580405454151937 coverage: 0.18857459789240155 popularity: 6.717603555313228
