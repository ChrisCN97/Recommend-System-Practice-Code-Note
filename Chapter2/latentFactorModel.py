from tool import splitData, Evaluator
from numpy import *

def initModel(train):
    items = {}
    total = 0
    for iSet in train.values():
        for i in iSet:
            if i not in items:
                items[i] = 0
            items[i] += 1
            total += 1
    itemsPool = list(items.keys())
    itemsPop = array(list(items.values())) / total
    return itemsPool, itemsPop

def randSelNegSample(items, ratio, itemsPool, itemsPop):
    ret = dict()
    for i in items:
        ret[i] = 1
    itemNum = len(ret)
    # 直接多找一点然后删除重复的，强求达到ratio不现实
    item = random.choice(itemsPool, int(itemNum * ratio * 3), p=itemsPop)
    item = [x for x in item if x not in ret][:int(itemNum * ratio)]
    ret.update({x: 0 for x in item})
    return ret

def latentFactorModel(train, ratio, N, F=100, alpha=0.02, lamb=0.01):
    itemsPool, itemsPop = initModel(train)
    P, Q = {}, {}
    for user in train:
        P[user] = random.random(F)
    for item in itemsPool:
        Q[item] = random.random(F)
    print("随机梯度下降...")
    for step in range(N):
        print("\r%d/%d" % (step+1, N), end='')
        for user, items in train.items():
            samples = randSelNegSample(items, ratio, itemsPool, itemsPop)
            for item, rui in samples.items():
                eui = rui - matmul(P[user], Q[item])
                P[user] += alpha*(Q[item]*eui-lamb*P[user])
                Q[item] += alpha*(P[user]*eui-lamb*Q[item])
        alpha *= 0.9
    print()
    return P, Q

def recommend(user, P, Q, topN):
    rank = {}
    for i in Q:
        rank[i] = matmul(Q[i], P[user])
    return list(sorted(rank.items(), key=lambda x: x[1], reverse=True))[:topN]

def lfmRecommend(ratio, N, lineRate, topN=10):
    print("加载数据...")
    train, test = splitData(lineRate)
    print("生成参数...")
    P, Q = latentFactorModel(train, ratio, N)
    rankList = dict()
    print("进行推荐...")
    for user in test.keys():
        if user not in train:
            rankList[user] = {}
            continue
        rankList[user] = recommend(user, P, Q, topN)
    print("进行评估...")
    eva = Evaluator(train, test, rankList, ratio=ratio)
    eva.show()

def test():
    d = {2:0,3:0,4:0}
    print(len(d))

#test()
lfmRecommend(1, 2, 0.1)
