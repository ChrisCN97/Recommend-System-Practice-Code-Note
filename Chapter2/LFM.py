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
    retSet = set(ret)  # in 查询在 set 里更快
    itemNum = len(retSet)
    # 直接多找一点然后删除重复的，强求达到ratio不现实
    item = random.choice(itemsPool, int(itemNum * ratio * 3), p=itemsPop)
    item = [x for x in item if x not in retSet][:int(itemNum * ratio)]
    ret.update({x: 0 for x in item})
    return ret

# 随机梯度下降，适合线上，线下运行慢
def latentFactorModel(train, ratio, N, F=100, alpha=0.02, lamb=0.01):
    itemsPool, itemsPop = initModel(train)
    P, Q = {}, {}
    for user in train:
        P[user] = random.random(F)
    for item in itemsPool:
        Q[item] = random.random(F)
    print("随机梯度下降...")
    aErr = 0.0
    total = 1
    for step in range(N):
        for user, items in train.items():
            samples = randSelNegSample(items, ratio, itemsPool, itemsPop)
            for item, rui in samples.items():
                eui = rui - float(matmul(P[user], Q[item]))
                aErr = (aErr+eui)/total
                total += 1
                print("\rstep: %d/%d  error: %10.10f averageErr: %10.10f" % (step+1, N, abs(eui), abs(aErr)), end='')
                P[user] += alpha*(Q[item]*eui-lamb*P[user])
                Q[item] += alpha*(P[user]*eui-lamb*Q[item])
        alpha *= 0.7
    print()
    return P, Q

def recommend(user, P, Q, topN, train):
    rank = {}
    n = 1
    trainLen = len(train)
    for i in Q:
        print("\r%d/%d" % (n, trainLen), end='')
        n += 1
        if i not in train[user]:
            rank[i] = matmul(Q[i], P[user])
    print()
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
        rankList[user] = recommend(user, P, Q, topN, train)
    print("进行评估...")
    eva = Evaluator(train, test, rankList, ratio=ratio, dataScale=lineRate, loopN=N)
    eva.show()

def test():
    import time
    d = {1:1, 2:2, 3:3}
    l = len(d)
    start = time.clock()
    for i in range(10000):
        a = len(d)
    print(time.clock() - start)
    start = time.clock()
    for i in range(10000):
        a = l
    print(time.clock() - start)

#test()
# ratio, N, lineRate
lfmRecommend(1, 10, 0.1)
"""
dataScale: 0.5 ratio: 1 loopN: 1 precision: 0.0008846487424111015 recall: 0.0008181205685135872 coverage: 0.7415324819544697 popularity: 2.240384139936817
"""
