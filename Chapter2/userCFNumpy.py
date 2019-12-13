from numpy import *
import tool

def splitDataNumpy(M, k, seed=1):
    train, test = tool.splitData(1, M, k, seed)
    m = 6040
    n = 3952
    trainM = mat(zeros((m, n)))  # 最好使用默认float类型进行矩阵计算
    for user, items in train.items():
        for item in items:
            trainM[user-1, item-1] = 1
    return trainM, train, test

def userSimilarityNumpy(trainM):
    sorted_user_sim = dict()
    m, n = trainM.shape
    NM = sum(trainM, axis=1)
    NM = sqrt(matmul(NM, NM.T))  # math.sqrt(N[u]*N[v])
    similarMatrix = matmul(trainM, trainM.T) / NM
    sortedM = argsort(similarMatrix)
    for user in range(m):
        sorted_user_sim[user+1] = [x + 1 for x in sortedM[user, :].tolist()[0][::-1]]
    return sorted_user_sim

# sim: dict{int: list[]}
def recommendNumpy(user, train, sim, userN, itemN):
    rank = dict()
    if user not in sim or user not in train:
        return rank
    interacted_items = train[user]
    n = 1
    for v in sim[user][1:userN+1]:  # 把自己去掉
        for vItem in train[v]:
            if vItem not in interacted_items:
                if vItem not in rank:
                    rank[vItem] = 0
                rank[vItem] += userN/float(n)  # 因为矩阵计算没有计算相似度，只进行了排序，所以用排名的反比来做rank权重
        n += 1
    recs = list(sorted(rank.items(), key=lambda x: x[1], reverse=True))[:itemN]
    return recs

def userCFNumpy(itemN=10):
    print("加载数据...")
    trainM, train, test = splitDataNumpy(8, 0)
    print("计算用户兴趣相似度...")
    W = userSimilarityNumpy(trainM)
    print("进行推荐...")
    K = [5, 10, 20, 40, 80, 160]
    for k in K:
        print("K:", k)
        rankList = dict()
        userNum = len(test)
        un = 1
        # 对test集的user进行推荐
        for user in test.keys():
            print("\r%d/%d" % (un, userNum), end='')
            un += 1
            rankList[user] = recommendNumpy(user, train, W, k, itemN)
        print("\n进行评估...")
        eva = tool.Evaluator(train, test, rankList, k, 1)
        eva.show()

userCFNumpy()
"""
topN: 5 precision: 0.14962355696837878 recall: 0.08061840800504823 coverage: 0.5733225108225108 popularity: 6.780254317956431
topN: 10 precision: 0.17661033963526854 recall: 0.09515910934823762 coverage: 0.49053030303030304 popularity: 6.904756735421595
topN: 20 precision: 0.19364229546595282 recall: 0.10433606779049852 coverage: 0.4375 popularity: 6.981826484742841
topN: 40 precision: 0.20522001003848084 recall: 0.11057423600468765 coverage: 0.3944805194805195 popularity: 7.046129228883746
topN: 80 precision: 0.21311694830182365 recall: 0.1148291715496259 coverage: 0.3476731601731602 popularity: 7.102158228460846
topN: 160 precision: 0.21872176677262842 recall: 0.11784909402325791 coverage: 0.3106060606060606 popularity: 7.151814575059906
"""