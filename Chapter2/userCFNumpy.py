from numpy import *
import tool

def splitDataNumpy(M, k, seed=1):
    train, test = tool.splitData(M, k, 1, seed)
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
    similarMatrix = matmul(trainM, trainM.T)
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
