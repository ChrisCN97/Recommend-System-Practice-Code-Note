from tool import splitData, Evaluator
from numpy import *

# Batch梯度下降
def latentFactorModel(train, N, alpha=0.02, r=0.9, F=100, lamb=0.01):
    uNum = 6040
    iNum = 3952
    P = mat(random.random((uNum, F)))-0.5
    Q = mat(random.random((iNum, F)))-0.5
    R = mat(zeros((uNum, iNum)))
    m = 0
    for user, items in train.items():
        for i in items:
            R[user-1, i-1] = 1
            m += 1
    print("批量梯度下降...")
    for step in range(N):
        E = multiply((1 - P * Q.T), R)
        err = abs(sum(E)/m)
        print("\rstep: %d/%d  error: %.5g" % (step + 1, N, err), end='')
        Pt = alpha * (E * Q / m - lamb * P)
        Qt = alpha * (E.T * P / m - lamb * Q)
        P += Pt
        Q += Qt
        alpha *= r
    print()
    Pre = P * Q.T
    sortedPre = argsort(Pre)
    sortedUserItem = {}
    for user in range(uNum):
        sortedUserItem[user+1] = [x + 1 for x in sortedPre[user, :].tolist()[0][::-1]]
    return sortedUserItem

def lfmRecommend(N, alpha, rate, topN=10):
    print("加载数据...")
    train, test = splitData(1)
    print("生成参数...")
    sortedUserItem = latentFactorModel(train, N, alpha, rate)
    print("进行推荐...")
    rankList = dict()
    itemSet = set()
    for items in train.values():
        itemSet.update(items)
    for items in test.values():
        itemSet.update(items)
    for user in test.keys():
        if user not in train:
            rankList[user] = {}
            continue
        rankList[user] = [(item, 1) for item in sortedUserItem[user] if item not in train[user] and item in itemSet][:topN]
    print("进行评估...")
    eva = Evaluator(train, test, rankList, loopN=N)
    eva.show()

# N, alpha, rate, topN
lfmRecommend(1, 0.02, 0.9, 50)
"""
0.02, 0.9: loopN: 50 precision: 0.005223482321547699 recall: 0.012545262280898516 coverage: 0.34444444444444444 popularity: 4.253319369702654
"""
