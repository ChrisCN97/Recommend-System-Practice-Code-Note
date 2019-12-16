from numpy import *
import tool

def initData(train):
    item_user = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_user:
                item_user[item] = set()
            item_user[item].add(user)
    i2rU = list(train.keys())
    i2rI = list(item_user.keys())
    uNum = len(i2rU)
    iNum = len(i2rI)
    r2iU = {u: i for i, u in enumerate(i2rU)}
    r2iI = {u: i+uNum for i, u in enumerate(i2rI)}
    v = uNum + iNum
    M = mat(zeros((v, v)))
    for user in i2rU:
        for item in train[user]:
            M[r2iU[user], r2iI[item]] = 1 / float(len(item_user[item]))
    for item in item_user:
        for user in item_user[item]:
            M[r2iI[item], r2iU[user]] = 1 / float(len(train[user]))
    return M, i2rI, r2iU, uNum

def personalRankNumpy(M, root, alpha):
    v = shape(M)[0]
    R = mat(zeros((v, 1)))
    R[root] = 1
    return (1 - alpha) * linalg.inv(eye(v) - alpha * M) * R

def recommend(userI, M, alpha, iStart, i2rI, topN, trainItem):
    R = personalRankNumpy(M, userI, alpha).T
    sortedRIndex = argsort(-R[0, iStart:]).tolist()[0]
    itemR = [i2rI[x] for x in sortedRIndex]
    rec = []
    for item in itemR[:topN]:
        if item not in trainItem:
            rec.append((item, 1))
    return rec

def main(lineRate, alpha, topN, testN):
    print("加载数据...")
    train, test = tool.splitData(lineRate)
    # train, test = tool.splitDataMock()
    M, i2rI, r2iU, iStart = initData(train)
    rankList = dict()
    print("进行推荐...")
    tN = 0
    testNew = dict()
    for user in test:
        if tN == testN:
            break
        tN += 1
        print("\r%d/%d" % (tN, testN), end='')
        testNew[user] = test[user]
        if user not in train:
            rankList[user] = []
        rankList[user] = recommend(r2iU[user], M, alpha, iStart, i2rI, topN, train[user])
    eva = tool.Evaluator(train, testNew, rankList, dataScale=lineRate, topN=topN)
    eva.show()


def test():
    i2rU = [1,3,2,4]
    print({u: i for i, u in enumerate(i2rU)})

# lineRate, alpha, topN, testN
main(0.1, 0.8, 50, 5)


# [[0.31390666 0.16583748 0.07581142 0.15873016 0.18905473 0.0331675,  0.06349206]]
# {'A': 0.269336111714077, 'B': 0.18507057054096726, 'C': 0.08626975015801294, 'a': 0.1537324937273993, 'b': 0.039296071244436064, 'c': 0.19036571254903534, 'd': 0.07592929006607209}