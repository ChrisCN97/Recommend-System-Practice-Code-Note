import random
import math

"""
从 http://files.grouplens.org/datasets/movielens/ml-1m.zip 下载data
使用 ml-1m/ratings.dat 的前两列 UserID::MovieID
- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
"""
def loadRatingData(lineRate=-1):
    data = open("ml-1m/ratings.dat").readlines()  # UserID::MovieID::Rating::Timestamp
    lineNum = len(data)
    data = [tuple(map(int, s.split("::")[:2])) for s in data]
    # 取少部分数据进行测试
    if lineRate != -1:
        data = random.sample(data, int(lineRate*lineNum))  # 随机抽取 lineNum 个数据
    return data

def list2dic(L):
    retDic = dict()
    for item in L:
        retDic[item[0]] = retDic.get(item[0], set())
        retDic[item[0]].add(item[1])
    return retDic

def splitData(M, k, lineRate=-1, seed=1):
    data = loadRatingData(lineRate)
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0, M) == k:
            test.append([user, item])
        else:
            train.append([user, item])
    return list2dic(train), list2dic(test)

class Evaluator:
    """
    train, test: dict(int: set())
    topN: int
    GetRecommendation: function()
    rankList: dict(int: dict(int: int))
    """
    def __init__(self, train, test, rankList, topN):
        self.train = train
        self.test = test
        self.rankList = rankList
        self.topN = topN

    def recall(self):
        all = 0
        hit = 0
        for user in self.train.keys():
            if user not in self.test:
                continue
            Tu = self.test[user]
            Ru = set(self.rankList[user].keys())
            hit += len(Tu & Ru)
            all += len(Tu)
        return hit / float(all)

    def precision(self):
        all = 0
        hit = 0
        for user in self.train.keys():
            if user not in self.test:
                continue
            Tu = self.test[user]
            Ru = set(self.rankList[user].keys())
            hit += len(Tu & Ru)
            all += len(Ru)
        return hit / float(all)

    def coverage(self):
        recommendItem = set()
        allItems = set()
        for user in self.train.keys():
            allItems.update(self.train[user])
            recommendItem.update(list(self.rankList[user].keys()))
        return len(recommendItem) / float(len(allItems))

    def popularity(self):
        itemPopularity = dict()
        for items in self.train.values():
            for item in items:
                itemPopularity[item] = itemPopularity.get(item, 0) + 1
        ret = 0
        n = 0
        for user in self.train.keys():
            for item in self.rankList[user].keys():
                ret += math.log(1 + itemPopularity[item])
                n += 1
        ret /= n
        return ret

    def show(self):
        recall = self.recall()
        precision = self.precision()
        coverage = self.coverage()
        popularity = self.popularity()
        print("topN:", self.topN, "recall:", recall, "precision:", precision, "coverage:", coverage, "popularity:", popularity)
        return recall, precision, coverage, popularity

def test():
    train, test = splitData(8, 0)
    rankList = dict()
    eva = Evaluator(train, test, rankList)


# test()
