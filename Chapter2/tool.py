import random
import math

"""
从 http://files.grouplens.org/datasets/movielens/ml-1m.zip 下载data
使用 ml-1m/ratings.dat 的前两列 UserID::MovieID
- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
"""
def loadRatingData():
    data = open("ml-1m/ratings.dat").readlines()  # UserID::MovieID::Rating::Timestamp
    data = [tuple(map(int, s.split("::")[:2])) for s in data]
    return data

def splitData(lineRate=1.0, M=8, k=0, seed=1):
    data = loadRatingData()
    test, train = {}, {}
    random.seed(seed)
    for user, item in data:
        if random.randint(0, M - 1) == k:  # randint -> [a, b]
            if user not in test:
                test[user] = set()
            test[user].add(item)
        else:
            if user not in train:
                train[user] = set()
            train[user].add(item)
    # 缩减数据
    if lineRate != 1:
        userList = random.sample(train.keys(), int(lineRate*len(train)))
        train = {user: set(random.sample(train[user], int(lineRate*len(train[user])))) for user in userList}
        test = {user: set(random.sample(test[user], int(lineRate*len(test[user])))) for user in userList if user in test}
    return train, test

def splitDataMock():
    train = {1: {1, 3},
             2: {1, 2, 3, 4},
             3: {3, 4}}
    test = {1: {3, 4}}
    return train, test

class Evaluator:
    """
    train, test: dict(int: set())
    topN: int
    GetRecommendation: function()
    rankList: dict(int: list[tuple(int, int)])
    """
    def __init__(self, train, test, rankList, topN=-1, dataScale=-1, ratio=-1, loopN=-1):
        self.train = train
        self.test = test
        self.rankList = rankList
        self.topN = topN
        self.dataScale = dataScale
        self.ratio = ratio
        self.loopN = loopN
        itemSet = set()
        for items in train.values():
            itemSet.update(items)
        for items in test.values():
            itemSet.update(items)
        self.itemSet = itemSet

    # recall 和 precision 应该针对test集的数据进行分析
    def recall(self):
        all = 0
        hit = 0
        for user in self.test.keys():
            # 在取少部分值进行推荐时，可能会有rank为空的情况，直接跳过
            if len(self.rankList[user]) == 0:
                continue
            Tu = self.test[user]
            for item in self.rankList[user]:
                if item[0] in Tu:
                    hit += 1
            all += len(Tu)
        return hit / float(all)

    def precision(self):
        all = 0
        hit = 0
        for user in self.test.keys():
            if len(self.rankList[user]) == 0:
                continue
            Tu = self.test[user]
            for item in self.rankList[user]:
                if item[0] in Tu:
                    hit += 1
            all += len(self.rankList[user])
        return hit / float(all)

    def coverage(self):
        recommendItem = set()
        allItems = set()
        for user in self.train.keys():
            if user not in self.rankList or len(self.rankList[user]) == 0:
                continue
            allItems.update(self.train[user])
            for item in self.rankList[user]:
                recommendItem.add(item[0])
        return len(recommendItem) / float(len(allItems))

    def popularity(self):
        itemPopularity = dict()
        for items in self.train.values():
            for item in items:
                if item not in itemPopularity:
                    itemPopularity[item] = 0
                itemPopularity[item] += 1
        for items in self.test.values():
            for item in items:
                if item not in itemPopularity:
                    itemPopularity[item] = 0
                itemPopularity[item] += 1
        ret = 0
        n = 0
        for user in self.test.keys():
            if len(self.rankList[user]) == 0:
                continue
            for item in self.rankList[user]:
                if item[0] not in self.itemSet:
                    continue
                ret += math.log(1 + itemPopularity[item[0]])
                n += 1
        ret /= n
        return ret

    def show(self):
        recall = self.recall()
        precision = self.precision()
        coverage = self.coverage()
        popularity = self.popularity()
        if self.dataScale != -1:
            print("dataScale:", self.dataScale, end=' ')
        if self.topN != -1:
            print("topN:", self.topN, end=' ')
        if self.ratio != -1:
            print("ratio:", self.ratio, end=' ')
        if self.loopN != -1:
            print("loopN:", self.loopN, end=' ')
        print("precision:", precision, "recall:", recall, "coverage:", coverage, "popularity:", popularity)
        return recall, precision, coverage, popularity
