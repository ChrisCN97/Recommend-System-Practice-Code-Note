from numpy import *

# Root Mean Square Error 均方根误差
def RMSE(records):
    return sqrt(sum(square(records[:, 2] - records[:, 3]))) / len(records)

# Mean Absolute Error 平均绝对误差
def MAE(records):
    return sum(abs(square(records[:, 2] - records[:, 3]))) / len(records)

def GiniIndex(pA):
    n = len(pA)
    sortedPIndex = argsort(pA)
    jn = 2*arange(1, n+1)-n-1
    return sum(jn*pA[sortedPIndex])/(n-1)

def test():
    records = mat([[1, 1, 1, 2],
                   [2, 1, 3, 3],
                   [2, 1, 4, 4],
                   [2, 1, 5, 2]])
    print("RMSE:", RMSE(records))
    print("MAE:", MAE(records))
    pA = array([0.2, 0.3, 0.1, 0.4])
    print("GiniIndex:", GiniIndex(pA))

test()
