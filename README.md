# Recommend System Practice Code & Note
项亮《推荐系统实践》的Python3代码实现，原书的code只是帮助理解，实际运行效率较低。我尝试重写这些代码，可能的话使用numpy。
> Start on 2019.12.9

## 笔记
### 第一章 好的推荐系统
> 这一章主要是文字描述，对推荐系统和它的评价方式进行了说明
* 推荐系统的三个参与方：用户、物品提供者、提供推荐系统的网站
* 评测指标：
1. 用户满意度
2. 预测准确度：RMSE，MAE
3. 覆盖率：系统对物品长尾的发掘能力：信息熵，基尼系数
4. 多样性
5. 新颖性：不推荐已经交互过的物品
6. 惊喜度（serendipity，歪打正着）：推荐跟用户习惯不符但用户喜欢的物品
7. 信任度：用户对系统的信任程度
8. 时效性
9. 健壮性：防水军能力
10. 商业目标：赚钱
* 还可以加入用户类型，物品类型，时间段这三种分类维度来使用不同的推荐系统

### 第二章 利用用户行为数据
> [Code Data](http://files.grouplens.org/datasets/movielens/ml-1m.zip)

#### userCF 基于用户的协同过滤算法（基于邻域）
1. 通过用户物品列表的相似度，计算用户相似度
2. 根据最相似的用户们所喜爱的物品，进行推荐
3. 对结果进行评估
> 随机推荐的准确度：0.631%；只推荐热门信息的准确度：12.79%
* userCF.py：更改了原书明显不合理的地方，增加了随机抽取部分Data进行debug的功能，但计算效率比较低，跑完整数据非常耗时
* userCFNumpy.py：使用numpy重构，更改最耗时的相似度计算部分。直接生成userNum * itemNum的矩阵A，保存user-item的记录，利用A * A.T来计算user-user发生关联的次数，在把关联次数除以sqrt(N1 * N2)。6040 * 3952的矩阵算的还是很快的。最后结果也和书上差不多
* userCF重视用户圈子，用户的兴趣爱好，适合用户变化比物品慢的系统（新闻）

#### itemCF 基于物品的协同过滤
1. 通过是否被同一群人交互过，来计算物品的相似度
2. 根据用户交互过的物品列表和物品相似度，推出推荐列表
* itemCF.py: item比user少，所以运行的能快一点，结果和书上一致，就没做numpy版
* itemCF重视个性化，适合豆瓣，netflix这种物品变化比用户慢的系统

#### LFM（隐语义模型）
* 对物品自动归类，自动生成用户对各种类的喜爱程度
* Prediction(u, i) = Puk * Qik (k是自动归类的类型数量)
* 使用随机梯度下降法来学习参数
* LFM.py：运行极慢，感觉无法再快了
* LFMBGD：尝试使用BGD进行优化，没有成功。参考[这篇文章](https://blog.csdn.net/fjssharpsword/article/details/78257126)，可能还需要使用sigmoid函数，在负样例生成方面进行改进。

#### 基于图的模型
* PersonalRank算法：随机游走，计算从user开始，停在每个item上的几率，从而得出关联性，进而推荐
* personalRank.py personalRankNumpy.py：都很慢

### 第三章 推荐系统冷启动问题
* 冷启动问题：用户/物品/系统冷启动
* 解决方案：
1. 提供非个性化推荐：热门排行榜
2. 利用注册信息：人口统计学信息（demographic）
3. 利用其它关联账户的历史数据
4. 利用用户的兴趣描述：首次登录时选择感兴趣的物品，利用决策树寻找呈现给新用户的最有分类能力的物品
5. 利用物品自身的内容特征：LDA话题模型
6. 引入专家知识

### 第四章 利用用户标签数据
* UGC(User Generated Content)中的标签应用
