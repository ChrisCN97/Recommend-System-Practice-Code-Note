# Recommend System Practice Code & Note
项亮《推荐系统实践》的Python3代码实现，原书的code只是帮助理解，实际运行效率较低。我尝试重写这些代码。
> Start on 2019.12.9

## 笔记
### 第一章
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
### 第二章
#### userCF 基于用户的协同过滤算法（基于邻域）
1. 通过用户物品列表的相似度，计算用户相似度
2. 根据最相似的用户们所喜爱的物品，进行推荐
3. 对结果进行评估
> [Code Data](http://files.grouplens.org/datasets/movielens/ml-1m.zip)

* 更改了原书明显不合理的地方，增加了随机抽取部分Data进行debug的功能
* 感觉无法矩阵化，计算效率比较低
