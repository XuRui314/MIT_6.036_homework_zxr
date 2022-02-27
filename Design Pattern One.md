> Background: 把 `1-7` 章的note结合视频看了，补充资料也找了，并且把  `1-5` HW都做了，然后来整理一下目前的收获。

主要还是谈谈关于`Supervised Learning`设计和处理问题的Paradigm：

1. **Problem class:** What is the nature of the training data and what kinds of queries will be made at testing time?

2. **Assumptions:** What do we know about the source of the data or the form of the solution?

3. **Evaluation criteria:** What is the goal of the prediction or estimation system? How will the answers to individual queries be evaluated? How will the overall performance of the system be measured?

4. **Model type:** Will an intermediate model be made? What aspects of the data will be modeled? How will the model be used to make predictions?

5. **Model class:** What particular parametric class of models will be used? What criterion will we use to pick a particular model from the model class?

6. **Algorithm:** What computational process will be used to fit the model to the data and/or to make predictions?

这个是mit的总结，写的也是很不错，但是如果没有具体的实操经验，看起来也是有些空洞的，所以接下来也会细化这些内容。

大体来说就是**`Analysis` + `Pattern`**，根据具体问题，具体需求去分析，然后再利用设计模板去实现。

- `Analysis`的部分就是理论的分析，这个也是靠积累和尝试。而且注意，这个analysis前后是有关联的，并不是完全线性，也就是所谓的整体性。

- `Pattern`指代码的思路，实现的过程。比如说传统的`Gradient Descent`，一系列的Training & Evaluation的流程，其实都会有比较固定的模板。



## Analysis

### Data

`feature representation` 在note上已经写的很清楚了，这里不再赘述。

这里强调说一下`scaling/standardization`对learning algorithm的影响：

[link](https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35)

**对GD的影响**

<img src="https://miro.medium.com/max/750/1*yi0VULDJmBfb1NaEikEciA.png" alt="img" style="zoom: 80%;" />



**Explanation** ：At every iteration, SGD takes a step in the direction opposite to the (stochastic) gradient, with the same step size in each "direction" of parameter space. However, with raw features, you would naturally want larger step sizes to update parameters which have a larger range, and smaller ones for those which have a small range. Using a large step size can then often lead to divergence of SGD due to the parameters with a smaller range; using a small step size can lead to very slow convergence due to those parameters with a larger range. Standardizing features largely removes this problem, making it "more acceptable" to use a constant step size across all directions.

具体可见[Week_5](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week5/week5_homework/?child=first) Predicting mpg values

### Model

模型的选择也是很重要，涉及到`structure error`，比如潜在的数据分布是正弦函数，结果用线性去预测，那肯定会存在很大的结构误差，所以对于模型也要河里选择。



### Learning

#### Gradient Descent

主要的也就是矩阵求导啥的了，我的建议是，可以从单个数据入手，然后再扩展至求和求平均的情况，标量对矩阵求导本质也是多元函数求导。



### Evaluation

大体分为两个板块：

1. 具体形式的目标函数和损失函数的设计
2. 如何选取最优参数和超参数

**第一个板块**需要根据想要的效果去分析设计，same Model type, different Evaluation criteria。比如看到了感知机算法的局限后，可以选择`logistic regression`优化模型，然后选用`log loss(cross entropy)`作为损失函数，或者针对于`margin`的大小，设计对应的`hinge loss`损失函数，这也就是SVM的思路。

此外还有正则化，`structure error`和`estimation error`，看[Week_5](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week5/week5_homework/?child=first)的source of error部分。

1. `estimation error` measures how close we can get to the optimal linear predictions with limited training data
2. `structural error` measures the error introduced by the limited function class (infinite training data)

**第二个板块**肯定是看mit的[Week_5](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week5/week5_homework/?child=first)的Evaluation part，堪称经典。
简而言之就是$\lambda^*$ focus on the future behavior, just cost function considered; $\theta^*$ is always with the structure error with the constraint on $\theta$. ($*$ means optimal)



## Pattern

### Data

**Convention:** 

1. `2D` arrays to represent both matrices(batch of data) and vectors(single data, **label**)
2. data vectors are represented as col vector

`Polynomial`处理：待补完



standardization处理后，值得注意的是，对于线性，计算损失函数的时候，是要相应的还原，毕竟standard只是为了方便得到结果，最终的参数和不去standard还是一样的。



### Learning

#### Perceptron

这个没啥好说的，就按照算法来就行，判断+更新。

#### Gradient Descent

1. 写出**目标函数导数**的chain rule形式，如果有必要也可以加上**目标函数**的对应chain rule形式
2. 套用`gd or sgd`的模板即可



```python
def gd(f, df, x0, step_size_fn, max_iter):
    prev_x = x0
    fs = []; xs = []
    for i in range(max_iter):
        prev_f, prev_grad = f(prev_x), df(prev_x)
        fs.append(prev_f); xs.append(prev_x)
        if i == max_iter-1:
            return prev_x, fs, xs
        step = step_size_fn(i)
        prev_x = prev_x - step * prev_grad
        
        
        
def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    n = y.shape[1]
    prev_w = w0
    fs = []; ws = []
    np.random.seed(0)
    for i in range(max_iter):
        j = np.random.randint(n)
        Xj = X[:,j:j+1]; yj = y[:,j:j+1]
        prev_f, prev_grad = J(Xj, yj, prev_w), dJ(Xj, yj, prev_w)
        fs.append(prev_f); ws.append(prev_w)
        if i == max_iter - 1:
            return prev_w, fs, ws
        step = step_size_fn(i)
        prev_w = prev_w - step * prev_grad
```



**Example**

```Python
def d_lin_reg_th(x, th, th0):
    return x

def d_square_loss_th(x, y, th, th0):
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th(x, th, th0)

def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, th0), axis = 1, keepdims = True)

def d_lin_reg_th0(x, th, th0):
    return np.ones((1, x.shape[1]))

def d_square_loss_th0(x, y, th, th0):
    return -2 * (y - lin_reg(x, th, th0)) * d_lin_reg_th0(x, th, th0)

def d_mean_square_loss_th0(x, y, th, th0):
    return np.mean(d_square_loss_th0(x, y, th, th0), axis= 1, keepdims = True)

def d_ridge_obj_th(x, y, th, th0, lam):
    return d_mean_square_loss_th(x, y, th, th0) + 2 * lam * th

def d_ridge_obj_th0(x, y, th, th0, lam):
    return d_mean_square_loss_th0(x, y, th, th0)

def ridge_obj_grad(x, y, th, th0, lam):
    grad_th = d_ridge_obj_th(x, y, th, th0, lam)
    grad_th0 = d_ridge_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])
# --------------------------------------------------------------------------#


def ridge_min(X, y, lam):
    """ Returns th, th0 that minimize the ridge regression objective
    Assumes that X is NOT 1-extended. Interfaces to our sgd by 1-extending
    and building corresponding initial weights.
    """
    def svm_min_step_size_fn(i):
        return 0.01/(i+1)**0.5

    d, n = X.shape
    X_extend = np.vstack([X, np.ones((1, n))])
    w_init = np.zeros((d+1, 1))

    def J(Xj, yj, th):
        return float(ridge_obj(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam))

    def dJ(Xj, yj, th):
        return ridge_obj_grad(Xj[:-1,:], yj, th[:-1,:], th[-1:,:], lam)

    np.random.seed(0)
    w, fs, ws = sgd(X_extend, y, J, dJ, w_init, svm_min_step_size_fn, 1000)
    return w[:-1,:], w[-1:,:]
```



### Evaluation

对于学习算法的Evaluation目前用的就是两种：

1. Evaluating a learning algorithm using a data source
2. Evaluating a learning algorithm with a fixed dataset

这两种都是十分抽象的封装，也就是直接把idea编程化，基本相当于API来用，只要提供**数据，标签，超参数以及specific model evaluation的function**即可。

```python
def xval_learning_alg(X, y, lam, k):
    _, n = X.shape
    # randomize the data 
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    X, y = X[:,idx], y[:,idx]

    split_X = np.array_split(X, k, axis=1)
    split_y = np.array_split(y, k, axis=1)

    score_sum = 0
    for i in range(k):
        X_train = np.concatenate(split_X[:i] + split_X[i+1:], axis=1)
        y_train = np.concatenate(split_y[:i] + split_y[i+1:], axis=1)
        X_test = np.array(split_X[i])
        y_test = np.array(split_y[i])
        score_sum += eval_predictor(X_train, y_train, X_test, y_test, lam)
    return score_sum/k
```



其实关键就是对于specific model的Evaluation，也就是**通过学习算法先求出对应参数**，**然后选择相应的评价函数算出score**。

这里需要注意的是，评价时用到的函数是并不一定与目标函数一致，一般是**损失函数**，也就是不会带正则化项。

下面用`linear Classifier`和`linear regression`的`eval_predictor`说明：

```python
# linear classifier
def score(data, labels, th, th0):
    return (np.sum(labels == np.sign(th.T.dot(data)+th0)))

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0)/data_test.shape[1]
# SVM
def hinge(v):
    return np.where(v >= 1, 0, 1 - v)


# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    temp = y * (th.T @ x + th0)
    return hinge(temp)


# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_score(x, y, th, th0, lam):
    temp = hinge_loss(x, y, th, th0)

    (d, n) = x.shape

    return np.sum(temp) / n 

# linear regression
def eval_predictor(X_train, Y_train, X_test, Y_test, lam):
    th, th0 = ridge_min(X_train, Y_train, lam)
    return np.sqrt(mean_square_loss(X_test, Y_test, th, th0))
```



### Picking Hyperparameters

由于超参数不会因为训练而改变，实际上也是确定超参后再去训练，所以对于超参数的选择是要进行尝试和比较的。

下面linear regression为例，提供了不同的值范围，选择最优的$\lambda$。

```python
#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
order = [1,2,3]
lambda_2 = np.arange(0.01,0.11,0.01)
lambda_3 = np.arange(20,420,20)
#Your code for cross-validation goes here
#Make sure to scale the RMSE values returned by xval_learning_alg by sigma,
#as mentioned in the lab, in order to get accurate RMSE values on the dataset
score_lst = {}
for i in order:
    if i==3:
        lambda_range = lambda_3
    else:
        lambda_range = lambda_2
    for lam in lambda_range:
        for j in range(2):
            score = xval_learning_alg(make_polynomial_feature_fun(i)(auto_data[j]), auto_values, lam, 10)
            rmse = (score * sigma)[0][0]
            #[feature_set, order, lambda]
            score_lst[rmse] = [j+1, i, lam]
```



## Numpy

> 做一个比较完整的局部sum捏

待补完



numpy.vstack











## Theory

> 这部分还是处于没有自己手推总结的状态，但是后续一定会补上来的捏。

#### Loss function

MSE, Cross Entropy 和 Hinge loss 的比较 - akon的文章 - 知乎 https://zhuanlan.zhihu.com/p/158448293

常见的损失函数(loss function)总结 - yyHaker的文章 - 知乎 https://zhuanlan.zhihu.com/p/58883095



#### Regularization

**在经验风险最小化的基础上（也就是训练误差最小化），尽可能采用简单的模型，以此提高泛化预测精度。**



#### Information Theory

交叉熵(`cross entropy`)



https://www.bilibili.com/video/BV15V411W7VB?t=1094.5

https://zhuanlan.zhihu.com/p/165139520







#### Optimization

**SVM**

> basic knowledge

**Part 1 Understanding of Lagrange multipliers**



> Q: Since $h_i(x) = 0$, why do we still add it?

ANS: Because what we focus on is gradient, which may not be 0. 



 https://www.zhihu.com/question/58584814/answer/159863739



**Part 2 Convex Optimization**

the definition of convex set and convex optimization problem

https://www.bilibili.com/video/BV1HP4y1Y79e?t=1139.7



**Part 3 Understanding of Lagrange duality** 



[(42条消息) 对偶函数求解（一）_yu132563的专栏-CSDN博客_对偶函数](https://blog.csdn.net/yu132563/article/details/111402607)

https://www.cnblogs.com/90zeng/p/Lagrange_duality.html

 https://www.zhihu.com/question/300015357/answer/715354302



> Q: Why Dual?

ANS: Because of convex: https://blog.csdn.net/weixin_44273380/article/details/109034549

Get an intuitive understanding from the picture below:

<img src="https://s2.loli.net/2022/02/20/SVT42p1I3vfxXoO.png" style="zoom:80%;" />

strong duality v.s. weak duality

https://www.bilibili.com/video/BV1HP4y1Y79e?t=1565.5



**Part 4 Slater / KKT**

https://www.bilibili.com/video/BV1HP4y1Y79e?t=2215.5



**Part 5 soft margin and hard margin**

https://zhuanlan.zhihu.com/p/94214743





[支持向量机(SVM)解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/66789668?utm_source=wechat_session)

https://www.youtube.com/watch?v=6-ntMIaJpm0

https://www.bilibili.com/video/BV1HP4y1Y79e?spm_id_from=333.999.0.0

https://blog.csdn.net/qq_41995574/article/details/89676894



**Optimal sum**

https://zhuanlan.zhihu.com/p/100041443

https://www.jianshu.com/p/52aeaa540d25?utm_campaign

#### Regularization

王大佬的视频

https://space.bilibili.com/504715181/?spm_id_from=333.999.0.0





<img src="https://s2.loli.net/2022/02/27/FDIrT8WhzpeiHBy.png" style="zoom: 67%;" />