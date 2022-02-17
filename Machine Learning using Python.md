> Just record the codes implement without using the library functions.
>
> One algorithm One day.

> Courses: 
>
> [MIT_6.036](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/course/)
>
> [Python](https://pythonprogramming.net/machine-learning-tutorials/)
>
> [Stanford_cs231n](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=4)



```mit
MIT's CAT~

ML            
/    /\__/\     mio~ 
\__=(  o_O )= 
(__________) 
 |_ |_ |_ |_ 
```





# Preparation Knowledge

> [Click here!](https://github.com/XuRui314/MIT_6.036_homework_zxr/blob/main/MIT_6_036_HW01_Linear_Classfier.ipynb)







# Supervised Learning

## No Model

### KNN

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')

def k_nearest(dataset, test_point, k = 3):
    if len(dataset) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []

    for group in dataset:
        for feature_point in dataset[group]:
            euclidean_distance = sqrt((test_point[0] - feature_point[0] )**2 + (test_point[1] - feature_point[1] )**2)
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    # votes contain the group character ['k', 'k', 'r']
    # print(Counter(votes).most_common(1))
    # You can see the Counter method also contains the #votes, so we use[0] to get['k', 2]
    # and [0][0] to get 'k'

    return vote_result



dataset = {'k': [[10, 20], [20, 30], [30, 10]], 'r': [[60, 50], [70, 45], [80, 90]]}

test_point = [20, 70]

[[plt.scatter(ii[0],ii[1],s=100,color= i) for ii in dataset[i]] for i in dataset]

# the same as:
# for i in dataset:
#     for j in dataset[i]:
#         plt.scatter(j[0],j[1],s=100,color= i)
# i represents the 'k' and 'r'

prediction = k_nearest(dataset, test_point, k = 3)
print("prediction is", prediction)

plt.scatter(test_point[0], test_point[1], s = 100, marker = '+', color = prediction)

plt.show()
```

<img src="https://s2.loli.net/2022/01/19/xK7rW6mZHIaPRkY.png" style="zoom: 80%;" />





## Linear classification

> The same model, with the same prediction rule: $sign( \theta^{\mathsf T}\mathsf x + \theta_0 )$
>
> And i will introduce some learning algorithm below:

### Perceptron

[download lab](https://introml_oll.odl.mit.edu/cat-soop/_static/6.036/homework/hw02/code_for_hw02_downloadable.zip)



**python**

```python
# 1. transpose is not equal to reshape(col, row)

# 2. understand broadcast

# 3. split & concatenate
ji = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
ji = np.array_split(ji, 2,, axis = 0) # axis = 0 means each row
output: a list
[array([[1, 2],
       [3, 4],
       [5, 6]]), 
 array([[ 7,  8],
       [ 9, 10]])]


j2 = np.concatenate(j, axis = 0) # 参数不一定是()tuple，包住sequence of array即可
output: a matrix(ndarray) 
j2 is the same as ji

# 4. list & matrix addation
list addition:
	j = ji[0:1] + ji[1:2]
output: the same as ji
[array([[1, 2],
       [3, 4],
       [5, 6]]), 
 array([[ 7,  8],
       [ 9, 10]])]

matrix addition:  
	j2 = ji[0:1, :] + ji[2:4, :]
output: broadcast
    [[ 6  8]
 	[ 8 10]]

    
# 4. mulitplication
C = np.array([[1,2,3,4]])
C*C
output: array([[ 1,  4,  9, 16]])
    mulitplied entry by entry
```

`1.` [真正理解 transpose 和 reshape - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/72128096)

`2.` [numpy broadcast](https://numpy.org/doc/stable/user/basics.broadcasting.html)





> Notation convention: mit lab uses col vector, and the matrix is represented as $d$ x $n$，$d$ is the dimension of data vectors and $n$ is the number of vectors 

**Implement perceptron**

```python
import numpy as np

# x is dimension d by 1
# th is dimension d by 1
# th0 is dimension 1 by 1
# return 1 by 1 matrix of +1, 0, -1
def positive(x, th, th0):
   return np.sign(th.T@x + th0)

# Perceptron algorithm with offset.
# data is dimension d by n
# labels is dimension 1 by n
# T is a positive integer number of steps to run
# Perceptron algorithm with offset.
# data is dimension d by n
# labels is dimension 1 by n
# T is a positive integer number of steps to run
def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0
```





**Implement averaged perceptron**

Regular perceptron can be somewhat sensitive to the most recent examples that it sees. Instead, averaged perceptron produces a more stable output by outputting the average value of `th` and `th0` across all iterations. 

```python
import numpy as np
# row representation
def averaged_perceptron(data, labels, params = {}, hook = None):
       # if T not in params, default to 100
    T = params.get('T', 100)

    # Your implementation here
    (d, n) = data.shape
    th = np.zeros((1, d)); ths = np.zeros((1, d)) # 2-dim
    th0 = np.zeros((1, 1)); th0s = np.zeros((1, 1)) # 2-dim
    data = data.T # row representation

    for t in range(T):
        for i in range(n):
            yi = labels[:, i : i + 1] # 2-dim
            xi = data[i:i+1, :] # 2-dim
            '''
            also can be written as
            yi = labels[0, i] # 1-dim
            xi = data[i] # 1-dim
            '''
            if yi * (xi@th.T + th0) <= 0.0:
                th += yi * xi
                th0 += yi
                if hook: hook((th, th0))
            ths += th
            th0s += th0

    return (ths / (n * T)).T, th0s / (n * T)
```



**Evaluating a classifier**

- `data`: a d by n array of floats (representing n data points in d dimensions)
- `labels`: a 1 by n array of elements in (+1, -1), representing target labels
- `th`: a d by 1 array of floats that together with
- `th0`: a single scalar or 1 by 1 array, represents a hyperplane

```python
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    (d, t) = data_test.shape
    th = np.zeros((d, 1)); th0 = np.zeros((1, 1))
    th, th0 = learner(data_train, labels_train)
    ans = 0
    
    for i in range(t):
        x = data_test[:,i:i+1]
        y = labels_test[:,i:i+1]
        if(np.sign(th.T@x + th0) == y):
            ans = ans + 1
    
    return float(ans) / k
```



**Evaluating a learning algorithm using a data source **

Returns a scalar number of data points that the separator correctly classified.

The eval_classifier function should accept the following parameters:

- `learner` - a function, such as perceptron or averaged_perceptron
- `data_train` - training data
- `labels_train` - training labels
- `data_test` - test data
- `labels_test` - test labels

and returns the percentage correct on a new testing set as a float between 0. and 1.

```python

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    ans = 0
    for i in range(it):
        (data_train, labels_train) = data_gen(n_train)
        (data_test, labels_test) = data_gen(n_test)
        an = eval_classifier(learner, data_train, labels_train, data_test, labels_test)
        ans += an
        
    return ans / it
```



**The difference between evaluating the classifier and the learning algorithm**

One classifier is just one specific result from the learning algorithm. To evaluate the learning algorithm, we can choose to average over a set of test data.





**Evaluating a learning algorithm using a data source**

- `data_gen` - a data generator, call it with a desired data set size; returns a tuple (data, labels)
- `it` - the number of iterations to average over

```python
def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    ans = 0
    for i in range(it):
        (data_train, labels_train) = data_gen(n_train)
        (data_test, labels_test) = data_gen(n_test)
        an = eval_classifier(learner, data_train, labels_train, data_test, labels_test)
        ans += an
        
    return ans / it
```





**Evaluating a learning algorithm with a fixed dataset**

```python
def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k
```





### SVM

#### brute force

[svm_basic](https://pythonprogramming.net/predictions-svm-machine-learning-tutorial/?completed=/svm-optimization-python-2-machine-learning-tutorial/)

[一起学ML之支持向量机，一个SVM算法的Python实现（2）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV19E411c7B5?from=search&seid=16239299400440471946&spm_id_from=333.337.0.0)

```python
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # train
    def fit(self, data):
        self.data = data

        # {||w||: [w, b]}
        opt_dict = {}

        # w is first set with angle 45 degree
        rotMatrix = lambda theta: np.array([np.sin(theta), np.cos(theta)])

        theta_step = np.pi / 10
        transforms = [np.array(rotMatrix(theta))
                        for theta in np.arange(0, np.pi, theta_step)]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)


        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # starts getting very high cost after this
                      self.max_feature_value * 0.001]

        # extremely expensive
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                    1 * (self.max_feature_value * b_range_multiple),
                                    step * b_multiple):
                    for transform in transforms:
                        w_t = w * transform
                        found_option = True

                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("+1 optimized\n")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] * step * 2


    def predict(self, features):
        classification = np.sign(np.dot(self.w, np.array(features)) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s = 200, marker = '*', c = self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 8],[2, 3],[3, 6]]),
             1: np.array([[1, -2],[3, -4],[3, 0]])}



svm =Support_Vector_Machine()
svm.fit(data = data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()
```

<img src="https://s2.loli.net/2022/01/23/IjNAGhmgzQJo3F9.png" style="zoom:80%;" />





#### Kernel method

\# loading...





## Network classification

### NN(neural network)

> This part may be a little abundant, but each part will be introduced really in detail.

For simplicity but enough to cover the basic ideas of NN, i will just choose the Two Layer NN for the code example.

So, before diving into the neural net construction, let's first set down the main parts and each part' targets.

1.  The understanding **fundamentals of neural net**.
    Including the functions(based on mini-batch) that are used through the building process.

2.  The **preparation of data set**, that is the MINIST data set.

    Including the operations of loading the data set, using the data.

3.  The **implement of structure of NN**, just the structure.

    Including the weight、bias matrix.

4. The **mini-batch method and its implement**.

    Including the design of functions and design of echo.

I'll use python to realize the whole procedure.

As a result of using mini-batch, there may be some methods will be used later that you are not so familiar with.

So now, I'm gonna to introduce them: 

#### Methods

**python basic**

```python
#in python, you should know that
Invariant type: integer, float, long, complex, string, tuple, frozenset
Variant type: list, dictionary, set, numpy array, user defined objects
""""
When we pass a paramter to a function, if a the variable type is variant, the inner change in the function will also cause the outter value change.
""" 


## lambda
def loss(m, n):
    return m - n
x = 3
t = 2
f = lambda w: loss(x, t)
# the same as
def f(w):
    return loss(x, t) # This is calling another function, and x, t are arguments not parameters
ans = f(x) # ans = 1
""" w is only a pseudo paramter, the reason of using lambda is to simplify coding."""

## enumerate()
# iterate through an array
bar = np.array([[1,2],[3, 5],[8,9]])
list(enumerate(bar))
# output: [(0, array([1, 2])), (1, array([3, 5])), (2, array([8, 9]))]

for idx,x in enumerate(bar):
    print(f"idx is {idx}, x is",x)
    
"""
output: idx is 0, x is [1 2]
idx is 1, x is [3 5]
idx is 2, x is [8 9]
"""
```



**numpy basic: shape**

```python
## numpy basic
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.6]])

x.ndim # output: 2
x.shape # output: (4, 3)
x.size # output: 12 numbers of elements
x.argmax(axis = 1) # output: array([1, 2, 1, 0])
""" axis = 1 means in each row, find the max number's index """

x.reshape(1, x.size)
# change x to 2d-vector form, not 1d-vector
# output:
array([[0.1, 0.8, 0.1, 0.3, 0.1, 0.6, 0.2, 0.5, 0.3, 0.8, 0.1, 0.6]])

x.reshape(2, 6)
# output: 
array([[0.1, 0.8, 0.1, 0.3, 0.1, 0.6],
       [0.2, 0.5, 0.3, 0.8, 0.1, 0.6]])
# change x to 2 rows, 6 colunms array, in the order of row index
```



**numpy basic: operation**

```python
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.6]])

# get one single element
x[0, 1] or x[0][1]
# output: 0.8

# get row vecotors
x[[0, 2]]
# output:
array([[0.1, 0.8, 0.1], [0.2, 0.5, 0.3]])

# matrix - 1d vector
# every row minus the vector
x - [0.1, 0.1, 0.1]
# output:
array([[0. , 0.7, 0. ],
       [0.2, 0. , 0.5],
       [0.1, 0.4, 0.2],
       [0.7, 0. , 0.5]])
""" 
The vector must have the same size as the row vector.
In the softmax function, we use the traverse method to avoid the mistake.
"""

# Traverse method
x.T
# output: 
array([[0.1, 0.3, 0.2, 0.8],
       [0.8, 0.1, 0.5, 0.1],
       [0.1, 0.6, 0.3, 0.6]])

# get specific elements 
x <= 0.5
# output:
array([[ True, False,  True],
       [ True,  True, False],
       [ True,  True,  True],
       [False,  True, False]])

x[(x <= 0.5)]
# output:
array([0.1, 0.1, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1])

```



**numpy random**

```python
## numpy random
# np.random.randn: array ramdom generation
x = np.random.randn(2, 4)
# get a 2 row, 4 column array, whose elements are correspond to normal distribution. 

# np.random.choice: randomly pick numbers
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# the same as np.random.choice(60000, 10), pick 10 numbers randomly from 0-59999, and generate an index array.

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```





**numpy.nditer**

```python
## numpy.nditer
it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # type: numpy.nditer

""""
np.nditer is used to iterare through an array, the parameter flags = ['multi_index'] means you can just use it.multi_index to get the index of it[0](current element) in the original array.
""""

# example
x = np.arange(6).reshape(2,3)
it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    print("%d <%s>" % (it[0], it.multi_index))
    it.iternext()

# 0 <(0, 0)>
# 1 <(0, 1)>
# 2 <(0, 2)>
# 3 <(1, 0)>
# 4 <(1, 1)>
# 5 <(1, 2)>
```



#### Data processing

```python
from dataset.mnist import load_mnist
"""
	load MNIST dataset
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
"""
(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
""""
 x_train and t_train are array type
 x_train.shape: (60000, 784)
 t_train.shape: (60000, ) just the 1d-vector
""""

```



#### Functions

Neural network can be used in classification and regression problems, but the output needs to be changed according to the specific situation.

When it comes to the choice of output layer activation function, generally speaking, in the regression problem, we usually use the identity function and in the classification problem  we usually use the softmax function.

[二分类问题，应该选择sigmoid还是softmax?](https://www.zhihu.com/question/295247085/answer/1778398778) 

[sigmoid and softmax](https://zhuanlan.zhihu.com/p/105722023)

**softmax**
$$
y_k= {exp(a_k) \over \sum_{i=1}^n exp(a_i)}= {exp(a_k + C) \over \sum_{i=1}^n exp(a_i + C)}
$$
Because of  the property of $y_k \in [0,1]$ and $\sum_ky_k = 1$, we can interpret $y_k$ as probability. And in practice, we usually use $C$ (often chosen with the maximum number in $y_k$) to avoid overflow.

<img src="https://s2.loli.net/2022/01/16/NUKG2JmAEserZPb.png" style="zoom:50%;" />

> The picture above use the sigmoid function as the activation function,  whereas the below one uses softmax.

<img src="https://s2.loli.net/2022/01/16/baoAPnTRO4qj3dS.png" style="zoom: 50%;" />



**loss function**

mean squared error
$$
E = {1\over2} \sum_k(y_k-t_k)^2
$$
cross entropy error
$$
E = -\sum_kt_kln^{y_k}
$$


```python
import numpy as np

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# softmax
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return y.T
    x = x - np.max(x) # 1-dim case
    return np.exp(x) / np.sum(np.exp(x))

# mean_squared_error
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# cross_entropy_error

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # one-hot-label case
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

```



#### TwoLayerNet

> The Notation Convention: row represents the input, column represents the output.

<img src="https://s2.loli.net/2022/01/16/VJad6KqtX7wxsu8.png" style="zoom:67%;" />

<img src="https://s2.loli.net/2022/01/16/EPDxcaywLG2gRrB.png" style="zoom:80%;" />

```python
import numpy as np
from functions import *
from gradient import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        # initialize
        self.params['w1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)


    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        y1 = np.dot(x, w1) + b1
        z1 = sigmoid(y1)
        y2 = np.dot(z1, w2) + b2
        z2 = softmax(y2)

        return z2

    def loss(self, x, t):
        z = self.predict(x)
        return cross_entropy_error(z, t)


    def gradient_without_back_propagation(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['w1'] = numerical_gradient(loss_W, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_W, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    
	# pseudo back propagation
    def gradient(self, x, t):
        # forward
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward

        y1 = np.dot(x, w1) + b1
        z1 = sigmoid(y1)
        y2 = np.dot(z1, w2) + b2
        z2 = softmax(y2)

        # backward
        dz = (z2 - t) / batch_num # derivative of the mean squired loss function
        grads['b2'] = np.sum(dz, axis = 0)
        grads['w2'] = np.dot(z1.T, dz)

        dy1 = np.dot(dz, w2.T)
        dz1 = sigmoid_grad(y1) * dy1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis = 0)

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y==t) / float(y.shape[0])

        return accuracy

```



#### Mini-batch training without backward propagation

```python
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet
from functions import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

# parameters
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# recode the train loss
train_loss_list = []

# record the accuracy
train_acc_list = []
test_acc_list = []

# the average iters per epoch
iter_per_epoch = max(train_size / batch_size, 1)


# training
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

for i in range(iters_num):
    # get mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    # update the parameters
    for key in ('w1', 'b1', 'w2','b2'):
        network.params[key] -= learning_rate * grad[key]

    # record the loss and training process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # echo design
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc, test_acc |" + str(train_acc) + ', ' + str(test_acc))


```

### Back Propagation

#### Math

$$
z = f (Y), \ Y = XW + B \to {\partial z \over \partial W} = X^T{\partial z \over \partial Y} \\
z = f (Y), \ Y = XW + B \to {\partial z \over \partial X} = {\partial z \over \partial Y} W^T
$$

For the proof of this formula, you can just take 1-dimension case for example,  and increase the dimension, then smoothly generalizing to the matrix form. 

That is, to get an intuition understanding, look the following, and draw the neural connection picture to gain a feeling over the formula.
$$
X = [x_1, x_2, x_3], \  \ W=
\begin{bmatrix}
w_{1} \\
w_{2}\\
w_{3}
\end{bmatrix}
, \ \  Y=y
$$
And now, just add one more dimension to see the process again:
$$
X = [x_1, x_2, x_3], \  \ W=
\begin{bmatrix}
w_{11}& w_{12} \\
w_{21}& w_{22}\\
w_{31}& w_{32} 
\end{bmatrix}
, \ \  Y=[y_1, y_2]
$$


#### The representation of layers in the back propagation algorithm

$x$, $a$ are the nodes in the previous representation, but now we are gonna to talk about the forward propagation and back propagation, so we need to express the transformations in each layers explicitly.

<img src="https://s2.loli.net/2022/01/18/P7McH53ZVQK9iyd.png" style="zoom:80%;" />



#### BP design

```python
import numpy as np
from functions import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out =  None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

    def backward(self, dout):
        dx = sigmoid_grad(self.out) * dout
        return dx

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

```



#### Mini-batch training with backward propagation

```python
# the training part are the same
import numpy as np
from functions import *
from gradient import *
from BP_design import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        # initialize
        # weight_init_std is the initial standard variance of weight
        self.params['w1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        # ordered dictionary
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        z = self.predict(x)
        return self.lastLayer.forward(z, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t) # loss function will call the predict function

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)


        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        return grads


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y==t) / float(y.shape[0])

        return accuracy

```





### CNN

#### Introduction

Why CNN?

In the previous neural network, we just use the fully connected layer. However, many important spatial information are lost in the fully connected layer,  such as one pixel is closer to its neighboring pixel. And also, the fully connected layer just has too many parameters, so complex.

https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215

#### Convolution

对卷积的一些看法，

在深度学习中，卷积中的过滤器不经过反转。严格来说，这是互相关。我们本质上是执行逐元素乘法和加法。但在深度学习中，直接将其称之为卷积更加方便。这没什么问题，因为过滤器的权重是在训练阶段学习到的。如果上面例子中的反转函数 g 是正确的函数，那么经过训练后，学习得到的过滤器看起来就会像是反转后的函数 g。因此，在训练之前，没必要像在真正的卷积中那样首先反转过滤器。

卷积其实要把核翻转一下的，但在图像中大部分核都是对称的，所以两者就一样了。

<img src="https://s2.loli.net/2022/01/21/GHfwx1Z4vznmO2j.jpg" style="zoom:80%;" />



对于图像而言，离散卷积的计算过程是模板翻转，然后在原图像上滑动模板，把对应位置上的元素相乘后加起来，得到最终的结果。如果不考虑翻转，这个滑动-相乘-叠加的过程就是相关操作。事实上我也一直用相关来理解卷积。在时域内可以从两个角度来理解这样做的含义。

一种是滤波，比如最简单的高斯模板，就是把模板内像素乘以不同的权值然后加起来作为模板的中心像素值，如果模板取值全为1，就是滑动平均；如果模板取值为高斯，就是加权滑动平均，权重是中间高，四周低，在频率上理解就是低通滤波器；如果模板取值为一些边缘检测的模板，结果就是模板左边的像素减右边的像素，或者右边的减左边的，得到的就是图像梯度，方向不同代表不同方向的边缘；

另一种理解是投影，因为当前模板内部图像和模板的相乘累加操作就是图像局部patch和模板的内积操作，如果把patch和模板拉直，拉直的向量看成是向量空间中的向量，那么这个过程就是patch向模板方向上的投影，一幅图像和一个模板卷积，得到的结果就是图像各个patch在这个方向上的response map或者feature map；如果这样的模板有一组，我们可以把这一组看成一组基，得到的一组feature map就是原图像在这组基上的投影。常见的如用一组Garbor滤波器提取图像的特征，以及卷积神经网络中的第一层，图像在各个卷积核上的投影。



**Original idea in probability and signal processing**

不稳定输入、稳定输出，求系统存量。



**Convolution in CNN**

> Why filter is called filter?

From the respect of Fourier Transform:

https://www.robots.ox.ac.uk/~az/lectures/ia/lect2.pdf





#### The whole procedure

![](https://s2.loli.net/2022/01/19/4iDgATEk71j3dLa.png)

![](https://s2.loli.net/2022/01/19/UsAnwuXZqc34EQf.png)

> pseudo code

![](https://s2.loli.net/2022/01/19/tNvT9HEpzkBmefM.png)



#### 

#### **Data representation**

$$
4 Dimensions \ \ \ \ \ \ (batch\_num,channel,height, width)
$$



**Channel**

<img src="https://s2.loli.net/2022/01/20/fhebwyr7POj8Jpm.png" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/01/20/uIl1OxnXhyCPNQM.png" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/01/20/as9dL7CNURZA24w.png" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/01/20/vNPUkcdfjzWZhSg.png" style="zoom:80%;" />



**Batch_num**

<img src="https://s2.loli.net/2022/01/20/bTLWOHYa6JKh9xm.png" style="zoom:80%;" />



#### Convolution layer

[反向传播之六：CNN 卷积层反向传播 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40951745)

**im2col**

The normal convolution operation use many for loops, which behave really slowly with numpy. So, actually, we choose **im2col**, basically saying, it's just reshape convolution into matrix multiplication. 

In more detail,  the number of cols after the im2col transform is the **number of kernel traverses**, the rows are the filter's size times #channel.  

<img src="https://s2.loli.net/2022/01/20/XZdEPa7IQOHclCG.png" style="zoom:80%;" />

<img src="C:\Users\19772\Pictures\blog解释图片\20171105162824191" alt="img" style="zoom:80%;" />



<img src="https://s2.loli.net/2022/01/20/UhOfc53aIetqjdR.jpg" style="zoom: 50%;" />

<img src="https://s2.loli.net/2022/01/20/G1yqHIgJaYKxBpF.png" style="zoom:80%;" />



```python
import numpy as np
from common.util import im2col
from common.util import col2im
class Covolution:
    def __init__(self, w, b, stride = 1, pad = 0 ):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dw = None
        self.db = None

    def forward(self, x):
        # w means filter
        # get the shape
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        # output height and width size
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # transformation and convolution
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.w.reshape(FN, -1).T # traverse for dot product
        out = np.dot(col, col_w) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


    def backward(self, dout):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
```





**Difference between transpose and reshape**

[真正理解 transpose 和 reshape - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/72128096)

[(42条消息) 深度学习入门-某些细节的理解2（im2col，transpose交换维度）_超级菜-CSDN博客](https://blog.csdn.net/weixin_42820722/article/details/120483962)



#### Pooling layer

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None


    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # reshape can be visualized as first make the matrix 1-d with order rows to rows
        # and then, reshape it from the order of width, height...

        out = np.max(col, axis = 1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
```





#### Fully connection layer





