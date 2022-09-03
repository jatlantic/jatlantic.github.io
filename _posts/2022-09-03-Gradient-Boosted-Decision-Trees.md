---
layout: "post"
title: "Gradient Boosted Decision Trees"
date: "2022-09-02 11:51"
comments: false
use_math: true
---

You may have heard of [Numerai](https://numer.ai/), a decentralized hedge fund whose investment decisions and payouts are driven by the model predictions that anyone (really anyone) can share with that company. And even if not, the baseline model they provide every user with is a **Gradient Boosted Decision Tree (GDBTs)**. Now, it happens that this simple base model outperforms many of the individual user based models consistently (according to Richard Craib, founder of Numerai).
This and other opportunities to effectively use GDBTs, motivated me to have a closer look at this modeling approach. 

As in the previous post we will have a closer look at the [numpy-ml package](https://numpy-ml.readthedocs.io/en/latest/) implementation of GDBTs. 

## Introduction & Motivation

GDBTs combine a set of "weak" machine learning models (learners) into one strong machine learning model. In the training process we aim to minimize the loss between our given $Y$ values and the predicted values $f(X)$. 

Formally and following the [numpy-ml package](https://numpy-ml.readthedocs.io/en/latest/) package this can be expressed as:

$f_m(X) = b(X) + \eta w_1 g_1 + \ldots + \eta w_m g_m$

In practice and as you will see below in the code the we start from an initial estimate which is $b(X)$. This estimate is often just the average of the given $Y$ values (see `MeanBaseEstimator()` in the code). With that in mind, we start to iteratively approximate the negative gradient of the loss function (in our example MSE) using the predictions of the previous model. 
Let's have a look at observation $i$ only. The negative gradient would be:

$g_i = - \left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x_i)=f_{m-1}(x_i)}$

Our goal would then be to find the weights $w$ and gradients $g$ that minimize the following loss function at each iteration $k$:

$w_k, g_k = \arg \min_{w_k, g_k} L(Y, f_{m-1}(X) + w_k g_k)$


Why does it work? Informally said the negative gradients of the loss function are proportional to the residuals between our prediction and the true values. Hence, approximating the negative gradients is roughly equivalent to minimizing the residuals. 

What are the advantages of GDBTs?

- they handle both classification as well as continuous prediction tasks
- train fast on large datasets
- generally perform well

What are the disadvantages of GDBTs?

- overfitting can be an issue (requires regularization)




<p align="center">
  <img  align="center" alt="rightcensoring" src="/assets/images/2022-08-25_decision_tree_intro.png" width="70%" /> 
   <!-- <figcaption>Right censoring (Kleinbaum, Klein)</figcaption> -->
</p>

## Evaluation Criteria

As a first step and similar to the [decision trees]({{ site.baseurl }}{% post_url 2022-08-25-Decision_Trees %}) described in the previous post we define the loss function. In numpy-ml case these are mean squared error and cross entropy loss. For completeness I define them below:

1. Mean Squared Error (MSE)
$MSE = \frac{1}{n}\sum_{i=1}^n (yi-\hat{y}_i)^2$

2. Cross Entropy Loss 
$H(p,q) = -\sum p(x) \log q(x)


In [numpy-ml package](https://numpy-ml.readthedocs.io/en/latest/) these are coded a little bit more extensively. Let's have a brief look at it. Note that for the sake of simplicity I removed the `line_search()`

{% highlight python %}
class MSELoss:
    def __call__(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def base_estimator(self):
        return MeanBaseEstimator()

    def grad(self, y, y_pred):
        # simple derivative
        return -2 / len(y) * (y - y_pred)
{% endhighlight %}


{% highlight python %}
class ClassProbEstimator:
    def fit(self, X, y):
        self.class_prob = y.sum() / len(y)

    def predict(self, X):
        pred = np.empty(X.shape[0], dtype=np.float64)
        pred.fill(self.class_prob)
        return pred

class CrossEntropyLoss:
    def __call__(self, y, y_pred):
        # Machine limits for floating point types.
        eps = np.finfo(float).eps
        # return cross entropy loss
        return -np.sum(y * np.log(y_pred + eps))

    def base_estimator(self):
        return ClassProbEstimator()

    def grad(self, y, y_pred):
        eps = np.finfo(float).eps
        # derivative
        return -y * 1 / (y_pred + eps)
{% endhighlight %}

## Implementation

### Fit

Now, let's follow the [numpy-ml package](https://numpy-ml.readthedocs.io/en/latest/) step by step and evaluate the code bits. On a high level we have the *fit* and *predict* functions. The *fit* function learns on the data and the *predict* function gives us the predictions depending on the input data we provide it with. Let's start with the *fit()* function (for the continuous case) and with MSE loss only.

{% highlight python %}
    def fit(self, X, Y):
        """
        Fit the gradient boosted decision trees on a dataset.
        # set loss function
        loss = MSELoss()

        Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y

        N, M = X.shape
        self.out_dims = Y.shape[1]
        # each iteration is one row of learners with the same length as the output y
        self.learners = np.empty((self.n_iter, self.out_dims), dtype=object)

        # the same is valid for the weights
        self.weights = np.ones((self.n_iter, self.out_dims))
        # all but the first (i.e. zero positioned) row
        self.weights[1:, :] *= self.learning_rate

        # fit the base estimator
        Y_pred = np.zeros((N, self.out_dims))
        for k in range(self.out_dims):
            # this calls the MeanBaseEstimator() in the case of MSELoss
            # or the ClassProbEstimator() in the case of CrossEntropyLoss
            # here we consider only MSE loss
            t = loss.base_estimator()
            # in our case this takes the mean/avg of Y's column k
            t.fit(X, Y[:, k])
            # now we predict the values by putting the mean into each cell
            Y_pred[:, k] += t.predict(X)
            # the prediction is just the avg value
            # and we add the vector of avg values to the zero row and k-th column
            # verify shapes!!!
            self.learners[0, k] = t

        # incrementally fit each learner on the negative gradient of the loss
        # wrt the previous fit (pseudo-residuals)
        for i in range(1, self.n_iter):
            for k in range(self.out_dims):
                y, y_pred = Y[:, k], Y_pred[:, k]
                # use derivative of MSE loss to obtain negative gradient
                neg_grad = -1 * loss.grad(y, y_pred)

                # use MSE as the surrogate loss when fitting to negative gradients
                t = DecisionTree(
                    classifier=False, max_depth=self.max_depth, criterion="mse"
                )

                # fit current learner to negative gradients
                t.fit(X, neg_grad)
                self.learners[i, k] = t

                # compute step size and weight for the current learner
                step = 1.0
                h_pred = t.predict(X)

                # update weights and our overall prediction for Y
                self.weights[i, k] *= step
                Y_pred[:, k] += self.weights[i, k] * h_pred
{% endhighlight %}




### Predict

Now, after the training (fit) we can focus on the prediction where we go through every iteration and add the result of the multiplication of the learned weights and the respective learner's prediction. 


{% highlight python %}
def predict(self, X):
    """
    Use the trained model to classify or predict the examples in `X`.
    Parameters 
    """
    Y_pred = np.zeros((X.shape[0], self.out_dims))
    for i in range(self.n_iter):
        # removed k loop from original
        Y_pred[:, 0] += self.weights[i, 0] * self.learners[i, 0].predict(X)

    return Y_pred
{% endhighlight %}

Running the whole thing can be done with the code below. Keep in mind to download the *dt.py* as well (same folder as the notebook).
{% highlight python %}
Y = np.random.uniform(0, 100, 100)
X = np.random.uniform(0, 100, (100,4))
t = GradientBoostedDecisionTree(n_iter=100)
t.fit(X,Y)
t.predict(X)
{% endhighlight %}

You can find the [notebook here](https://github.com/jatlantic/jatlantic.github.io/blob/main/notebooks/Decision_Trees_25.08.22.ipynb).
Hope this was illustrative. 

## Sources

1. Bourgin, D. numpy-ml. (2022).
2. Breiman, L., Friedman, J. H., Olshen, R. A. & Stone, C. J. Classification and regression trees. Wadsworth. Inc. Monterey, California, USA (1984).
3. Hastie, T., Tibshirani, R., Friedman, J. H. & Friedman, J. H. The elements of statistical learning: data mining, inference, and prediction. vol. 2 (Springer, 2009).

from https://neptune.ai/blog/gradient-boosted-decision-trees-guide