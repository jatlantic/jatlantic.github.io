---
layout: "post"
title: "Survival Analysis III - Cox Proportional Hazard"
date: "2022-08-18 11:51"
comments: false
use_math: true
---

This post builds on the [Survival Analysis I - Introduction]({{ site.baseurl }}{% post_url 2022-08-09-Survival-Analysis_I %}) as well as [Survival Analysis II - Kaplan-Meier Estimator]({{ site.baseurl }}{% post_url 2022-08-14-Survival-Analysis_II %}) to illustrate the Cox Proportional Hazards (Cox PH) model.


## Motivation & Goals


## The Cox Proportional Hazards Model

The Cox PH model can be expressed in the following form (Kleinbaum, Klein, 2012):

$h(t,X) = h_0(t)e^{\sum_{i=1}^p \beta_i X_i}$

where $h(t)$ describes the baseline hazard that is only dependent on time $t$ and $e^{\sum_{i=1}^p}$ contains the time-independent explanatory variables $X$ and the coefficients $\beta$.

This implies that if all explanatory variables would be zero we would only look at the baseline hazard. Moreover, since the baseline hazard $h_0(t)$ is not specific we consider the Cox PH model a semiparametric model. And this points us to two of the reasons of why the Cox PH model is so popular: First, it allows us to approximate the parametric model that would specify the baseline hazard (e.g. Weibull, Exponential etc.)  without knowing it exactly. Second, we are able to estimate the $\beta$ coefficients without specifying $h_0(t)$.

## Cox PH Model Estimation

The $\beta$ coefficients are obtained by maximizing the "partial" log likelihood considering only failed subjects but not censored ones. The partial log likelihood involves the multiplicative sum of the different likelihoods at each time step and includes the risk set of all not yet censored subjects. The likelihood function of the Cox PH model is dependent on the order of the failure of events and the covariates that are affected with it. 

Let's illustrate the maximum likelihood (ML) estimation with an example from Klein, Kleinbaum (2012): 

<div class="divTable">
    <div class="thead">
        <div class="row">
            <div class="cell">Subject</td>
            <div class="cell">Time</td>
            <div class="cell">Status</td>
            <div class="cell">Diabetes</td>
        </div>
    </div>
    <div class="tbody">
        <div class="row">
            <div class="cell">Alex</td>
            <div class="cell">2</td>
            <div class="cell">1</td>
            <div class="cell">1</td>
        </div>
        <div class="row">
            <div class="cell">Celine</td>
            <div class="cell">3</td>
            <div class="cell">1</td>
            <div class="cell">0</td>
        </div>
        <div class="row">
            <div class="cell">Dennis</td>
            <div class="cell">5</td>
            <div class="cell">0</td>
            <div class="cell">0</td>
        </div>
        <div class="row">
            <div class="cell">Estelle</td>
            <div class="cell">8</td>
            <div class="cell">1</td>
            <div class="cell">1</td>
        </div>
    </div>
</div>

We apply the Cox PH model where $h(t) = h_0(t)\exp{(\beta_1 \text{DIABETES})}$. With the above table the exponent will be zero for Celine and Dennis. The Cox likelihood will be the multiplication of the likelihood at each failure time step:

$L = \left[ \frac{h_0(t)e^{\beta_1}}{h_0(t)e^{\beta_1} + h_0(t)e^{0} + h_0(t)e^{0} + h_0(t)e^{\beta_1}} \right] \times \left[ \frac{h_0(t)e^{0}}{h_0(t)e^{0} + h_0(t)e^{0}+ h_0(t)e^{\beta_1}}\right] \times \left[ \frac{h_0(t)e^{\beta_1}}{h_0(t)e^{\beta_1}}\right]$

We can easily see that $h_0(t)$ cancels out which allows us to generalize the above formula to following partial likelihood:

$PL(\beta)=\prod_{i:C_i=1}\frac{h(Y_i \vert X_i)}{\sum_{j:Y_j\geq Y_j}\exp{h(Y_i\vert X_j)}}$

which together with the Cox assumptions turns into [1]:

$\begin{equation} PL(\beta)=\prod_{i:C_i=1}\frac{\exp{(X_i \beta)}}{\sum_{j:Y_j\geq Y_j}\exp{(X_j \beta)}} \end{equation}$

where 

- $𝐶_i$ = type of event either one for the event or 0 for right−censoring so that product sum in the formula above goes over all individuals that experienced the event.
- $Y_t$ represents the time of the event for individual i so that $j:Y_j\geq Y_j$ represents the individuals at risk at $Y_t$.




Taking the log gives us:


$logPL(\beta)=\sum_{i:C_i=1}\left[ \exp{(X_i \beta)}- \log \left(\sum_{j:Y_j\geq Y_j}\exp{(X_j \beta)}\right)\right]$

If we add a minus to the beginning of the above equation we obtain the Cox PH loss function that we would like to minimize. Now instead of taking the derivative with respect to $\beta$ and set it to zero, we'll use JAX here (following the post by Sidravi). First, we note that the above can only be applied when no ties are present that is there is not more than one event occurring at the same time. If this is the case *Breslow*'s or *Efron*'s method can be used.
So to conform with that assumption we prepare the data accordingly:

{% highlight python %}
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import numpy as np
import pandas as pd
from jax import grad, hessian
from scipy.optimize import minimize
import jax.scipy as jsp
import jax

# import data
rossi = load_rossi()
# prepare for no ties
rossi = rossi.drop_duplicates('week', keep='first')
# sort data by time
rossi = rossi.sort_values(by='week',ascending=True)
{% endhighlight %}

Now we create a helper matrix for summing the correct individuals in the second part of the above formula:

{% highlight python %}
# calculate at risk helper matrix (i.e. riskset)
# essential to capture the decreasing number of individuals at risk when taking the exp sums below.
at_risk = np.triu(np.ones(rossi.shape[0]))
{% endhighlight %}


Here we code negative log likelihood function and minimize it:

{% highlight python %}
@jax.jit
def neglogp(betas, x = rossi[['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']].to_numpy(), riskset=at_risk, observed=rossi.arrest.to_numpy()):
    betas_x = betas @ x.T
    # now we want to sum in decreasing order of elements
    # first all are in the risk set and then every time step one less
    # this is well achieved with the np.triu function
    riskset_beta = betas_x * riskset
    # Compute the log of the sum of exponentials of input elements.
    # b is the weighting factor for each input element
    # i.e. we sum only over the elements of the riskset, 
    # in other words it allows us to get rid of the values exp(0) = 1 that #are created without the weight.
    res_vec = (betas_x - jsp.special.logsumexp(riskset_beta, b = riskset, axis=1))
    # we sum the result on ly for those individuals where the event occurred.
    return -(observed * res_vec).sum()
{% endhighlight %}

The results are the same as can be verified when running the [notebook here](https://github.com/jatlantic/jatlantic.github.io/blob/main/notebooks/Cox_PH_Model_18.08.22.ipynb).


## Statistical Inference

Having estimated the coefficients, we would like to know whether these are statistically significant and what the confidence intervals are. Without going into too much detail we can obtain the standard errors from the diagonals of the Hessian that we calculated before (for more info see [here](https://stats.stackexchange.com/questions/68080/basic-question-about-fisher-information-matrix-and-relationship-to-hessian-and-s). This allows then to obtain the p-values by dividing the coefficient by the standard error (=Std. Err./Coef.) and assuming a certain distribution depending on the hypothesis that we test (for more see Wald statistic).

Moreover, we can also consider the likelihood ratio (LR) statistic. This allows us to test for the significance of a coefficient by subtracting the LR statistics of the two models (reduced vs. full model) whose result is chi-squared distributed and enables us to obtain the p-value.


## Hazard Ratio
Another question that you might have is how to read and interpret these coefficients? What is commonly done is to look at the **hazard ratio (HR)** where we compare the coefficients and $X$ values of one individual to another individual or the baseline. Mathematically, this would look like so:

$\hat{\text{HR}} = \exp{(\sum_{i=1}^p \hat{\beta}_i (X_i^*-X_i))}$

## Assumptions
What are the underlying assumptions of this model? One of the key assumptions in the Cox PH model holds that the HR is constant over time. In other words, we say that $\frac{\hat{h}(t,X^*)}{\hat{h}(t,X)}= \hat{\theta}$ is constant. This asks for preliminary anlyses and contextual thinking to verify whether the Cox PH model is the right one to use for the case at hand. One option could be to just compare different Kaplan-Meier curves as explained in previous posts or to extent the Cox PH model by a time dependent variable.


You can get the [notebook here](https://github.com/jatlantic/jatlantic.github.io/blob/main/notebooks/Cox_PH_Model_18.08.22.ipynb).

## Sources

1. Boeva, V. Machine Learning on Genomics. (2022).
2. Klein, J. P. & Moeschberger, M. L. Survival analysis: techniques for censored and truncated data. vol. 1230 (Springer, 2003).
3. Kleinbaum, D. G. & Klein, M. Survival analysis: a self-learning text. (Springer, 2012).
4. Ravinutala, S. [Survival models] Cox proportional hazard model. Sid Ravinutala https://sidravi1.github.io/blog/2021/10/11/cox-proportional-hazard-model (2021).
5. Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime: Some Experimental Results. New York: Academic Press. John Fox, Marilia Sa Carvalho (2012). The RcmdrPlugin.survival Package: Extending the R Commander Interface to Survival Analysis. Journal of Statistical Software, 49(7), 1-32.
6. Sawyer, S. The Greenwood and Exponential Greenwood Confidence Intervals in Survival Analysis. (2003).
7. Rodriguez, G. GR’s Website. https://data.princeton.edu/wws509/notes/c7s1.
8. Tableman, M. Survival Analysis Using S R*. (2016).


