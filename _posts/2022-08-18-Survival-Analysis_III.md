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

| Subject | Time | Status | Diabetes |
|---------|------|--------|----------|
| Alex    | 2    | 1      | 1        |
| Celine  | 3    | 1      | 0        |
| Dennis  | 5    | 0      | 0        |
| Estelle | 8    | 1      | 1        |

We apply the Cox PH model where $h(t) = h_0(t)\exp{»beta_1DIABETES}. With the above table the exponent will be zero for Celine and Dennis. The Cox likelihood will be the multiplication of the likelihood at each failure time step:

$L = \left[ \frac{h_0(t)e^{\beta_1}}{h_0(t)e^{\beta_1} + h_0(t)e^{0} + h_0(t)e^{0} + h_0(t)e^{\beta_1}} \right] \times \left[ \frac{h_0(t)e^{0}}{h_0(t)e^{0} + h_0(t)e^{0}+ h_0(t)e^{\beta_1}}\right] \times \left[ \frac{h_0(t)e^{\beta_1}}{h_0(t)e^{\beta_1}}\right]$

We can easily see that $h_0(t)$ cancels out which allows us to generalize the above formula to:

$PL(\beta | D) = \prod_{j=1}^{d} \frac{\exp(x_j'\beta)}{\sum_{l \in R_j} \exp(x'_l\beta)}$

Taking the log gives us:


$pll(\beta | D) = \sum_{j=1}^{d} \left( x_j'\beta - \log\left(\sum_{l \in R_j} \exp(x'_l\beta)\right)\right)$

Now instead of taking the derivative with respect to $\beta$ and set it to zero, we'll use JAX here (following the post by Sidravi):

{% highlight python %}
import jax.scipy as jsp
import jax
from jax import grad, hessian
from scipy.optimize import minimize


@jax.jit
def neglogp(betas, riskset=riskset, observed=observed):
    betas_x = betas @ x.T
    riskset_beta = betas_x * riskset
    ll_matrix = (betas_x - jsp.special.logsumexp(riskset_beta, b=riskset, axis=1))
    return -(observed * ll_matrix).sum()
{% endhighlight %}


dlike = grad(neglogp)
dlike2 = hessian(neglogp)
res = minimize(neglogp, np.ones(2), method='Newton-CG', jac=dlike, hess=dlike2)
res.x


## Statistical Inference

## Hazard Ratio

## Assumptions







You can get the [notebook here](https://github.com/jatlantic/jatlantic.github.io/blob/main/notebooks/Kaplan_Meier_Estimator_14.08.22.ipynb).

## Sources

1. Kleinbaum, D. G. & Klein, M. Survival analysis: a self-learning text. (Springer, 2012).
2. Tableman, M. Survival Analysis Using S R*. (2016).
3. Ravinutala, S. [Survival models] Cox proportional hazard model. Sid Ravinutala https://sidravi1.github.io/blog/2021/10/11/cox-proportional-hazard-model (2021).
4. Rodriguez, G. GR’s Website. https://data.princeton.edu/wws509/notes/c7s1.
5. Sawyer, S. The Greenwood and Exponential Greenwood Confidence Intervals in Survival Analysis. (2003).

