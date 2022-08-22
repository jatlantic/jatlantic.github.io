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

This implies that if all explanatory variables would be zero we would only look at the baseline hazard. Moreover, since the baseline hazard $h_0(t)$ is not specific we consider the Cox PH model a semiparametric model.





You can get the [notebook here](https://github.com/jatlantic/jatlantic.github.io/blob/main/notebooks/Kaplan_Meier_Estimator_14.08.22.ipynb).

## Sources

1. Kleinbaum, D. G. & Klein, M. Survival analysis: a self-learning text. (Springer, 2012).
2. Tableman, M. Survival Analysis Using S R*. (2016).
3. Rodriguez, G. GR’s Website. https://data.princeton.edu/wws509/notes/c7s1.
4. Sawyer, S. The Greenwood and Exponential Greenwood Confidence Intervals in Survival Analysis. (2003).

