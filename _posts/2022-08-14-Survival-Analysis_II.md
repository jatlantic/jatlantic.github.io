---
layout: "post"
title: "Survival Analysis II - Kaplan-Meier Estimator"
date: "2022-08-14 11:51"
comments: true
use_math: true
---

This post builds on the survival analysis introduction and delves deeper into the "how" and "why" of the Kaplan-Meier (KM) estimator.


## Motivation & Goals

## The Kaplan-Meier Estimator

The KM estimator is expressed as

$\begin{aligned} \hat{S}(t) = \prod_{y_{(i)}\leq t} \hat{p_i} = \prod_{y_{(i)}\leq t} \left[ \frac{n_i -d_i}{n_i}\right] 
= \prod_{i = 1}^{k} \left[ \frac{n_i -d_i}{n_i}\right] \end{aligned}$ where $y_{(k)} \leq t < y_{(k+1)}$.

- $n_i$ describes that number of people that are alive and that have not been censored just prior to time $t$ ($=y_{(i)}$).
- $d_i$ describes the number of people that died at $y_{(i)}$.

Now, where does this formula actually come from?

It comes from applying the multiplication rule $P(A \cap B) = P(B\vert A)P(A)$ repeatedly. This leads us to express the survivor function as

$S(t) = P(T>t) = \prod_{y_{(i)}\leq t} p_i$

- $\hat{q_i} =\ frac{d_i}{n_i}$
- $\hat{p_i} = 1-\hat{q_i}= 1 - \frac{d_i}{n_i} = \left[ \frac{n_i-d_i}{n_i} \right]$

If we would observe no censoring, we could just take the empirical estimator instead.

## The Kaplan-Meier Estimator in Practice

Now, let's have a look at how we would implement this in practice. Let's load a dataset and visualize some survival times. The data is taken from the lifelines package (Moeschberger dataset).

{% highlight python %}
import numpy as np
import pandas as pd
import lifelines
from lifelines.datasets import load_kidney_transplant
from lifelines import KaplanMeierFitter
import plotly.express as px


df = lifelines.datasets.df = lifelines.datasets.load_kidney_transplant().rename_axis('patient').reset_index()

df['start'] = 0
df['end'] = df.time

new = df[0:11]
fig = px.line(new, x='time', y='patient',color='patient', title='Survival Time of Patients')
for r,col in new.iterrows():  
    fig.add_shape(type="line", x0=new.start[r], x1=new.end[r],y0=new.index[r],y1=new.patient[r], line_width=2, line_dash="dash", line_color="blue")
fig
{% endhighlight %}

Now, we create a new dataframe that contains a sorted list of the number of deaths and the number of people at risk over time. 

{% highlight python %}
def KM():
   ts = df.time.unique().sort()
   at_risk = pd.Series(0, index=ts)
   for t in ts:
      k = (t <= df['end']) # true false if time t is below or equal end time
      at_risk[t] = k.sum() # sum trues to get all cancer patients by duration
    
   d = pd.Series(0, index=ts)

   for t in ts:
      k = (df['death'] == 1) & (t == df['end'])
      d[t] = k.sum()
      dff = pd.DataFrame(dict(death=d, 
            at_risk=at_risk), index=ts)

   dff['hazard'] = dff.death/dff.at_risk

   dff['surv'] = (1-dff.hazard).cumprod()
   return dff
{% endhighlight %}

Now, we plot the the results with Plotly Express and compare with the *lifelines* results.

{% highlight python %}
fig1 = px.line(dff, x='index',  y='surv', title='Kaplan Meier Survival Function', labels={'surv':'KM_estimate', 'index':'timeline'})

kmf = KaplanMeierFitter()
kmf.fit(df.time, df.death)
est = kmf.survival_function_
est = est.reset_index()
fig2 = px.line(est, x='timeline',  y='KM_estimate', title='Kaplan Meier Survival Function (lifelines)')Function (lifelines)')
{% endhighlight %}

And voila the curves are actually the same.

<p align="center">
  <img  align="center" alt="rightcensoring" src="/assets/surv1.png" width="70%" /> 
</p>

<p align="center">
  <img  align="center" alt="rightcensoring" src="/assets/surv2.png" width="70%" /> 
</p>


## Confidence Intervals - Greenwood's Formula

Now it is time to actually look at the estimator from more statistical perspective - what are the confidence intervals of the estimator?



## Log-Rank Test

## Other Tests

## Sources

1. Kleinbaum, D. G. & Klein, M. Survival analysis: a self-learning text. (Springer, 2012).
2. Tableman, M. Survival Analysis Using S R*. (2016).
3. Rodriguez, G. GR’s Website. https://data.princeton.edu/wws509/notes/c7s1.
