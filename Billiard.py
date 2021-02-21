import pymc3 as pm
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import binom
import seaborn as sns

pos_true = 0.37
number_balls_periods = 3
for i in range(number_balls_periods):
  n=6*(i+1)
  X = binom.rvs(n=n, p=pos_true)
  with pm.Model():
    r = pm.Uniform('R',lower=0.,upper=1.) # Prior
    y_obs = pm.Binomial('y_obs', p=r, n=n, observed=X) # Likelihood
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step, chains=1)
  fig=plt.figure()
  sns.kdeplot(trace['R'],label='Number of balls dropped = '+str(n))
  plt.hlines(1.,0.,1.,color='orange',label='Prior')
  plt.xlim(0.,1.)
  plt.savefig('billiard_'+str(i)+'.png',format='png')