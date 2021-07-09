'''
This file defines all of the means and covariances needed for lambda dynamics.
These functions are the numerical factors required for calculating the covariances. The naming convention
is the function and then the powers of n and e. An underscore indicates the power is negative.
For example, fn_e13 is the term in front of the term in f with n/e^(1/3).

For the covariances, the naming convention is covxy, where x and y are the two things we are taking the covariance of.
One important notation note: I decided to pass all the means as an array rather than individually. This array
is called "m" and contains all sums over n and e requires for all covariances.
It will be calculated separately and then passed into these calculations.
Finally, note these are symmetric so covfh = covhf, so I only define it once.

For all of these functions,
s are state variables, call S, N, or E
p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
ds are derivatives of state variables, call dS, dN, or dE
m are means which contain all sums over n and e
'''

# Import
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import DynaMETE_Rfunctions as rf

# For <f>
def fn_e13(s,p):
    return p['b0']-p['d0']*s['E']/p['Ec']
def fn(s,p):
    return p['m0']/s['N']

# For <h>
def hne23(s,p):
    return p['w0']-p['d0']*s['E']/p['Ec']
def hne(s,p):
    b0i=0.0001
    beta = fsolve(rf.beta_constraint,b0i,args=s)[0]
    return -p['w10']/np.log(1/beta)**(2/3)
def hn(s,p):
    return p['m0']/s['N']

# For <q>
def qc(s,p):
    return p['m0']*np.exp(-p['mu']*s['S']-np.euler_gamma)
def qdn_e13(s,p):
    return -s['S']*p['d0']*s['E']/p['Ec'] # This has the delta function d_{n,1}

# For <df/dt>
def dfn(s,p,ds):
    return -p['m0']/s['N']**2*ds['dN']
def dfn_e13(s,p,ds):
    return -p['d0']/p['Ec']*ds['dE']

# For <dh/dt>
def dhn(s,p,ds):
    return -p['m0']/s['N']**2*ds['dN']
def dhne23(s,p,ds):
    return -p['d0']/p['Ec']*ds['dE']
def dhne(s,p,ds):
    b0i=0.0001
    beta = fsolve(rf.beta_constraint,b0i,args=s)[0]
    dbdt = (ds['dS']-s['S']*ds['dN']/s['N'])/(s['N']*(np.log(1/beta)-1))
    return -2*p['w10']/(3*beta*np.log(1/beta)**(5/3))*dbdt

# <dq/dt>
def dqc(s,p,ds):
    return -p['mu']*p['m0']*ds['dS']*np.exp(-p['mu']*s['S']-np.euler_gamma)
def dqdn_e13(s,p,ds):
    dqds = -p['d0']*s['E']/p['Ec']*ds['dS']
    dqde = -s['S']*p['d0']/p['Ec']*ds['dE']
    return dqds + dqde

# <f>
def fm(s,p,m):
    return fn_e13(s,p)*m['n_e13'] + fn(s,p)*m['n']
# <h>
def hm(s,p,m):
    return hne23(s,p)*m['ne23'] + hne(s,p)*m['ne'] + hn(s,p)*m['n']
# <q>
def qm(s,p,m):
    return qc(s,p) + qdn_e13(s,p)*m['dn_e13']

# <df>
def dfm(s,p,ds,m):
    return dfn(s,p,ds)*m['n'] + dfn_e13(s,p,ds)*m['n_e13']
# <dh>
def dhm(s,p,ds,m):
    return dhn(s,p,ds)*m['n'] + dhne23(s,p,ds)*m['ne23'] + dhne(s,p,ds)*m['ne']
# <dq>
def dqm(s,p,ds,m):
    return dqc(s,p,ds) + dqdn_e13(s,p,ds)*m['dn_e13']

# Covariances
# With f
def covnf(s,p,m):
    return fn_e13(s,p)*m['n2_e13'] + fn(s,p)*m['n2'] - m['n']*fm(s,p,m)
def covnef(s,p,m):
    return fn_e13(s,p)*m['n2e23'] + fn(s,p)*m['n2e'] - m['ne']*fm(s,p,m)
def covff(s,p,m):
    f2m = fn_e13(s,p)**2*m['n2_e23'] + fn(s,p)**2*m['n2'] + 2*fn_e13(s,p)*fn(s,p)*m['n2_e13']
    return f2m - fm(s,p,m)**2
# With h
def covfh(s,p,m):
    fhm1 = fn_e13(s,p)*(hne23(s,p)*m['n2e13'] + hne(s,p)*m['n2e23'] + hn(s,p)*m['n2_e13'])
    fhm2 = fn(s,p)*(hne23(s,p)*m['n2e23'] + hne(s,p)*m['n2e'] + hn(s,p)*m['n2'])
    return fhm1 + fhm2 - fm(s,p,m)*hm(s,p,m)
def covnh(s,p,m):
    return hne23(s,p)*m['n2e23'] + hne(s,p)*m['n2e'] + hn(s,p)*m['n2'] - m['n']*hm(s,p,m)
def covneh(s,p,m):
    return hne23(s,p)*m['n2e53'] + hne(s,p)*m['n2e2'] + hn(s,p)*m['n2e'] - m['ne']*hm(s,p,m)
def covhh(s,p,m):
    h2m1 = hne23(s,p)**2*m['n2e43'] + hne(s,p)**2*m['n2e2'] + hn(s,p)**2*m['n2']
    h2m2 = 2*(hne23(s,p)*hne(s,p)*m['n2e53'] + hne23(s,p)*hn(s,p)*m['n2e23'] + hne(s,p)*hn(s,p)*m['n2e'])
    return h2m1 + h2m2 - hm(s,p,m)**2
# With q
def covnq(s,p,m):
    # Is qc(s,p)*m['n'] + qdn_e13(s,p)*m['dn_e13'] - m['n']*qm[s,p,m]
    # This simplifies
    return qdn_e13(s,p)*m['dn_e13']*(1-m['n'])
def covneq(s,p,m):
    # Is qc(s,p)*m['ne'] + qdn_e13(s,p)*m['dne23'] - m['ne']*qm[s,p,m]
    # But simplifies to
    return qdn_e13(s,p)*(m['dne23']-m['ne']*m['dn_e13'])
def covfq(s,p,m):
    # Constant cross term cancels, so when we subtract away means we have to remove it
    fq = qdn_e13(s,p)*(fn_e13(s,p)*m['dn_e23'] + fn(s,p)*m['dn_e13'])
    return fq - fm(s,p,m)*(qm(s,p,m)-qc(s,p))
def covhq(s,p,m):
    hq = qdn_e13(s,p)*(hne23(s,p)*m['dne13'] + hne(s,p)*m['dne23'] + hn(s,p)*m['dn_e13'])
    return hq - hm(s,p,m)*(qm(s,p,m)-qc(s,p))
def covqq(s,p,m):
    # Constant terms go away
    return qdn_e13(s,p)**2*(m['dn_e23'] - m['dn_e13']**2)

# With df
def covndf(s,p,ds,m):
    return dfn(s,p,ds)*m['n2'] + dfn_e13(s,p,ds)*m['n2_e13'] - m['n']*dfm(s,p,ds,m)
def covnedf(s,p,ds,m):
    return dfn(s,p,ds)*m['n2e'] + dfn_e13(s,p,ds)*m['n2e23'] - m['ne']*dfm(s,p,ds,m)
# With dh
def covndh(s,p,ds,m):
    ndhm = dhn(s,p,ds)*m['n2'] + dhne23(s,p,ds)*m['n2e23'] + dhne(s,p,ds)*m['n2e']
    return ndhm - m['n']*dhm(s,p,ds,m)
def covnedh(s,p,ds,m):
    nedhm = dhn(s,p,ds)*m['n2e'] + dhne23(s,p,ds)*m['n2e53'] + dhne(s,p,ds)*m['n2e2']
    return nedhm - m['ne']*dhm(s,p,ds,m)
# With dq
def covndq(s,p,ds,m):
    return dqdn_e13(s,p,ds)*m['dn_e13']*(1-m['n'])
def covnedq(s,p,ds,m):
    return dqdn_e13(s,p,ds)*(m['dne23']-m['ne']*m['dn_e13'])

# n and ne alone
def covnn(m):
    return m['n2']-m['n']**2
def covnen(m):
    return m['n2e']-m['n']*m['ne']
def covnene(m):
    return m['n2e2']-m['ne']**2