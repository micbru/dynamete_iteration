'''
This file defines all of the necessary functions for DynaMETE, including the transition functions and
the structure function R. This function does NOT include sums over n, since it is designed to be a 
more flexible version incorporating different transition functions. This will be very slow for large N or E.
It also defines the METE constraint for beta, which is needed, and a function to obtain mete_lambdas.

To change the functional form of the transition functions, you need only change f, h, and/or q, and the 
corresponding function dfdt, dhdt, and/or dqdt.

This version specifically replaces d0/E_c n/e^(-1/3) with d0/E_c n^2/e^(-1/3) to test adding a new degree of
freedom in n dependence.
'''

# Import
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy import integrate

# METE functions
def beta_constraint(b,s):
    '''This is the beta constraint in METE with give state variables. Use this as a function call to get beta.
    Inputs s as state variables, call S, N, or E
    Also inputs beta
    outputs beta constraint to minimize'''
    return b*np.log(1/(1-np.exp(-b)))-s['S']/s['N']

def get_beta(s,b0=0.0001):
    '''This returns beta from METE. Inputs s as state variables.'''
    return fsolve(beta_constraint,b0,args=s)[0]

def mete_lambdas(s,b0=0.0001):
    '''This returns the METE lambdas for a given set of state variables.
    Inputs s as state variables, call S, N, or E
    Optional input of an initial beta, if we know it's going to be somewhere other than small positive.
    outputs array of lambdas'''
    beta = get_beta(s,b0)
    l2 = s['S']/(s['E']-s['N']) 
    ls = np.array([beta-l2,l2,0,0,0])
    return ls

# Transition functions
# The idea here is to make everything easy to change by changing only these functions.
# For f
def fb0(s,p):
    return p['b0']
def fd0(s,p):
    return -p['d0']*s['E']/p['Ec']
def fm0(s,p):
    return p['m0']/s['N']
def f(n,e,s,p):
    '''Transition function for dN/dt. n and e are microscopic variables.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    return (fb0(s,p)+fd0(s,p)*n)*n/e**(1/3)+fm0(s,p)*n

# For h
def hw0(s,p):
    return p['w0']
def hd0(s,p):
    return -p['d0']*s['E']/p['Ec']
def hw10(s,p):
    b0i=0.0001
    beta = get_beta(s,b0i)
    return -p['w10']/np.log(1/beta)**(2/3)
def hm0(s,p):
    return p['m0']/s['N']
def h(n,e,s,p):
    '''Transition function for dE/dt. n and e are microscopic variables.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w1, Ec, or mu '''
    return (hw0(s,p)+hd0(s,p)*n)*n*e**(2/3)+hw10(s,p)*n*e+hm0(s,p)*n

# For q
def qc(s,p):
    return p['m0']*np.exp(-p['mu']*s['S']-np.euler_gamma)
def qd0(s,p):
    return -s['S']*p['d0']*s['E']/p['Ec']
def q(n,e,s,p):
    '''Transition function for dS/dt. n and e are microscopic variables.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    For now this doesn't implement speciation models, ie. s1=s2=0'''
    # Set up kronecker delta in an easy way. Round n to nearest int, if it's 1 then include term
    # I actually need this to be vectorized, so let's do it slightly differently.
    # First check if n is scalar
    if np.isscalar(n):
        kn1 = int(np.rint(n))==1
    else:
        kn1 = np.zeros(len(n))
        kn_arg = np.where(np.rint(n)==1) 
        # I included the rounded int here because really this kronecker delta can be defined in continuous space
        # In that case there should also be a correction factor though, but let's ignore that.
        # The good news here is that below we only pass in arange, which by default passes in integers
        # So we should be ok as long as we are only using arange for passing in ranges of n
        # That is because arange rounds the variable it takes in so it can take steps of length 1
        kn1[kn_arg] = 1
    return qc(s,p) + qd0(s,p)*kn1/e**(1/3)

# Also need derivatives for lambda dynamics. Note that these have to be manually editted for alternate f,h,q
def dfdt(n,e,s,p,ds):
    return fd0(s,p)/s['E']*ds['dE']*n**2/e**(1/3) - fm0(s,p)*ds['dN']/s['N']*n

def dhdt(n,e,s,p,ds):
    return hd0(s,p)/s['E']*ds['dE']*n**2*e**(2/3) - hm0(s,p)*ds['dN']/s['N']*n

def dqdt(n,e,s,p,ds):
    # See q for how the kronecker delta works.
    if np.isscalar(n):
        kn1 = int(np.rint(n))==1
    else:
        kn1 = np.zeros(len(n))
        kn_arg = np.where(np.rint(n)==1)
        kn1[kn_arg] = 1
    return -qc(s,p)*ds['dS']*p['mu'] + qd0(s,p)*(ds['dS']/s['S']+ds['dE']/s['E'])*kn1/e**(1/3)

# R itself
def R(n,e,l,s,p):
    '''Unnormalized struture function for DynaMETE.
    n,e are microscopic variables.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    return np.exp(-l[0]*n-l[1]*n*e-l[2]*f(n,e,s,p)-l[3]*h(n,e,s,p)-l[4]*q(n,e,s,p))

# For calculating a single mean with specific powers of n and e
def mean_pow(npow,epow,l,s,p,z=1):
    '''
    This function returns the mean of n^npow*e^epow over the R function.
    It is NOT normalized, but it does take in z as an optional argument to normalize.
    This function uses quad integral over log e for each n then sums over n.
    Note that npow=epow=0 corresponds to Z, so by default these are not normalized.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    '''
    nrange = np.arange(s['N'])+1
    eint = integrate.quad_vec(lambda loge: np.exp(loge*(1+epow))*R(nrange,np.exp(loge),l,s,p),0,np.log(s['E']))[0]
    return np.sum(nrange**npow*eint)/z

# For calculating a covariance with specific powers of n and e for each function
def cov_pow(npow,epow,l,s,p,z):
    '''
    This function returns the covariance of two functions with the form n^npow*e^epow over the R function.
    You have to pass in the normalization so that things are faster than calculating normalization each time.
    npow and epow should both be 2d arrays with the functions.
    For example, if you want COV(n^2,ne), pass npow=[2,1], epow=[0,1]
    This function uses quad integral over log e for each n then sums over n.
    z is the normalization
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    '''
    nrange = np.arange(s['N'])+1
    # Get integral over both functions
    ffeint = integrate.quad_vec(lambda loge: np.exp(loge*(1+np.sum(epow)))*R(nrange,np.exp(loge),l,s,p),0,np.log(s['E']))[0]
    ff = np.sum(nrange**np.sum(npow)*ffeint)/z
    # Get integral over each function
    f1f2 = 1
    for nn,ee in zip(npow,epow):
        feint = integrate.quad_vec(lambda loge: np.exp(loge*(1+ee))*R(nrange,np.exp(loge),l,s,p),0,np.log(s['E']))[0]
        f1f2 *= np.sum(nrange**nn*feint)/z
    return ff-f1f2

# For calculating a single mean over an arbitrary function
# Use mean_pow for non-functions
def mean(func,l,s,p,*args,z=1):
    '''
    This function returns the mean of an arbitrary function over the R function.
    It is NOT normalized, but it does take in z as an optional argument to normalize.
    Because I put *args first, you have to use z=z0 if you want to put in a normalization.
    The arbitrary function must take arguments of the form (n,e,s,p) for this to work.
    This is the form of the f,h, and q functions above.
    You can pass additional arguments as required for the function (ie. pass ds for df/dt)
    To pass in n or n*e, use lambda n,e,s,p: n or lambda n,e,s,p: n*e, or similar.
    Alternatively, use mean_pow
    This function uses quad integral over log e for each n then sums over n.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    z is the normalization
    '''
    nrange = np.arange(s['N'])+1
    # Below is to make this easier for lambda functions, but it isn't worth it. Just require s and p passed, 
    # and let other things be passed as args if needed.
    # Check if we need args by looking at function passed in
#    funcargs = func.__code__.co_argcount
#    if funcargs >= 4:
#        args = s,p,args
    eint = integrate.quad_vec(lambda loge: np.exp(loge)*R(nrange,np.exp(loge),l,s,p)*func(nrange,np.exp(loge),s,p,*args),0,np.log(s['E']))[0]
    return np.sum(eint)/z

# For calculating a covariance
# Note if you want to do this with non-functions, use cov_pow
def cov(func1,func2,l,s,p,z,*args):
    '''
    This function returns the covariance of two arbitrary functions over the R function.
    You have to pass in the normalization so that things are faster than calculating normalization each time.
    The arbitrary functions must take arguments of the form (n,e,s,p) for this to work.
    This is the form of the f,h, and q functions above.
    You can pass additional arguments as required for the function (ie. pass ds for df/dt)
    To pass in n or n*e, use lambda n,e,s,p: n or lambda n,e,s,p: n*e, or similar.
    This function uses quad integral over log e for each n then sums over n.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    z is the normalization
    '''
    nrange = np.arange(s['N'])+1
    # Get integral over both functions
    ffeint = integrate.quad_vec(lambda loge: np.exp(loge)*R(nrange,np.exp(loge),l,s,p)*func1(nrange,np.exp(loge),s,p,*args)*func2(nrange,np.exp(loge),s,p,*args),0,np.log(s['E']))[0]
    ff = np.sum(ffeint)/z
    # Get integral over each function
    f1f2 = 1
    for func in [func1,func2]:
        feint = integrate.quad_vec(lambda loge: np.exp(loge)*R(nrange,np.exp(loge),l,s,p)*func(nrange,np.exp(loge),s,p,*args),0,np.log(s['E']))[0]
        f1f2 *= np.sum(feint)/z
    return ff-f1f2
    
def get_dXdt(l,s,p):
    '''
    Returns the time derivatives of the state variables. This makes it easier than calling mean three times
    every time I want to see the derivatives.
    Inputs lambdas, state variables, and parameters.
    Outputs a pandas series of ds.
    '''
    # Create storage
    ds = pd.Series(np.zeros(3),index=['dS','dN','dE'])
    # To normalize
    z = mean_pow(0,0,l,s,p)
    ds['dS'] = mean(q,l,s,p,z=z)
    ds['dN'] = s['S']*mean(f,l,s,p,z=z)
    ds['dE'] = s['S']*mean(h,l,s,p,z=z)
    return ds
