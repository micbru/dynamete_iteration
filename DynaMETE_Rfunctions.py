'''
This file defines all of the necessary functions for DynaMETE, including the transition functions,
the structure function R, and the analytic sum of R, nR, and n^2R over n.
It also defines the METE constraint for beta, which is needed, and a function to obtain mete_lambdas.
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

def mete_lambdas(s,b0=0.0001):
    '''This returns the METE lambdas for a given set of state variables.
    Inputs s as state variables, call S, N, or E
    Optional input of an initial beta, if we know it's going to be somewhere other than small positive.
    outputs array of lambdas'''
    beta = fsolve(beta_constraint,b0,args=s)[0]
    l2 = s['S']/(s['E']-s['N']) 
    ls = np.array([beta-l2,l2,0,0,0])
    return ls

# Transition functions
def f(n,e,s,p):
    '''Transition function for dN/dt. n and e are microscopic variables.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    return (p['b0']-p['d0']*s['E']/p['Ec'])*n/e**(1/3)+p['m0']/s['N']*n

def h(n,e,s,p):
    '''Transition function for dE/dt. n and e are microscopic variables.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    # This function requires beta. There are a few different ways to calculate this. 
    # Is it l1+l2 in Dynamete?
    # Should we use w1 instead of w10
    # For now we implemented the same thing Kaito did -- we will solve for beta in normal METE and use that
    # Use fsolve to solve for beta
    # Set initial guess
    b0i = 0.0001
    beta = fsolve(beta_constraint,b0i,args=s)[0]
    return (p['w0']-p['d0']*s['E']/p['Ec'])*n*e**(2/3)-p['w10']/np.log(1/beta)**(2/3)*n*e+p['m0']/s['N']*n

def q(n,e,s,p):
    '''Transition function for dS/dt. n and e are microscopic variables.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    For now this doesn't implement speciation models, ie. s1=s2=0'''
    # Set up kronecker delta in an easy way. Round n to nearest int, if it's 1 then include term
    kn1 = int(np.rint(n))==1
    return p['m0']*np.exp(-p['mu']*s['S']-np.euler_gamma) - kn1*p['d0']/p['Ec']*s['E']*s['S']/e**(1/3)

# These functions are needed to express the sum over R, nR, and n^2R analytically.
def R(n,e,l,s,p):
    '''Unnormalized struture function for DynaMETE.
    n,e are microscopic variables.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    return np.exp(-l[0]*n-l[1]*n*e-l[2]*f(n,e,s,p)-l[3]*h(n,e,s,p)-l[4]*q(n,e,s,p))

def Rsum(e,l,s,p):
    '''Unnormalized struture function for DynaMETE, summed over n.
    e is a microscopic variable.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    # Define exponent for lambdas 1-4
    l14 = l[0]+l[1]*e+l[2]*f(1,e,s,p)+l[3]*h(1,e,s,p)
    # 5 requires special treatment and a different definition depending on if n=1 or not
    l5_1 = l[4]*q(1,e,s,p)
    # Only the constant term
    l5_0 = l[4]*q(0,e,s,p)
    # Only the non-constant term
    l5_d = l5_1-l5_0
    return np.exp(-l14-l5_1)+np.exp(-2*l14-l5_0)*(1-np.exp(-l14*(s['N']-1)))/(1-np.exp(-l14))

def z(l,s,p):
    '''Calculate partition function for DynaMETE. 
    This function uses the analytic sum from Rsum and single integral over e.
    The integral is done with quad over log e.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu'''
    # Return only value not error
    return integrate.quad(lambda loge: np.exp(loge)*Rsum(np.exp(loge),l,s,p),0,np.log(s['E']))[0]

def nRsum(e,l,s,p):
    '''Unnormalized struture function for DynaMETE multiplied by n, then summed over n. 
    e is a microscopic variable.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    # Define exponent for lambdas 1-4, with n=1 since we've done sum over n already.
    l14 = l[0]+l[1]*e+l[2]*f(1,e,s,p)+l[3]*h(1,e,s,p)
    # 5 requires special treatment and a different definition depending on if n=1 or not
    l5_1 = l[4]*q(1,e,s,p)
    # Only the constant term
    l5_0 = l[4]*q(0,e,s,p)
    # Split up to make it easier (Note rewritten to be more interpretable from trick)
    # Trick is add n=1 and subtract n=1 with and without delta function
    t1 = np.exp(-l14-l5_1)
    t2 = np.exp(-l14-l5_0)
    t3fac = np.exp(-l14-l5_0)
    t3num = 1+s['N']*np.exp(-l14*(s['N']+2))-(s['N']+1)*np.exp(-l14*(s['N']+1))
    t3denom = (1-np.exp(-l14))**2
    return t1-t2+t3fac*t3num/t3denom

def n2Rsum(e,l,s,p):
    '''Unnormalized struture function for DynaMETE multiplied by n^2, then summed over n. 
    e is a microscopic variable.
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu '''
    # Define exponent for lambdas 1-4, with n=1 since we've done sum over n already.
    l14 = l[0]+l[1]*e+l[2]*f(1,e,s,p)+l[3]*h(1,e,s,p)
    # 5 requires special treatment and a different definition depending on if n=1 or not
    l5_1 = l[4]*q(1,e,s,p)
    # Only the constant term
    l5_0 = l[4]*q(0,e,s,p)
    # Split up terms
    # From doing the trick with the delta function
    t1 = np.exp(-l14-l5_1)
    t2 = np.exp(-l14-l5_0)
    t3fac = np.exp(-l14-l5_0)
    t3num = np.exp(-l14) + 1 - np.exp(-l14*(s['N']))*(s['N']+1)**2 + \
            np.exp(-l14*(s['N']+1))*(2*(s['N']+1)*s['N']-1) - s['N']**2*np.exp(-l14*(s['N']+2))
    t3denom = (1-np.exp(-l14))**3
    return t1 - t2 + t3fac*t3num/t3denom

# Calculate needed means
def get_means(l,s,p,ds,alln=True):
    '''
    This function gets all of the needed means over R for the constraints and the covariances. This includes 
    n e^{0,-1/2,2/3,1} R
    n^2 e^{0,-1/3,-2/3,1/3,2/3,1,4/3,5/3,2} R
    delta_{n,1} e^{-1/3,-2/3,1/3,2/3} R
    where
    l are lambdas
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu 
    ds are derivatives of state variables, call dS, dN, or dE

    Finally, to make this calculation faster, there is a way to only calculate the powers we need for the constraints (f,h,q),
    and ignore the ones for the covariances.
    To do this, set alln=False.

    Returns a pandas series with all of the means calculated.
    '''
    if alln:    
        # Make an array of powers of e that we need for linear n
        epow_arr_n = np.array([0,-1/3,2/3,1])
        # Make an array of powers of e that we need for quadratic n
        epow_arr_n2 = np.array([0,-1/3,-2/3,1/3,2/3,1,4/3,5/3,2])
        # Make an array of powers of e that we need for delta n
        epow_arr_dn = np.array([-1/3,-2/3,1/3,2/3])
    else:
        epow_arr_n = np.array([0,-1/3,2/3,1])
        epow_arr_n2 = np.array([])
        epow_arr_dn = np.array([-1/3])
        
    # Make arrays for storage
    nR_e_arr = np.zeros(len(epow_arr_n))
    # Always define this, even if we don't do the integrals, so the output is correct.
    n2R_e_arr = np.zeros(len(epow_arr_n2)) 
    dnR_e_arr = np.zeros(len(epow_arr_dn))
    
    # define logE
    logE = np.log(s['E'])
    
    # Loop over these and do the corresponding integrals
    # Switched to using vectorized version rather than looping
#    for i,epow in enumerate(epow_arr_n):
    nR_e_arr = integrate.quad_vec(lambda loge: np.exp((epow_arr_n+1)*loge)*nRsum(np.exp(loge),l,s,p),0,logE)[0]
    # Now n^2
#    for i,epow in enumerate(epow_arr_n2):
    n2R_e_arr = integrate.quad_vec(lambda loge: np.exp((epow_arr_n2+1)*loge)*n2Rsum(np.exp(loge),l,s,p),0,logE)[0]
    # Now for dn
#    for i,epow in enumerate(epow_arr_dn):
    dnR_e_arr = integrate.quad_vec(lambda loge: np.exp((epow_arr_dn+1)*loge)*R(1,np.exp(loge),l,s,p),0,logE)[0]
    
    # Get normalization and normalize
    Z = z(l,s,p)
    nR_e_arr /= Z
    n2R_e_arr /= Z
    dnR_e_arr /= Z
    # Save as pandas Series so we can call appropriately.
    if alln:
        labels = ['n','n_e13','ne23','ne', \
                  'n2','n2_e13','n2_e23','n2e13','n2e23','n2e','n2e43','n2e53','n2e2', \
                  'dn_e13','dn_e23','dne13','dne23']
    else:
        labels = ['n','n_e13','ne23','ne','dn_e13']
    means = pd.Series(np.concatenate([nR_e_arr,n2R_e_arr,dnR_e_arr]),index=labels)
    return means