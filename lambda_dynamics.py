'''
This file defines three functions that get the needed means for lambda dynamics,
get the needed matrix and vector for inversion, and define the necessary iteration scheme.
'''

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import root
import means_covariances as mc
import DynaMETE_Rfunctions as rf

def get_dl_matrix_vector(l,s,p,ds,m):
    '''
    Calculates the matrix and vector of covariances needed for lambda dynamics
    l are lambdas as an array, so l[0] is lambda_1 etc.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    ds are derivatives of state variables, call dS, dN, or dE
    m are means which contain all sums over n and e
    returns array and vector that need to be solved.
    '''
    # Calculate rows of covariances
    r1 = np.array([mc.covnf(s,p,m),mc.covnef(s,p,m),mc.covff(s,p,m),mc.covfh(s,p,m),mc.covfq(s,p,m)])
    r2 = np.array([mc.covnh(s,p,m),mc.covneh(s,p,m),mc.covfh(s,p,m),mc.covhh(s,p,m),mc.covhq(s,p,m)])
    r3 = np.array([mc.covnq(s,p,m),mc.covneq(s,p,m),mc.covfq(s,p,m),mc.covhq(s,p,m),mc.covqq(s,p,m)])
    r4 = np.array([mc.covnn(m),mc.covnen(m),mc.covnf(s,p,m),mc.covnh(s,p,m),mc.covnq(s,p,m)])
    r5 = np.array([mc.covnen(m),mc.covnene(m),mc.covnef(s,p,m),mc.covneh(s,p,m),mc.covneq(s,p,m)])

    # Also calculate and return vector for Ax = B
    v4 = ds['dS']*s['N']/s['S']**2-ds['dN']/s['S'] - \
         mc.covndf(s,p,ds,m)*l[2]-mc.covndh(s,p,ds,m)*l[3]-mc.covndq(s,p,ds,m)*l[4]
    v5 = ds['dS']*s['E']/s['S']**2-ds['dE']/s['S'] - \
         mc.covnedf(s,p,ds,m)*l[2]-mc.covnedh(s,p,ds,m)*l[3]-mc.covnedq(s,p,ds,m)*l[4]
    return np.array([r1,r2,r3,r4,r5]),np.array([0,0,0,v4,v5])

# Give defaults to l and ds so that by default we assume the lambdas and the derivatives have to be calculated from
# the corresponding state variable. Allow them as optional arguments if we want to iterate from a specific state.
def iterate(t,s,p,dt=0.2,l=np.array([]),ds=np.array([])):
    '''
    Iterates the lambda dynamics theory. This works by updating lambdas from the derivatives obtained from the
    covariance matrix calculated in get_dl_matrix_vector. The scheme is:
    1. Get derivatives of lambdas
    2. Update state variables s_t+1 = s_t + ds_t
    3. Update derivatives with new state variables, ie. dN/dt = S <f(X_t+1,lambda_t)>
    4. Update lambdas as l_t+1 = l_t + dt dl/dt
    The first step is slightly different. We can either pass in only the state variables and parameters,
    in which case the theory assumes that we start in METE and calculates the corresponding lambdas and derivatives,
    or we can pass in lambdas and derivatives explicitly to iterate from anywhere.
    The former is basically the perturbation way, and the latter is a generic iteration.
    Inputs
    t is the number of iteration steps to take
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    Optional:
    l are lambdas as an array, so l[0] is lambda_1 etc.
    ds are derivatives of state variables, call dS, dN, or dE
    dt is the length of the time step, 0.2 by default.
    Outputs:
    lambdas, state variables, and their derivatives at each time t. 
    The arrays are length t+1 since it prints the initial values.
    '''

    # Set up dataframes for storage
    ld = np.zeros([t+1,5])
    sd = pd.DataFrame(np.zeros([t+1,3]),columns=['S','N','E'])
    dsd = pd.DataFrame(np.zeros([t+1,3]),columns=['dS','dN','dE'])

    # Initialize zeroth element
    # Copy if present
    if bool(l.size):        
        ld[0] = l.copy()
    else:
        ld[0] = rf.mete_lambdas(s)
    # Same for ds
    if bool(ds.size):
        dsd.iloc[0] = ds.copy()
    else:
        ds0 = pd.Series(np.zeros(3),index=['dS','dN','dE'])
        m0 = rf.get_means(ld[0],s,p,ds0,alln=False)
        ds0['dN'] = s['S']*mc.fm(s,p,m0)
        ds0['dE'] = s['S']*mc.hm(s,p,m0)
        ds0['dS'] = mc.qm(s,p,m0)
        dsd.iloc[0] = ds0.copy()
    # Now copy state variables
    sd.iloc[0] = s.copy()
    
    # Iterate t times
    for i in np.arange(t):
        print("Iteration {}/{}".format(i+1,t))
        # First get the required means to calculate the matrix and vector that need to be solved
        m = rf.get_means(ld[i],sd.iloc[i],p,dsd.iloc[i])
        dl_mat,dl_vec = get_dl_matrix_vector(ld[i],sd.iloc[i],p,dsd.iloc[i],m)
        # Solve for dlambdas/dt
        dl = linalg.solve(dl_mat,dl_vec)
        
        # Now update state variables
        sd.iloc[i+1] = sd.iloc[i] + dt*dsd.iloc[i].values
        # Get new means with old lambdas but new state variables
        # To do this faster I may want to make a turnoff for n^2 and dn terms when I don't need covariances.
        m_new = rf.get_means(ld[i],sd.iloc[i+1],p,dsd.iloc[i],alln=False)
        # Update derivatives from means over f and h
        dsd.iloc[i+1] = np.array([mc.qm(sd.iloc[i+1],p,m_new),sd.iloc[i+1]['S']*mc.fm(sd.iloc[i+1],p,m_new), \
                                  sd.iloc[i+1]['S']*mc.hm(sd.iloc[i+1],p,m_new)])
        # Update lambdas
        ld[i+1] = ld[i] + dt*dl
    return ld,sd,dsd

def stability_test(ld,sd,p,dsd):
    '''
    This function outputs <n> and <ne> for each step to check that lambda dynamics is satisfying the
    constraints it has to at each time t. Note this is a slow function.
    ld are lambdas as an array with length t and 5 columns, so l[0,0] is lambda_1 at time 0 etc.
    ss are state variables at each time t, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    dsd are derivatives of state variables at each time t, call dS, dN, or dE
    '''
    # Get total time
    t = len(sd)
    # Set up arrays
    N_S = np.zeros(t)
    E_S = np.zeros(t)
    for i in np.arange(t):
        # Get means at each time
        mX_S = rf.get_means(ld[i],sd.iloc[i],p,dsd.iloc[i],alln=False)
        N_S[i] = mX_S['n']
        E_S[i] = mX_S['ne']
    return N_S, E_S

def dXdt_min(pmin,l,s,p):
    '''
    This function is needed for get_ss_params as it changes which parameters are minimized over.
    pmin is the parameters optimized over. Right now that is m0, w10, and mu in order.
    '''
    # Change p to incorporate the new pmin.
    pnew = p.copy()
    pnew['m0'] = pmin[0]
    pnew['w10'] = pmin[1]
    pnew['mu'] = pmin[2]
    
    mX = rf.get_means(l,s,pnew,pd.Series({'dN':0,'dE':0,'dS':0}),alln=False)
    fm = s['S']*mc.fm(s,pnew,mX)
    hm = s['S']*mc.hm(s,pnew,mX)
    qm = mc.qm(s,pnew,mX)
    return np.array([fm,hm,qm])
    
def get_ss_params(s,pi):
    '''
    This function gets the steady state parameter values so that we can start from something roughly stable.
    It does this by fixing Ec, b0, d0, and w0, and solving the constraints for m0, w10, and mu.
    Note that which parameters are optimized over can be changed according to the function dXdt_min, which is what
    we are minimizing in practice.
    Inputs state variables and initial parameters guess
    Outputs new parameters
    '''
    # Get initial lambdas
    lmete = rf.mete_lambdas(s)
    sol = root(dXdt_min,[pi['m0'],pi['w10'],pi['mu']],args=(lmete,s,pi))
    pnew = pi.copy()
    pnew['m0'] = sol.x[0]
    pnew['w10'] = sol.x[1]
    pnew['mu'] = sol.x[2]
    return pnew
