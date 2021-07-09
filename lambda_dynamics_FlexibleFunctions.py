'''
This file defines three functions that get the needed means for lambda dynamics,
get the needed matrix and vector for inversion, and define the necessary iteration scheme.
This version uses the flexible definitions for cov and mean, not the hard coded ones. This will
make it much slower, but it also means that we can change f,h,q much easier.
This file imports the Flexible Functions version of the DynaMETE Rfunctions code, but is otherwise
nearly the same as the normal lambda_dynamics code. The other main difference is that dXdt_min and
get_ss_params have to be edited if there are new parameters to minimize over to get the initial parameters
that make the derivatives zero.
'''

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import root #<- better than fsolve apparently.
import DynaMETE_Rfunctions_FlexibleFunctions as rf

def get_dl_matrix_vector(l,s,p,ds):
    '''
    Calculates the matrix and vector of covariances needed for lambda dynamics
    l are lambdas as an array, so l[0] is lambda_1 etc.
    s are state variables, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    ds are derivatives of state variables, call dS, dN, or dE
    returns array and vector that need to be solved.
    '''
    # Calculate normalization
    z = rf.mean_pow(0,0,l,s,p)
    # Calculate rows of covariances
    # First define the covariances since many of them repeat and this will make the rows easier to read
    covnn = rf.cov_pow([1,1],[0,0],l,s,p,z)
    covnen = rf.cov_pow([1,1],[1,0],l,s,p,z)
    covnene = rf.cov_pow([1,1],[1,1],l,s,p,z)
    covnf = rf.cov(lambda n,e,s,p: n,rf.f,l,s,p,z)
    covnh = rf.cov(lambda n,e,s,p: n,rf.h,l,s,p,z)
    covnq = rf.cov(lambda n,e,s,p: n,rf.q,l,s,p,z)
    covnef = rf.cov(lambda n,e,s,p: n*e,rf.f,l,s,p,z)
    covneh = rf.cov(lambda n,e,s,p: n*e,rf.h,l,s,p,z)
    covneq = rf.cov(lambda n,e,s,p: n*e,rf.q,l,s,p,z)
    covff = rf.cov(rf.f,rf.f,l,s,p,z)
    covhh = rf.cov(rf.h,rf.h,l,s,p,z)
    covqq = rf.cov(rf.q,rf.q,l,s,p,z)
    covfh = rf.cov(rf.f,rf.h,l,s,p,z)
    covfq = rf.cov(rf.f,rf.q,l,s,p,z)
    covhq = rf.cov(rf.h,rf.q,l,s,p,z)
    # Then put them in rows
    r1 = np.array([covnf,covnef,covff,covfh,covfq])
    r2 = np.array([covnh,covneh,covfh,covhh,covhq])
    r3 = np.array([covnq,covneq,covfq,covhq,covqq])
    r4 = np.array([covnn,covnen,covnf,covnh,covnq])
    r5 = np.array([covnen,covnene,covnef,covneh,covneq])

    # Also calculate and return vector for Ax = B
    # Covariances. Note here we also have to pass ds since the derivatives take 5 variables
    # It has to go after z because *args is after z
    covndf = rf.cov(lambda n,e,s,p,ds: n,rf.dfdt,l,s,p,z,ds)
    covndh = rf.cov(lambda n,e,s,p,ds: n,rf.dhdt,l,s,p,z,ds)
    covndq = rf.cov(lambda n,e,s,p,ds: n,rf.dqdt,l,s,p,z,ds)
    covnedf = rf.cov(lambda n,e,s,p,ds: n*e,rf.dfdt,l,s,p,z,ds)
    covnedh = rf.cov(lambda n,e,s,p,ds: n*e,rf.dhdt,l,s,p,z,ds)
    covnedq = rf.cov(lambda n,e,s,p,ds: n*e,rf.dqdt,l,s,p,z,ds)
    # Vectors, including normalization in proper spot.
    v4 = ds['dS']*s['N']/s['S']**2-ds['dN']/s['S']-covndf*l[2]-covndh*l[3]-covndq*l[4]
    v5 = ds['dS']*s['E']/s['S']**2-ds['dE']/s['S']-covnedf*l[2]-covnedh*l[3]-covnedq*l[4]
    
    return np.array([r1,r2,r3,r4,r5]),np.array([0,0,0,v4,v5])

# Give defaults to l and ds so that by default we assume the lambdas and the derivatives have to be calculated from
# the corresponding state variable. Allow them as optional arguments if we want to iterate from a specific state.
def iterate(t,s,p,dt=0.2,l=np.array([]),ds=np.array([]),det=False):
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
    Determinant of dlambda matrix to see if something weird is happening there. This may be temporary, so I made it optional.
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
        dsd.iloc[0] = rf.get_dXdt(ld[0],s,p)
    # Now copy state variables
    sd.iloc[0] = s.copy()
    
    # Set up determinant matrix
    if det:
        det_ld = np.zeros([t])
    
    # Iterate t times
    for i in np.arange(t):
        print("Iteration {}/{}".format(i+1,t))
        # Get required matrix and vector
        dl_mat,dl_vec = get_dl_matrix_vector(ld[i],sd.iloc[i],p,dsd.iloc[i])
        # Solve for dlambdas/dt
        dl = linalg.solve(dl_mat,dl_vec)
        # Get determinant to output, if det is true
        if det:
            det_ld[i] = linalg.det(dl_mat-dl_vec*np.identity(5))
        
        
        # Now update state variables
        sd.iloc[i+1] = sd.iloc[i] + dt*dsd.iloc[i].values
        # Update derivatives from means over f and h
        dsd.iloc[i+1] = rf.get_dXdt(ld[i],sd.iloc[i+1],p)
        # Update lambdas
        ld[i+1] = ld[i] + dt*dl
    if det:
        return ld,sd,dsd,det_ld
    else:
        return ld,sd,dsd

def stability_test(ld,sd,p):
    '''
    This function outputs <n> and <ne> for each step to check that lambda dynamics is satisfying the
    constraints it has to at each time t. Note this is a slow function.
    ld are lambdas as an array with length t and 5 columns, so l[0,0] is lambda_1 at time 0 etc.
    ss are state variables at each time t, call S, N, or E
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu
    '''
    # Get total time
    t = len(sd)
    # Set up arrays
    N_S = np.zeros(t)
    E_S = np.zeros(t)
    for i in np.arange(t):
        # Get means at each time, normalized.
        z = rf.mean_pow(0,0,ld[i],sd.iloc[i],p)
        N_S[i] = rf.mean_pow(1,0,ld[i],sd.iloc[i],p,z)
        E_S[i] = rf.mean_pow(1,1,ld[i],sd.iloc[i],p,z)
    return N_S, E_S


def dXdt_min(pmin,l,s,p):
    '''
    This function is needed for get_ss_params as it changes which parameters are minimized over. It essentially
    just changes the order of the inputs from rf.get_dXdt.
    pmin is the parameters optimized over. Right now that is m0, w10, and mu in order.
    '''
    # Change p to incorporate the new pmin.
    pnew = p.copy()
    pnew['m0'] = pmin[0]
    pnew['w10'] = pmin[1]
    pnew['mu'] = pmin[2]
    return rf.get_dXdt(l,s,pnew).values
    
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
