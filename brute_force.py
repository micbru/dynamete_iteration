'''
This file takes the brute force approach to iterate the lambdas. It simply optimizes at each step.
'''

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy import integrate
import DynaMETE_Rfunctions as rf
import means_covariances as mc

# Now the constraints
def constraints(l,s,p,ds):
    '''Return all constraints in an array. 
    l are lambdas
    s are state variables, call N, E, or S
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu 
    ds are derivatives of state variables, call dN, dE, or dS 
    '''
    # Calculate needed means
    m = rf.get_means(l,s,p,ds,alln=False)
    # Make corresponding array
    sums = np.array([m['n'],m['ne'],s['S']*mc.fm(s,p,m),s['S']*mc.hm(s,p,m),mc.qm(s,p,m)])
    
    # n constraint
    ncon = s['N']/s['S'] - sums[0]
    # e constraint
    econ = s['E']/s['S'] - sums[1]
    # dn constraint
    dncon= ds['dN'] - sums[2]
    # de constraint
    decon = ds['dE'] - sums[3]
    # ds constraint
    dscon = ds['dS'] - sums[4]
    
    return np.array([ncon,econ,dncon,decon,dscon])

# Iteration time!

def iterate(t,s0,p,dt=0.2,l0=np.array([]),ds0=np.array([])):
    '''
    This function will iterate DynaMETE t steps. Returns vectors of lambdas, state variables, and time derivatives.
    1. Update state variables using time derivatives
    2. Put new state variables into transition functions
    3. Update time derivatives
    4. Update structure function
    The first step is slightly different. We can either pass in only the state variables and parameters,
    in which case the theory assumes that we start in METE and calculates the corresponding lambdas and derivatives,
    or we can pass in lambdas and derivatives explicitly to iterate from anywhere.
    The former is basically the perturbation way, and the latter is a generic iteration.

    Inputs
    t is the integer number of steps
    l0 are initial lambdas
    s0 are initial state variables, call N, E, or S
    p are parameters, call b0, d0, m0, w0, w10, Ec, or mu. 
        Note that if we want to change this over time we have to put in an array for p, which isn't implemented here.
    ds0 are initial derivatives of state variables, call dN, dE, or dS 
    dt is how much of one year is a time step. The default is 0.2 to make sure the steps are relatively small.'''
    
    # Make arrays of lambdas, state variables, and derivatives to store and return.
    # Note that we need +1 to keep the 0th element also
    lambdas = np.zeros([t+1,5])
    states = pd.DataFrame(np.zeros([t+1,3]),columns=['S','N','E'])
    dstates = pd.DataFrame(np.zeros([t+1,3]),columns=['dS','dN','dE'])

    # Initialize zeroth element
    # Copy if present
    if bool(l0.size):        
        lambdas[0] = l0.copy()
    else:
        lambdas[0] = rf.mete_lambdas(s0)
    # Same for ds
    if bool(ds0.size):
        dstates.iloc[0] = ds0.copy()
    else:
        ds0 = pd.Series(np.zeros(3),index=['dS','dN','dE'])
        m0 = rf.get_means(lambdas[0],s0,p,ds0,alln=False)
        ds0['dN'] = s0['S']*mc.fm(s0,p,m0)
        ds0['dE'] = s0['S']*mc.hm(s0,p,m0)
        ds0['dS'] = mc.qm(s0,p,m0)
        dstates.iloc[0] = ds0.copy()
    # Now copy state variables
    states.iloc[0] = s0.copy()

    # Iterate t times.
    for i in range(t):
        # Print out progress
        print('Iteration {:.0f}/{:.0f}'.format(i+1,t))
    
        # First update state variables with time derivatives, multiplied by how much of one year we want to step by
        states.iloc[i+1] = states.iloc[i] + dt*dstates.iloc[i].values
        # Get temporary means for calculating new derivatives
        m_temp = rf.get_means(lambdas[i],states.iloc[i+1],p,dstates.iloc[i],alln=False)
        dstates.iloc[i+1] = np.array([mc.qm(states.iloc[i+1],p,m_temp),states.iloc[i+1]['S']*mc.fm(states.iloc[i+1],p,m_temp), \
                                  states.iloc[i+1]['S']*mc.hm(states.iloc[i+1],p,m_temp)])

        # Now time for new lambdas. Use old l as starting point, new s and ds
        l = fsolve(constraints,lambdas[i],args=(states.iloc[i+1],p,dstates.iloc[i+1]))
        lambdas[i+1] = l
        # Sanity check to make sure it worked.
        #print('Constraints (should be small!): {}'.format(constraints(l,states.iloc[i+1],p,dstates.iloc[i+1])))
        #print('')
        
    return lambdas,states,dstates