from ._EpiModels_common_imports import *

comps = [
    'S0', 'SVR0', 'SVC0', 
    'S1', 'SVR1', 'SVC1', 
    'I0', 'IVR0', 'IVC0', 
    'I1', 'IVR1', 'IVC1',
]
n_comps = len(comps)

susceptible0_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('S') and comp.endswith('0')]
recovered_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('S') and comp.endswith('1')]
susceptible_idx = susceptible0_idx + recovered_idx
carrier0_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('I') and comp.endswith('0')]
carrier1_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('I') and comp.endswith('1')]
carrier_idx = carrier0_idx + carrier1_idx
vaccinatedR_idx = [idx for idx, comp in enumerate(comps) if 'V' in comp and 'R' in comp]
vaccinatedC_idx = [idx for idx, comp in enumerate(comps) if 'V' in comp and 'C' in comp]
vaccinated_idx = vaccinatedR_idx + vaccinatedC_idx

def ODE_system(t, y, b, m, K, alpha, beta, gamma, delta, chi, z, uR, uC, epsR, epsC, n_comps):
    S0, SVR0, SVC0, S1, SVR1, SVC1, I0, IVR0, IVC0, I1, IVR1, IVC1 = y.reshape(n_comps,-1)
    N = S0 + SVR0 + SVC0 + S1 + SVR1 + SVC1 + I0 + IVR0 + IVC0 + I1 + IVR1 + IVC1
    I = I0 + IVR0 + IVC0 + I1 + IVR1 + IVC1

    FOI = beta * chi * (K @ (I / N))

    dS0      = b - (FOI + uR + uC + m) * S0 + (1 - alpha) * gamma * I0 + delta * S1
    dSVR0    = - (FOI * (1 - epsR) + m) * SVR0 + (1 - alpha) * gamma * IVR0 + delta * SVR1 + uR * S0
    dSVC0    = - (FOI * (1 - epsC) + m) * SVC0 + (1 - alpha) * gamma * IVC0 + delta * SVC1 + uC * S0

    dS1      = - (z * FOI + delta + uR + uC + m) * S1 + alpha * gamma * I0 + gamma * I1
    dSVR1    = - (z * FOI * (1 - epsR) + delta + m) * SVR1 + alpha * gamma * IVR0 + gamma * IVR1 + uR * S1
    dSVC1    = - (z * FOI * (1 - epsC) + delta + m) * SVC1 + alpha * gamma * IVC0 + gamma * IVC1 + uC * S1
    
    dI0      = - (gamma + uR + uC + m) * I0 + FOI * S0 + delta * I1
    dIVR0    = - (gamma + m) * IVR0 + FOI * (1 - epsR) * SVR0 + delta * IVR1 + uR * I0
    dIVC0    = - (gamma + m) * IVC0 + FOI * (1 - epsC) * SVC0 + delta * IVC1 + uC * I0
    
    dI1      = - (gamma + delta + uR + uC + m) * I1 + z * FOI * S1
    dIVR1    = - (gamma + delta + m) * IVR1 + z * FOI * (1 - epsR) * SVR1 + uR * I1
    dIVC1    = - (gamma + delta + m) * IVC1 + z * FOI * (1 - epsC) * SVC1 + uC * I1
    
    dy = np.array([dS0, dSVR0, dSVC0, dS1, dSVR1, dSVC1, dI0, dIVR0, dIVC0, dI1, dIVR1, dIVC1]).flatten()

    return dy