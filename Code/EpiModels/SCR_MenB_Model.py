from ._EpiModels_common_imports import *

comps = [
    'S', 'SVR', 'SVC', 
    'I', 'IVR', 'IVC', 
    'R', 'RVR', 'RVC'
]
n_comps = len(comps)

susceptible_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('S')]
carrier_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('I')]
recovered_idx = [idx for idx, comp in enumerate(comps) if comp.startswith('R')]
noncarrier_idx = susceptible_idx + recovered_idx
VR_idx = [idx for idx, comp in enumerate(comps) if 'V' in comp and comp.endswith('R')]
VC_idx = [idx for idx, comp in enumerate(comps) if 'V' in comp and comp.endswith('C')]
vaccinated_idx = VR_idx + VC_idx

def ODE_system(t, y, b, m, K, alpha, beta, gamma, delta, chi, z, uR, uC, epsR, epsC, n_comps):
    S, SVR, SVC, B, BVR, BVC, R, RVR, RVC = y.reshape(n_comps,-1)
    N = S + SVR + SVC + B + BVR + BVC + R + RVR + RVC
    I = B + BVR + BVC

    FOI = beta * chi * (K @ (I / N))

    # Equations
    dS      = b - (FOI + uR + uC + m) * S + gamma * (1 - alpha) * B + delta * R
    dSVR    = - (FOI * (1 - epsR) + m) * SVR + uR * S + gamma * (1 - alpha) * BVR + delta * RVR
    dSVC    = - (FOI * (1 - epsC) + m) * SVC + uC * S + gamma * (1 - alpha) * BVC + delta * RVC

    dB      = - (gamma + uR + uC + m) * B + FOI * (S + z * R)
    dBVR    = - (gamma + m) * BVR + uR * B + FOI * (1 - epsR) * (SVR + z * RVR)
    dBVC    = - (gamma + m) * BVC + uC * B + FOI * (1 - epsC) * (SVC + z * RVC)

    dR      = - (delta + FOI * z + uR + uC + m) * R + gamma * alpha * B
    dRVR    = - (delta + FOI * (1 - epsR) * z + m) * RVR + uR * R + gamma * alpha * BVR
    dRVC    = - (delta + FOI * (1 - epsC) * z + m) * RVC + uC * R + gamma * alpha * BVC

    dy = np.array([dS, dSVR, dSVC, dB, dBVR, dBVC, dR, dRVR, dRVC]).flatten()

    return dy