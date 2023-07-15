import numpy as np

"""
alias | variable name | value | units
"""
T     = AIR_TEMPERATURE     = 300      # K
eps_0 = VACUUM_PERMITTIVITY = 8.85e-12 # F/m
k_B   = BOLTZMANN_CONSTANT  = 1.38e-23 # m^2*kg/(s^2*K)
q     = ELECTRON_CHARGE     = 1.6e-19  # C

def debye_length(eps: float, n_i: float) -> float:
    """
    Calculate a material's Debye length in cm
    """
    return np.sqrt(k_B*T*eps*eps_0/(q**2*n_i*100**3)) * 1e2

def reference_diffusivity(n_i: float) -> float:
    """
    Calculate a reference diffusivity
    """
    return 1 / (q * n_i)

def diffusivity(mu: float) -> float:
    """
    Calculate the diffusivity from mobility using the Einstein relation
    """
    return mu * k_B * T / q

class Material:
    """
    alias | variable name | value | units
    """
    n_i      = INTRINSIC_CARRIER_CONCENTRATION               = 1e10  # cm^-3
    phi_t    = SRH_RECOMBINATION_TRAP_LEVEL                  = 21.25 # qV/(kT) (thermal voltage)
    eps      = RELATIVE_PERMITTIVITY                         = 11.68
    N_D      = DONOR_CONCENTRATION                           = 1e6   # n_i (intrinsic carrier concentration)
    N_A      = ACCEPTOR_CONCENTRATION                        = 1e6   # n_i
    N_CD     = DONOR_CORNER_DOPING_CONCENTRATION             = 1.1e5 # n_i
    N_CA     = ACCEPTOR_CORNER_DOPING_CONCENTRATION          = 1.1e5 # n_i
    mu_0n    = MAXIMUM_ELECTRON_MOBILITY                     = 1400  # cm^2/(V*s)
    mu_0p    = MAXIMUM_HOLE_MOBILITY                         = 450   # cm^2/(V*s)
    v_scat_n = SCATTER_LIMITED_ELECTRON_DRIFT_VELOCITY       = 1e7   # cm/s
    v_scat_p = SCATTER_LIMITED_HOLE_DRIFT_VELOCITY           = 7e6   # cm/s
    tau_n    = EXCESS_ELECTRON_LIFETIME                      = 7e-5  # s
    tau_p    = EXCESS_HOLE_LIFETIME                          = 7e-5  # s
    A_n      = FACTOR_OF_MAXIMUM_ELECTRON_MOBILITY_REDUCTION = 1305  # cm^2/(V*s)
    A_p      = FACTOR_OF_MAXIMUM_HOLE_MOBILITY_REDUCTION     = 420   # cm^2/(V*s)

    L_D      = DEBYE_LENGTH                                  = debye_length(eps, n_i)       # cm
    D_0      = REFERENCE_DIFFUSION_COEFFICIENT               = reference_diffusivity(n_i)   # cm^2/s