import numpy as np

"""
alias | variable name | value | units
"""
T     = AIR_TEMPERATURE     = 300      # K
eps_0 = VACUUM_PERMITTIVITY = 8.85e-14 # F/cm
k_B   = BOLTZMANN_CONSTANT  = 1.38e-23 # m^2*kg/(s^2*K)
q     = ELECTRON_CHARGE     = 1.6e-19  # C
V_T   = THERMAL_VOLTAGE     = k_B * T / q # V

def debye_length(eps: float, n_i: float) -> float:
    """
    Calculate a material's Debye length in cm
    """
    return np.sqrt(k_B*T*eps*eps_0/(q**2*n_i))

class Material:
    """
    alias | variable name | value | units
    """
    n_i      = INTRINSIC_CARRIER_CONCENTRATION               = 1e10  # cm^-3
    phi_t    = SRH_RECOMBINATION_TRAP_LEVEL                  = 21.25 # qV/(kT) (thermal voltage)
    eps      = RELATIVE_PERMITTIVITY                         = 11.68
    N_D      = DONOR_CONCENTRATION                           = 1e16  # cm^-3
    N_A      = ACCEPTOR_CONCENTRATION                        = 1e16  # cm^-3
    N_CD     = DONOR_CORNER_DOPING_CONCENTRATION             = 1e17  # cm^-3
    N_CA     = ACCEPTOR_CORNER_DOPING_CONCENTRATION          = 5e16  # cm^-3
    mu_0n    = MAXIMUM_ELECTRON_MOBILITY                     = 1400  # cm^2/(V*s)
    mu_0p    = MAXIMUM_HOLE_MOBILITY                         = 480   # cm^2/(V*s)
    D_0      = REFERENCE_DIFFUSIVITY                         = 36    # cm^2/s
    v_scat_n = SCATTER_LIMITED_ELECTRON_DRIFT_VELOCITY       = 1e7   # cm/s
    v_scat_p = SCATTER_LIMITED_HOLE_DRIFT_VELOCITY           = 7e6   # cm/s
    tau_n    = EXCESS_ELECTRON_LIFETIME                      = 7e-5  # s
    tau_p    = EXCESS_HOLE_LIFETIME                          = 7e-5  # s
    A_n      = FACTOR_OF_MAXIMUM_ELECTRON_MOBILITY_REDUCTION = mu_0n / 93. # cm^2/(V*s)
    A_p      = FACTOR_OF_MAXIMUM_HOLE_MOBILITY_REDUCTION     = mu_0p / 30. # cm^2/(V*s)

    L_D      = DEBYE_LENGTH                                  = debye_length(eps, n_i)  # cm