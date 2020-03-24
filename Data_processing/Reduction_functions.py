
from Cit_par import *
import Reference_data_reader as ref_data
reference_data = ref_data.reference_data

rho0    = 1.2250            # air density at sea level [kg/m^3]
lambd   = -0.0065           # temperature gradient in ISA [K/m]
Temp0   = 288.15            # temperature at sea level in ISA [K]
R       = 287.05            # specific gas constant [m^2/sec^2K]
g       = 9.81              # [m/sec^2] (gravity constant)
gam     = 1.4               # ratio of specific heats for air [-]
P0      = 101325            # pressure at sea level [Pa]
W_s     = 60500             # Standard weight in N
m_fs    = 0.048             # standard fuel flow in kg/sec

def red_velocity(hp, V_c, T_m):
    '''
    :param hp: pressure height
    :param V_c: calibrated  velocity
    :param T_m: measured temperature
    :param rho: density
    :return: equivalant velocity
    '''

    P = P0 * ((1 + ((lambd * hp)/Temp0)) ** (-g/(lambd*R)))

    M = sqrt((2 / (gam - 1)) * ((1 + (P0/P)*(((1 + (gam - 1)/(2*gam) * (rho0/P0) * (V_c ** 2)) ** (gam/(gam - 1))) - 1)) ** ((gam - 1)/gam) - 1))

    T = T_m / ((1 + (gam-1)/2) * (M ** 2))

    V_true = M * sqrt(gam * R * T)


    rho1 = P/(R * T)

    V_e = V_true * sqrt(rho1/rho0)

    return V_e, V_true, rho1#comment out V_true and rho1 for cl alpha plots

def red_mass(V_e, W): # for the elevator trim curve

    '''
    :param V_e: equivalent airspeed
    :param W: weight of the aircraft in [N]
    :return: equivalent airspeed to be used in the elevator trim curve
    '''
    V_e2 = V_e * sqrt(W_s/W)

    return V_e2

# def red_elev_defl(C_m_delta, C_m_0, C_m_alpha, C_n_alpha):
#
#     delta_e_eq = - 1/C_m_delta * (C_m_0  + (C_m_alpha/C_n_alpha) * (W/(0.5 * rho * (V_e ** 2) * S)) + C_m_delta_f * delta_f + C_m_T_c * T_c_s)