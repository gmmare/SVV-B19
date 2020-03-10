import matplotlib.pyplot as plt
from Cit_par import *
import numpy as np
import Reference_data_reader as ref_data
reference_data = ref_data.reference_data
import Cit_par as Data

rho0    = 1.2250            # air density at sea level [kg/m^3]
lambd   = -0.0065           # temperature gradient in ISA [K/m]
Temp0   = 288.15            # temperature at sea level in ISA [K]
R       = 287.05            # specific gas constant [m^2/sec^2K]
g       = 9.81              # [m/sec^2] (gravity constant)
gam     = 1.4               # ratio of specific heats for air [-]
P0      = 101325            # pressure at sea level [Pa]

def red_velocity(hp, V_c, T_m, rho):
    '''

    :param hp: pressure height
    :param V_c: calibrated  velocity
    :param T_m: measured temperature
    :param rho: density
    :return: equivilant velocity
    '''



    P = P0 * ((1 + ((lambd * hp)/Temp0)) ** (-g/(lambd*R)))
    print(P)

    M =sqrt( * ((1 + (P0/P) * (((1 + (gam - 1)/(2*gam)) * (rho0/P0) * (V_c ** 2)) ** ((gam - 1)/gam) - 1) ** ((gam-1)/gam)) -1))

    (2 / (gam - 1))
    1 + (P0/P)
    1 + ((gam - 1)/(2*gam)


    M =





    T = T_m / ((1 + (gam-1)/2) * (M ** 2))

    V_true = M * sqrt(gam * R * T)

    V_e = V_true * sqrt(rho/rho0)

    return V_e

print(red_velocity(1000, 60, 285, 1.033))