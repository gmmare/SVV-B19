#Calculation for drag polar, thus CL-CD curve
from Reference_data_reader import get_stat_data
import numpy as np
import matplotlib.pyplot as plt
from math import *

#Wing parameters
S = 30
b = 15.911
A = b**2/S

stat_data = get_stat_data('20200310_V2.xlsx',28,35)

#Constants and standard sea level values
g0 = 9.80665 #[m/s2]
R = 287 #[J/kgK]
T0 = 288.15 #[K]
p0 = 101325 #[Pa]
rho0 = p0/(R*T0) #[kg/m3]

#temperature change per altitude
a1 = -0.0065    #0<11km

alt = stat_data[:,0]

rho_list = []
for i in range(len(alt)):
    h = 0.3048 * alt[i] #Conversion from ft to m

    #Troposphere calculations
    if h < 11000:
        T = T0 + a1*(h)
        p = p0*((T/T0)**(-g0/(a1*R)))
        rho = p/(R*T)

    #Tropopause calculations
    if h > 11000:
        T = T0 + a1*(h)
        p = p0*((T/T0)**(-g0/(a1*R)))
        T2 = T
        p2 = p*(exp((-g0/(R*T2))*(h-11000)))
        rho = p2/(R*T2)

    rho_list.append(rho)
    
#convert IAS to TAS
IAS = stat_data[:,1]
TAS = []
for i in range(len(IAS)):
    TAS.append(np.sqrt(rho0/rho_list[i])*IAS[i])

print(TAS)

#Choosing 
e = 0.8
CD0 = 0.04
CL_list = []
CD_list = []
cls = np.arange(0,1.5,0.1)
for i in cls:
    CL = i
    CL_list.append(CL**2)
    CD = CD0 + (CL**2)/(np.pi*A*e)
    CD_list.append(CD)
    
plt.plot(CD_list, CL_list)
plt.show()
