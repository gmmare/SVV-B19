from Reference_data_reader import get_stat_data
import numpy as np
from math import *
import matplotlib.pyplot as plt

#importing data
stat_data = get_stat_data("20200311_V1.xlsx", 28, 35)

#getting alpha values
alpha = stat_data[:,2]

W_total = 9165 + 80 + 102 + 60 + 67 + 59 + 78 + 66 + 86 + 87.5

#==================calculating density==================
#Constants and standard sea level values
g0 = 9.80665 #[m/s2]
R = 287 #[J/kgK]
T0 = 288.15 #[K]
p0 = 101325 #[Pa]
rho0 = p0/(R*T0) #[kg/m3]

#temperature change per altitude
a1 = -0.0065    #0<11km
a2 = 0          #11<20km

#getting the altitude
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


#==================converting velocity==================
IAS = stat_data[:,1]
TAS = []
for i in range(len(IAS)):
    TAS.append((np.sqrt(rho0/rho_list[i])*IAS[i])*0.514444)

#==================adjusting for weight==================
Weight_list = []
F_used = stat_data[:,-2]
for i in range(len(F_used)):
    Weight_list.append((W_total - F_used[i] * 0.453592))

#==================Calculating CL==================
CL_list = []
for i in range(len(alpha)):
    CL = Weight_list[i]/(0.5 * rho_list[i] * (TAS[i] ** 2) * 30)
    CL_list.append(CL)

coef = np.polyfit(alpha,CL_list,1)
poly1d_fn = np.poly1d(coef)

plt.plot(alpha,CL_list, 'ob', alpha, poly1d_fn(alpha))

print("CL alpha is:")
print(coef)

plt.ylabel('CL')
plt.xlabel('alpha')
plt.title('CL-alpha plot')
plt.xlim(0, max(alpha)+1)
plt.ylim(0, 0.2)
plt.show()
