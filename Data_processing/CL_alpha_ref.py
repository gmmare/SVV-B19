from Reference_data_reader import get_stat_data
import numpy as np
from math import *
import matplotlib.pyplot as plt
from Reduction_functions import red_velocity, red_mass

#importing data
stat_data = get_stat_data("Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx", 28, 34)

#getting alpha values
alpha = stat_data[:,2]

W_total = 6679.7 #kg. starting weight
#==================calculating density==================
#Constants and standard sea level values
g0 = 9.80665 #[m/s2]
R = 287 #[J/kgK]
T0 = 288.15 #[K]
p0 = 101325 #[Pa]
rho0 = p0/(R*T0) #[kg/m3]
b = 15.911
S = 30
A = (b**2) / S
print("A", A)
CD_list_ref = [0.029279170219420586, 0.03005639762525321, 0.03222683891736139, 0.03502748902093308, 0.05606963918219729, 0.0791452770131129]
#[0.03581365139872909,
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
    rho_list.append(rho)

#==================converting velocity==================
IAS = stat_data[:,1] # this is calibrated veloctiy in kts
TAS = []
TAT = stat_data[:,-1] #true air temp in degree
red_vel = []
for i in range(len(IAS)):
    TAS.append((np.sqrt(rho0/rho_list[i])*IAS[i])*0.514444)
    red_vel.append(red_velocity(alt[i] * 0.3048, IAS[i]*0.514444, TAT[i] + 273,rho_list[i]))

#==================adjusting for weight==================
Weight_list = []
F_used = stat_data[:,-2]
for i in range(len(F_used)):
    Weight = (W_total - F_used[i] * 0.453592) * 9.81
    Weight_list.append(Weight)

#==================Calculating CL==================
CL_list = []
for i in range(len(alpha)):
    CL = Weight_list[i]/(0.5 * rho_list[i] * (red_vel[i] ** 2) * 30)
    CL_list.append(CL)

coef = np.polyfit(alpha,CL_list,1) # first item is the slope
poly1d_fn = np.poly1d(coef)

plt.plot(alpha,CL_list, 'ob',alpha, poly1d_fn(alpha)) # first item is the slope
print("CL alpha is:", coef[0])
print("---------------------")
plt.ylabel('CL')
plt.xlabel('alpha')
plt.title('CL-alpha plot')
plt.xlim(0, max(alpha)+1)
plt.show()
plt.clf()

#==================Calculating CD==================
CL_squared = []
for i in range(len(CL_list)):
    CL_squared.append(CL_list[i] ** 2)

#generating cd plots
coef_cd = np.polyfit(CL_squared,CD_list_ref,1)
poly1d_fn_cd = np.poly1d(coef_cd)

#cd plot
plt.plot(CL_squared,CD_list_ref, 'ob',CL_squared, poly1d_fn_cd(CL_squared))
plt.ylabel('CD')
plt.xlabel('CL^2')
plt.title('CD-CL^2 plot test data reference data')

print("e reference:",  1/(pi * A * coef_cd[0]))
print("CD0 reference:", coef_cd[1])
print("---------------------")
plt.show()
plt.clf()