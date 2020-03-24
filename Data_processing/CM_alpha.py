from Reference_data_reader import get_stat_data
import numpy as np
from Reduction_functions import red_velocity, red_mass
from math import *
import matplotlib.pyplot as plt

#importing data
stat_data = get_stat_data("20200311_V1.xlsx", 75, 77)
stat_data_ref = get_stat_data("Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx", 75, 77)

W_total = 6471 #(9165 + 80 + 102 + 60 + 67 + 59 + 78 + 66 + 86 + 87.5)
W_total_ref = 6679.7

#getting alpha values
alpha = stat_data[:,2]
alpha_ref = stat_data[:,2]
#==================calculating density==================
#Constants and standard sea level values
g0 = 9.80665 #[m/s2]
R = 287 #[J/kgK]
T0 = 288.15 #[K]
p0 = 101325 #[Pa]
rho0 = p0/(R*T0) #[kg/m3]

X_cg_config1 = 214.614 #inch
X_cg_config2 = 196.999

X_cg_config1_ref = 210.04 #inch
X_cg_config2_ref = 187.74
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
alt_ref = stat_data_ref[:,0]

rho_list = []
rho_list_ref = []
for i in range(len(alt)):
    h = 0.3048 * alt[i] #Conversion from ft to m

    #Troposphere calculations
    if h < 11000:
        T = T0 + a1*(h)
        p = p0*((T/T0)**(-g0/(a1*R)))
        rho = p/(R*T)
    rho_list.append(rho)

for i in range(len(alt_ref)):
    h = 0.3048 * alt_ref[i] #Conversion from ft to m

    #Troposphere calculations
    if h < 11000:
        T = T0 + a1*(h)
        p = p0*((T/T0)**(-g0/(a1*R)))
        rho = p/(R*T)
    rho_list_ref.append(rho)


#==================converting velocity==================
IAS = stat_data[:,1] # this is calibrated veloctiy in kts
TAT = stat_data[:,-1] #true air temp in degree
red_vel = []

IAS_ref = stat_data_ref[:,1] # this is calibrated veloctiy in kts
TAT_ref = stat_data_ref[:,-1] #true air temp in degree
red_vel_ref = []
for i in range(len(IAS)):
    red_vel.append(red_velocity(alt[i] * 0.3048, IAS[i]*0.514444, TAT[i] + 273,rho_list[i]))
    red_vel_ref.append(red_velocity(alt_ref[i] * 0.3048, IAS_ref[i] * 0.514444, TAT_ref[i] + 273, rho_list_ref[i]))
#==================adjusting for weight==================
Weight_list = []
Weight_list_ref = []
F_used = stat_data[:,-2]
F_used_ref = stat_data_ref[:,-2]
for i in range(len(F_used)):
    Weight = (W_total - F_used[i] * 0.453592)*9.81
    Weight_list.append(Weight)
    Weight_ref = (W_total_ref - F_used_ref[i] * 0.453592) * 9.81
    Weight_list_ref.append(Weight_ref)

#==================Calculating CL==================
CL_list = []
CL_list_ref = []
for i in range(len(alpha)):
    CL = Weight_list[i]/(0.5 * rho_list[i] * (red_vel[i] ** 2) * 30)
    CL_list.append(CL)
    CL_ref = Weight_list_ref[i] / (0.5 * rho_list_ref[i] * (red_vel_ref[i] ** 2) * 30)
    CL_list_ref.append(CL_ref)

de_list = stat_data[:,3]
de_list_ref = stat_data_ref[:,3]

#==================Calculating Cm alpha==================
CN = (CL_list[0] * 0.5 * rho_list[0] * (red_vel[0] ** 2) * 30)/(Weight_list[1])
CM_delta = -1/(de_list[0] - de_list[1]) * CN * ((0.0254 *(X_cg_config1 - X_cg_config2))/2.0569)

CN_ref = (CL_list_ref[0] * 0.5 * rho_list_ref[0] * (red_vel_ref[0] ** 2) * 30)/(Weight_list_ref[1])
CM_delta_ref = -1/(de_list_ref[0] - de_list_ref[1]) * CN_ref * ((0.0254 *(X_cg_config1_ref - X_cg_config2_ref))/2.0569)

#calculating deflection gradient
stat_data1 = get_stat_data("20200311_V1.xlsx", 59, 66)
coeff_deflection = np.polyfit(stat_data1[:,2],stat_data1[:,3],1)
ddelta_da = coeff_deflection[0]

stat_data1_ref = get_stat_data("Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx", 59, 66)
coeff_deflection_ref = np.polyfit(stat_data1_ref[:,2],stat_data1_ref[:,3],1)
ddelta_da_ref = coeff_deflection_ref[0]


print('ddelta_da',ddelta_da)
print('CM_delta', CM_delta)
CM_alpha = - CM_delta * ddelta_da
print('CM_alpha', CM_alpha)
print("==================Reference data==================")
print('ddelta_da ref',ddelta_da_ref)
print('CM_delta ref', CM_delta_ref)
CM_alpha_ref = - CM_delta_ref * ddelta_da_ref
print('CM_alpha ref', CM_alpha_ref)
