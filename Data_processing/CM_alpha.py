from Reference_data_reader import get_stat_data
import numpy as np
from Reduction_functions import red_velocity
from math import *
import matplotlib.pyplot as plt

#importing data
stat_data = get_stat_data("20200311_V1.xlsx", 75, 77)
stat_data_ref = get_stat_data("Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx", 75, 77)
W_total = 6471 #kg
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
rho0 = 1.225 #p0/(R*T0) #[kg/m3]
X_cg_config1 = 0.2493
X_cg_config2 = 0.2247
# X_cg_config1 = 285.28
# X_cg_config2 = 281.64

b = 15.911
S = 30
A = (b**2) / S

X_cg_config1_ref = 210.04 #inch
X_cg_config2_ref = 187.74
#==================calculating density==================
#Constants and standard sea level values
g0 = 9.80665 #[m/s2]
R = 287 #[J/kgK]
T0 = 288.15 #[K]
p0 = 101325 #[Pa]
rho0 = 1.225 #p0/(R*T0) #[kg/m3]

#==================converting velocity==================
IAS = stat_data[:,1] # this is calibrated veloctiy in kts
TAT = stat_data[:,-1] #true air temp in degree
TAS = []
red_vel = []

IAS_ref = stat_data_ref[:,1] # this is calibrated veloctiy in kts
TAT_ref = stat_data_ref[:,-1] #true air temp in degree
TAS_ref = []
red_vel_ref = []

#temperature change per altitude
a1 = -0.0065    #0<11km

#getting the altitude
alt = stat_data[:,0]
alt_ref = stat_data_ref[:,0]

rho_list = []
rho_list_ref = []
for i in range(len(alt)):
    h = 0.3048 * alt[i] #Conversion from ft to m
    T = T0 + a1*(h)
    p = p0*((T/T0)**(-g0/(a1*R)))
    rho = p/(R*T)
    rho_list.append(rho)#red_velocity(alt[i] * 0.3048, IAS[i]*0.514444, TAT[i] + 273)[2])

for i in range(len(alt_ref)):
    h = 0.3048 * alt_ref[i] #Conversion from ft to m

    T = T0 + a1*(h)
    p = p0*((T/T0)**(-g0/(a1*R)))
    rho_ref = p/(R*T)
    rho_list_ref.append(rho_ref)#red_velocity(alt_ref[i] * 0.3048, IAS_ref[i] * 0.514444, TAT_ref[i] + 273)[2])


for i in range(len(IAS)):
    red_vel.append(red_velocity(alt[i] * 0.3048, IAS[i]*0.514444, TAT[i] + 273)[0])
    red_vel_ref.append(red_velocity(alt_ref[i] * 0.3048, IAS_ref[i] * 0.514444, TAT_ref[i] + 273)[0])

    TAS.append(IAS[i] * 0.514444 * sqrt(rho0/rho_list[i]))
    print(TAS[-1])
    TAS_ref.append(IAS_ref[i]*0.514444 * sqrt(rho0/rho_list_ref[i]))


#==================Weight==================
Weight_list = []
Weight_list_ref = []
F_used = stat_data[:,8]
F_used_ref = stat_data_ref[:,8]
for i in range(len(F_used)): #weight in N
    Weight = (W_total - F_used[i] * 0.453592) * 9.81
    Weight_list.append(Weight)
    Weight_ref = (W_total_ref - F_used_ref[i] * 0.453592) * 9.81
    Weight_list_ref.append(Weight_ref)

de_list = stat_data[:,3] #in degree
de_list_ref = stat_data_ref[:,3]

#==================Calculating Cm alpha==================
#calculating CN
CN = (Weight_list[1])/(0.5 * rho_list[1] * S * (TAS[1] ** 2)) * cos(radians(alpha[1]))
CN_ref = (Weight_list_ref[1])/(0.5 * rho_list_ref[1] * S * (TAS_ref[1] ** 2))*cos(radians(alpha_ref[1]))

#calculating cm delta
CM_delta = (-1/((de_list[0] - de_list[1])*(pi/180))) * CN *(X_cg_config1 - X_cg_config2)
CM_delta_ref = -(1/((de_list_ref[0] - de_list_ref[1])*(pi/180))) * CN_ref * (X_cg_config1 - X_cg_config2)

#elevator trim curve
stat_data1 = get_stat_data("20200311_V1.xlsx", 59, 66)
stat_data1_ref = get_stat_data("Post_Flight_Datasheet_Flight_1_DD_12_3_2018.xlsx", 59, 66)

alpha_tail = []
alpha_tail_ref = []
#calculating deflection gradient
for i in range(len(stat_data1[:,2])):
    alpha_tail.append(stat_data1[:,2][i] - 4 / (A + 2)) #adjusting for downwash
    alpha_tail_ref.append(stat_data1_ref[:,2][i] - 4 / (A + 2))

plt.plot(alpha_tail, stat_data1[:,3], 'ob')
plt.show()
coeff_deflection = np.polyfit(alpha_tail,stat_data1[:,3],1)
ddelta_da = coeff_deflection[0]

coeff_deflection_ref = np.polyfit(alpha_tail_ref,stat_data1_ref[:,3],1)
ddelta_da_ref = coeff_deflection_ref[0]

#results
print("CN", CN)
print("CN_ref", CN_ref)
print("==================own data==================")
print('ddelta_da',ddelta_da)
print('CM_delta', CM_delta)
CM_alpha = - CM_delta * ddelta_da
print('CM_alpha', CM_alpha)
print("==================Reference data==================")
print('ddelta_da ref',ddelta_da_ref)
print('CM_delta ref', CM_delta_ref)
CM_alpha_ref = - CM_delta_ref * ddelta_da_ref
print('CM_alpha ref', CM_alpha_ref)
