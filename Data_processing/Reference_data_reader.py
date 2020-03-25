import scipy.io as spio
import matplotlib.pyplot as plt
from Cit_par import *
import pandas as pd
import numpy as np



#reading the reference data
'Reference_data.mat'

#data file name stationary data
"20200310_V2.xlsx"

#Functions for reading and converting the data from .mat to dictionaries.
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:

            dict[strg] = elem
    return dict

#AC data
def get_data(dataset, measurement, detail = "data"):
    '''
    :param dataset: matlab file containing all measured parameters (str)
    :param measurement: which measurement you want to now (str)
    :param detail: optional if you want to now the units
    :return: list with lists containing all the data
    '''
    reference_data = loadmat(dataset)
    data_list = reference_data["flightdata"][measurement][detail]

    return data_list

#stationary data from the excell file
def get_stat_data(filename, start_line, endline):
    '''
    :param filename: excell file with stationary data
    :param start_line: line of the excel file you want to begin with reading
    :param endline: line of the excel file you want to begin with reading
    :return: numpy array with data
    '''
    data = pd.read_excel(filename)
    data_df = pd.DataFrame(data)

    stat_data = []

    for j in range(start_line-2, endline-2):
        row = []
        for i in range(3,10):
            value = float(data_df.iloc[j][i])
            row.append(value)
        stat_data.append(row)

    return np.array(stat_data)


#reference_data = loadmat('Reference_data.mat')
reference_data = loadmat('FTISxprt-20200311_flight1.mat')

'''
The mat file structure has 48 options. Each option is a parameter that is measured. Each option is split up into three options: units, data
and description. When selecting data one should adhere this format:

reference_data["flightdata"]["desired parameter to be read"]["data"]

This returns an array with the data
'''

#getting the kets
#print(reference_data.keys())
dictlist = reference_data["flightdata"].keys()

for i in dictlist:
     print(i,"  ",   reference_data["flightdata"][i]["description"],reference_data["flightdata"][i]["units"] )
print()

time_in_secs_utc = reference_data["flightdata"]["Gps_utcSec"]["data"]
yvalues = reference_data["flightdata"]["vane_AOA"]["data"]
plt.plot(time_in_secs_utc[13000:20000],yvalues[13000:20000])
#plt.show()
#plt.plot(time_in_secs_utc,yvalues)
plt.show()

#plt.savefig("AOA-Time(UTC).pdf")

#y2values = reference_data["flightdata"]["Dadc1_alt"]["data"]


rho0    = 1.2250            # air density at sea level [kg/m^3]
lambd   = -0.0065           # temperature gradient in ISA [K/m]
Temp0   = 288.15            # temperature at sea level in ISA [K]
R       = 287.05            # specific gas constant [m^2/sec^2K]
g       = 9.81              # [m/sec^2] (gravity constant)
gam     = 1.4               # ratio of specific heats for air [-]
P0      = 101325            # pressure at sea level [Pa]
W_s     = 60500             # Standard weight in N
m_fs    = 0.048             # standard fuel flow in kg/sec

def red_velocity(hp, V_c, T_m, rho):
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

    return V_e, M

def red_mass(V_e, W): # for the elevator trim curve

    '''
    :param V_e: equivalent airspeed
    :param W: weight of the aircraft in [N]
    :return: equivalent airspeed to be used in the elevator trim curve
    '''
    V_e2 = V_e * sqrt(W_s/W)

    return V_e2





def cl__values(utcSectime):
    collecteddata = False
    TOW = 6689  # Weight for reference data
    for j in range(len(reference_data["flightdata"]["Gps_utcSec"]["data"])):
        if int(reference_data["flightdata"]["Gps_utcSec"]["data"][j]) == utcSectime and collecteddata == False:
            altitude = reference_data["flightdata"]["Dadc1_alt"]["data"][j]
            hp0 = altitude * 0.3048
            rho = rho0 * pow(((1 + (lambd * hp0 / Temp0))), (-((g / (lambd * R)) + 1)))
            Vel = reference_data["flightdata"]["Dadc1_tas"]["data"][j] * 0.51444
            Fuel_out_weight = (reference_data["flightdata"]["lh_engine_FU"]["data"][j] +
                               reference_data["flightdata"]["rh_engine_FU"]["data"][j]) * 0.453592

            Aircraft_weight = TOW - Fuel_out_weight
            Aircraft_weight_newton = Aircraft_weight * g
            CL = 2 * Aircraft_weight_newton / (rho * Vel ** 2 * S)
            aoa = reference_data["flightdata"]["vane_AOA"]["data"][j]
            collecteddata = True

    print("hp0:",hp0)
    print("Weight:", Aircraft_weight)
    print("Velocity:", Vel)
    #print("Angle of attack:", aoa)
    #print("Pitch angle", reference_data["flightdata"]["Ahrs1_Pitch"]["data"][j])
    return CL, aoa

def input_for_thrust_values(utcSectime):
    collecteddata = False
    for j in range(len(reference_data["flightdata"]["Gps_utcSec"]["data"])):
        if int(reference_data["flightdata"]["Gps_utcSec"]["data"][j]) == utcSectime and collecteddata == False:
            #print(reference_data["flightdata"]["time"]["data"][j])
            altitude  =  reference_data["flightdata"]["Dadc1_alt"]["data"][j] * 0.3048
            machnum   =  reference_data["flightdata"]["Dadc1_mach"]["data"][j]
            tempdiff  = reference_data["flightdata"]["Dadc1_tat"]["data"][j] - (Temp0 + lambd * altitude) + 273.15
            fuelflowL = reference_data["flightdata"]["lh_engine_FMF"]["data"][j] /7936.64
            fuelflowR = reference_data["flightdata"]["rh_engine_FMF"]["data"][j] /7936.64
            collecteddata = True
            #print(altitude, machnum, tempdiff, fuelflowL, fuelflowR)
    return

def cd_values(utcSectime):
    if utcSectime == 31989:
        thrust = 3574.19 + 3682.47
    if utcSectime == 32129:
        thrust = 2920.64+  2999.72
    if utcSectime == 32258:
        thrust = 2359.71 + 2492.66
    if utcSectime == 32396:
        thrust = 1835.3 + 1988.97
    if utcSectime == 32619:
        thrust = 1874.02 + 2060.28
    if utcSectime == 32751:
        thrust = 2190.85 + 2379.92

    collecteddata = False
    for j in range(len(reference_data["flightdata"]["Gps_utcSec"]["data"])):
        if int(reference_data["flightdata"]["Gps_utcSec"]["data"][j]) == utcSectime and collecteddata == False:
            altitude = reference_data["flightdata"]["Dadc1_alt"]["data"][j]
            hp0 = altitude * 0.3048
            rho = rho0 * pow(((1 + (lambd * hp0 / Temp0))), (-((g / (lambd * R)) + 1)))
            vel = reference_data["flightdata"]["Dadc1_tas"]["data"][j] * 0.51444
            collecteddata = True

    cd = thrust / (0.5 * rho * vel *vel * S)
    return cd

def our_cd_values():
    '''This function returns the cd values based on the data in our excel sheet (V2)'''
    thrust_list = []
    altitude_list = [5010,12000,11990,12140,12140,12050,12080] #in ft
    vel_list = [233,249,222,191,162,130,118] #in kts, and these are IAS.
    TAT = [12,5.2,2.8,0.5,-1.5,-3,-3.8]
    left_fuelflow = [760,737,610,478,420,408,406]
    right_fuelflow = [895,787,650,530,450,444,444]
    Mach_list = []
    eq_vel_list = []
    true_vel_list = []
    rho =1.225
    for i in range(len(altitude_list)):
        temp_diff = Temp0 + (lambd * altitude_list[i]*0.3048)
        temp_diff = temp_diff - TAT[i] - 273.15
        eq_vel, mach = red_velocity(altitude_list[i]*0.3048, vel_list[i]*0.5144, TAT[i]+273.15, rho)
        true_vel = vel_list[i]*sqrt(rho0/rho)*0.5144
        mach2 = true_vel/(340.29*sqrt((TAT[i]+273.15)/Temp0))
        Mach_list.append(mach)
        eq_vel_list.append(eq_vel)
        true_vel_list.append(true_vel)
        print(altitude_list[i]*0.3048, mach2, temp_diff, left_fuelflow[i]/7936.64, right_fuelflow[i]/7936.64)

    LR_thrust_list = [[3704.02,4694.53],[4080.9,4397.64],[3251.84,3562.74],[2396.45,2794.23],[2081.75,2314.45],[2127.58,2417.8],[2169.1,2480.06]]
    for i in range(len(LR_thrust_list)):
        thrust_list.append(LR_thrust_list[i][0]+LR_thrust_list[i][1])

    cd_list = []
    for i in range(len(thrust_list)):
        rho = rho0 * pow(((1 + (lambd * altitude_list[i]*0.3048 / Temp0))), (-((g / (lambd * R)) + 1)))
        cd = thrust_list[i] / (0.5* rho * true_vel_list[i]* true_vel_list[i] * S)
        cd_list.append(cd)

    return cd_list

#cd = our_cd_values()
#print(cd)

def our_thrust_values_elevator():
    '''This function returns the cd values based on the data in our excel sheet (V2)'''
    thrust_list = []
    altitude_list = [11970,12300,12519,12890,12090,11770,10960] #in ft
    vel_list = [160,149,140,130,171,181,190] #in kts, and these are IAS.
    TAT = [-1.5,-2.5,-3.8,-4.2,-0.8,0.5,2.2]
    left_fuelflow = [424,420,416,411,426,432,444]
    right_fuelflow = [473,468,463,458,474,480,492]
    Mach_list = []
    eq_vel_list = []
    true_vel_list = []
    rho =1.225
    for i in range(len(altitude_list)):
        temp_diff = Temp0 + (lambd * altitude_list[i]*0.3048)
        temp_diff = temp_diff - TAT[i] - 273.15
        eq_vel, mach = red_velocity(altitude_list[i]*0.3048, vel_list[i]*0.5144, TAT[i]+273.15, rho)
        true_vel = vel_list[i]*sqrt(rho0/rho)*0.5144
        mach2 = true_vel/(340.29*sqrt((TAT[i]+273.15)/Temp0))
        Mach_list.append(mach)
        eq_vel_list.append(eq_vel)
        true_vel_list.append(true_vel)
        print(altitude_list[i]*0.3048, mach2, temp_diff, left_fuelflow[i]/7936.64, right_fuelflow[i]/7936.64)

    LR_thrust_list = [[2110.56,2489.12],[2148.42,2527.37],[2167.1,2541.55],[2202.69,2579.48],[2086.78,2453.21],[2071.86,2434.85],[2074.65,2431.5]]
    for i in range(len(LR_thrust_list)):
        thrust_list.append(LR_thrust_list[i][0]+LR_thrust_list[i][1])
    print(thrust_list)
    cd_list = []
    for i in range(len(thrust_list)):
        rho = rho0 * pow(((1 + (lambd * altitude_list[i]*0.3048 / Temp0))), (-((g / (lambd * R)) + 1)))
        cd = thrust_list[i] / (0.5* rho * eq_vel_list[i]* eq_vel_list[i] * S)
        cd_list.append(cd)

    return cd_list

def ref_cd_values():
    '''This function returns the cd values based on the data in our excel sheet (V2)'''
    thrust_list = []
    altitude_list = [5010,5020,5020,5030,5020,5110] #in ft
    vel_list = [249,221,192,163,130,118] #in kts, and these are IAS.
    TAT = [12.5,10.5,8.8,7.2,6,5.2]
    left_fuelflow = [798,673,561,463,443,474]
    right_fuelflow = [813,682,579,484,467,499]
    Mach_list = []
    eq_vel_list = []
    true_vel_list = []
    rho =1.225
    for i in range(len(altitude_list)):
        temp_diff = Temp0 + (lambd * altitude_list[i]*0.3048)
        temp_diff = temp_diff - TAT[i] - 273.15
        eq_vel, mach = red_velocity(altitude_list[i]*0.3048, vel_list[i]*0.5144, TAT[i]+273.15, rho)
        true_vel = vel_list[i]*sqrt(rho0/rho)*0.5144
        mach2 = true_vel/(340.29*sqrt((TAT[i]+273.15)/Temp0))
        Mach_list.append(mach)
        eq_vel_list.append(eq_vel)
        true_vel_list.append(true_vel)
        print(altitude_list[i]*0.3048, mach2, temp_diff, left_fuelflow[i]/7936.64, right_fuelflow[i]/7936.64)

    LR_thrust_list = [[3874.09,3986.35],[3142.03,3205.91],[2502.87,2631.16],[1931.05,2086.87],[1941.73,2128.02],[2253.37,2452.13]]
    for i in range(len(LR_thrust_list)):
        thrust_list.append(LR_thrust_list[i][0]+LR_thrust_list[i][1])

    cd_list = []
    for i in range(len(thrust_list)):
        rho = rho0 * pow(((1 + (lambd * altitude_list[i]*0.3048 / Temp0))), (-((g / (lambd * R)) + 1)))
        cd = thrust_list[i] / (0.5* rho * true_vel_list[i]* true_vel_list[i] * S)
        cd_list.append(cd)


    return cd_list


cd = our_thrust_values_elevator()
print(cd)

#cd = our_thrust_values_elevator()
#print(cd)
def ref_time_to_utc(min,sec):
    time = min*60+sec +30832
    return time

def our_time_to_utc(min,sec):
    time = min*60+sec + 33490 #Roughly correct conversion
    return time

def stick_force_curve():
    Fe = [0,-23,-29,-46,26,40,83]
    IAS = [161,150,140,130,173,179,192]
    altitude_list = [6060,6350,6550,6880,6160,5810,5310]
    TAT = [5.5,4.5,3.5,2.5,5.0,6.2,8.2]
    Fuel_used = [230,494,519,565,593,627,657]
    aircraft_weight = []
    eq_vel_lst = []
    Feas = []
    for i in range(len(Fe)):

        weight = (14266-Fuel_used[i])*0.4536*9.81
        eq_vel, mach = red_velocity(altitude_list[i] * 0.3048, IAS[i] * 0.5144, TAT[i] + 273.15, rho)
        eq_vel_lst.append(eq_vel)
        aircraft_weight.append(weight)
        Feas.append(Fe[i]*60500/weight)

    plt.scatter(eq_vel_lst,Feas)
    plt.xlabel('$V_e  [m/s]$', fontsize ='large')
    plt.ylabel('$F_e  [N]$', fontsize ='large')
    plt.gca().invert_yaxis()
    plt.title('Elevator control force curve')
    #plt.show()
    plt.savefig('control_force_curve.pdf')

#stick_force_curve()

def reduced_elevator_curve():
    Fe = [0, -23, -29, -46, 26, 40, 83]
    IAS = [161, 150, 140, 130, 173, 179, 192]
    altitude_list = [6060, 6350, 6550, 6880, 6160, 5810, 5310]
    TAT = [5.5, 4.5, 3.5, 2.5, 5.0, 6.2, 8.2]
    Fuel_used = [230, 494, 519, 565, 593, 627, 657]
    aircraft_weight = []
    eq_vel_lst = []
    Cma = -0.18697327092815078
    Cm0 = 0.0297
    Cmtc = -0.0064
    Cmdelta = -0.43413107112501953
    Cna = Cma
    radius = float(0.686/2 )#source: wikipedia
    thrust = [4599.68, 4675.79, 4708.65, 4782.17, 4539.99, 4506.71, 4506.15]

    d_e_eq_list = []
    for i in range(len(Fe)):
        rho = rho0 * pow(((1 + (lambd * altitude_list[i] * 0.3048 / Temp0))), (-((g / (lambd * R)) + 1)))

        weight = (14266 - Fuel_used[i]) * 0.4536 * 9.81
        eq_vel, mach = red_velocity(altitude_list[i] * 0.3048, IAS[i] * 0.5144, TAT[i] + 273.15, rho)
        eq_vel_lst.append(eq_vel)
        Tcs = thrust[i]/(0.5*rho*eq_vel*eq_vel*np.pi*radius*radius)
        aircraft_weight.append(weight)
        d_e_eq = -1/Cmdelta*(Cm0 + Cma/Cna * weight/ (0.5*rho*eq_vel*eq_vel*S)+Cmtc*Tcs)
        d_e_eq_list.append(d_e_eq)

    plt.scatter(eq_vel_lst,d_e_eq_list)
    plt.xlabel('$V_e   [m/s]$', fontsize = 'large')
    plt.ylabel('$\delta_e  [-]$', fontsize='large')
    plt.title('Elevator trim curve')
    plt.savefig('elevator_trim_curve.pdf')
    #plt.show()

reduced_elevator_curve()

'''
reduced_elevator_curve()

list_for_seb = [[59,10],[61,57],[62,47],[65,20]]
list_for_seb = [[57,47],[58,44],[62,59]]
#list_for_seb = [[53,57],[60,35]]
ref_time_list_seb = []
for i in list_for_seb:
    minutes = i[0]
    seconds = i[1]
    ref_time = our_time_to_utc(minutes,seconds)
    ref_time_list_seb.append(ref_time)

print(ref_time_list_seb)
for i in range(len(ref_time_list_seb)):
    print("Timestamp:", list_for_seb[i])
    a,b = cl__values(ref_time_list_seb[i])'''
