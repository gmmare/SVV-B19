import scipy.io as spio

import numpy as np
import matplotlib.pyplot as plt


#reading the reference data
'Reference_data.mat'

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
def get_rf(filename):
    reference_data = loadmat('Reference_data.mat')
    return reference_data
'''
The mat file structure has 48 options. Each option is a parameter that is measured. Each option is split up into three options: units, data
and description. When selecting data one should adhere this format:

reference_data["flightdata"]["desired parameter to be read"]["data"]

This returns an array with the data
'''

def get_value(hour,minu,sec):
    
    index=(hour*3600+minu*60+sec)*10-90

    return index

def get_kv(lst1,lst2,lst3,lst4,lst5,lst6, index):
    return lst1[index], lst2[index], lst3[index], lst4[index], lst5[index],lst6[index]

def get_kv2(lst1,lst2,lst3,lst4,lst5, index):
    return lst1[index], lst2[index], lst3[index], lst4[index], lst5[index]

def get_iv(lst1,lst2,lst3,index,index_end,t):
    inputs_da=[]
    inputs_dr=[]
    inputs_de=[]
    time=[]
    for i in range(index,index_end+1):
        inputs_da.append(lst1[i])
        inputs_de.append(lst2[i])
        inputs_dr.append(lst3[i])
        time.append(t[i])
        
        
    print(len(time)) 
    return inputs_da,inputs_de,inputs_dr,time

def get_graph_values(lst1_a,lst2_a,lst3_a,lst4_a,lst5_a,lst6_a,lst7_a, lst8_a,start_hour,start_min,start_sec, end_hour,end_min,end_sec):
    index_start=get_value(start_hour,start_min,start_sec)
    index_end=get_value(end_hour,end_min,end_sec)
    lst1,lst2,lst3,lst4,lst5,lst6,lst7,lst8=[],[],[],[],[],[],[],[]
    for i in range(index_start,index_end+1):
        lst1.append(lst1_a[i]*0.5144444444444444)
        lst2.append(lst2_a[i]*np.pi/180)
        lst3.append(lst3_a[i]*np.pi/180)
        lst4.append(lst4_a[i]*np.pi/180)
        lst5.append(lst5_a[i]*np.pi/180)
        lst6.append(lst6_a[i]*np.pi/180)
        lst7.append(lst7_a[i]*np.pi/180)
        lst8.append(lst8_a[i]*np.pi/180)
    return lst1,lst2,lst3,lst4,lst5,lst6,lst7,lst8

#symmetric
def get_lists(tas,alt,pitch,AOA,PR,d_a,d_r,d_e,t):
    reference_data=get_rf('Reference_data.mat')
    test_list_tas = reference_data["flightdata"][str(tas)]["data"]
    test_list_alt=reference_data["flightdata"][str(alt)]["data"]
    theta_list=reference_data["flightdata"][str(pitch)]["data"]
    angle_of_attack_list=reference_data["flightdata"][str(AOA)]["data"]
    test_list_pitchrate=reference_data["flightdata"][str(PR)]["data"]

    delta_a=reference_data["flightdata"][str(d_a)]["data"]
    delta_r=reference_data["flightdata"][str(d_r)]["data"]
    delta_e=reference_data["flightdata"][str(d_e)]["data"]

    t=reference_data["flightdata"][str(t)]["data"]

    return test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate,  delta_a, delta_r, delta_e, t

#asymmetric
def get_lists_asymmetric(side_slip1,side_slip2,roll_angle, roll_rate, yaw_rate):
    reference_data=get_rf('Reference_data.mat')
    side_slip_list1 = reference_data["flightdata"][str(side_slip1)]["data"]
    side_slip_list2 = reference_data["flightdata"][str(side_slip2)]["data"]
    roll_angle_list=reference_data["flightdata"][str(roll_angle)]["data"]
    roll_rate_list=reference_data["flightdata"][str(roll_rate)]["data"]
    yaw_rate_list=reference_data["flightdata"][str(yaw_rate)]["data"]
    
    side_slip_list=[]
    for i in range(len(side_slip_list1)):
        number=np.arctan(side_slip_list1[i]/side_slip_list2[i])
        side_slip_list.append(number)

    return side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list


#symmetric
def get_Phugoid(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,pitchrate_list, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec, delta_a, delta_e, delta_r): #Phugoid motion
    index=get_value(start_hour,start_min,start_sec) #0,53,57
    index_end=get_value(end_hour,end_min,end_sec) #0,58,0
    Phugoid_tas, Phugoid_alt, Phugoid_theta, Phugoid_aoa,Phugoid_PR, Phugoid_time =get_kv(test_list_tas, test_list_alt,theta_list, angle_of_attack_list,pitchrate_list,t,index)
    Phugoid_inputs_de, Phugoid_time=get_iv(delta_a, delta_e, delta_r, index, index_end,t)[1], get_iv(delta_a, delta_e, delta_r, index, index_end,t)[-1]
    return Phugoid_tas, Phugoid_alt, Phugoid_theta, Phugoid_aoa,Phugoid_PR, Phugoid_inputs_de, Phugoid_time


def get_DR(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec, delta_a, delta_e, delta_r): #Dutch Roll motion
    index=get_value(start_hour,start_min,start_sec) #1,1,57
    index_end=get_value(end_hour,end_min,end_sec) #1,2,18
    print(index,index_end)
    DR_tas, DR_alt, DR_theta, DR_aoa, DR_PR, DR_time =get_kv(test_list_tas, test_list_alt,theta_list, angle_of_attack_list,test_list_pitchrate, t, index)
    lst=get_iv(delta_a, delta_e, delta_r, index, index_end,t)
    DR_inputs_da,DR_inputs_dr, DR_time=lst[0], lst[2], lst[-1]
    print(len(DR_time))
    return DR_tas, DR_alt, DR_theta, DR_aoa, DR_PR, DR_inputs_da, DR_inputs_dr, DR_time

def get_SP(test_list_tas, test_list_alt, theta_list, angle_of_attack_list,test_list_pitchrate, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec, delta_a, delta_e, delta_r): #Short Period motion
    index=get_value(start_hour,start_min,start_sec) #1,0,35

    index_end=get_value(end_hour,end_min,end_sec) #1,1,28

    SP_tas, SP_alt, SP_theta, SP_aoa,SP_PR, SP_time =get_kv(test_list_tas, test_list_alt,theta_list, angle_of_attack_list,test_list_pitchrate,t,index)
    SP_inputs_de,SP_time=get_iv(delta_a, delta_e, delta_r, index, index_end,t)[1], get_iv(delta_a, delta_e, delta_r, index, index_end,t)[-1]
    return SP_tas, SP_alt, SP_theta, SP_aoa, SP_PR, SP_inputs_de, SP_time

#asymetric #only for DR

def get_DRasym(side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, t, start_hour, start_min, start_sec, end_hour, end_min, end_sec): #Dutch Roll motion
    index=get_value(start_hour,start_min,start_sec) #1,1,57
    DR_sideslip, DR_roll_angle, DR_roll_rate, DR_yaw_rate, DR_time =get_kv2(side_slip_list,roll_angle_list,roll_rate_list,yaw_rate_list, t, index)
    return DR_sideslip, DR_roll_angle, DR_roll_rate, DR_yaw_rate

def get_mass(hours,minu,sec,t,alt,tas):
    reference_data=get_rf('Reference_data.mat')
    lh=reference_data["flightdata"]["lh_engine_FU"]["data"]
    rh=reference_data["flightdata"]["rh_engine_FU"]["data"]
    
    TOW = 6689
    rho0   = 1.2250          # air density at sea level [kg/m^3] 
    lambd = -0.0065         # temperature gradient in ISA [K/m]
    Temp0  = 288.15          # temperature at sea level in ISA [K]
    R      = 287.05          # specific gas constant [m^2/sec^2K]
    g      = 9.81
    for j in range(len(t)):
        if j==get_value(hours,minu,sec):
            altitude = alt[j]
            hp0  = altitude *0.3048
            rho    = rho0 * pow( ((1+(lambd * hp0 / Temp0))), (-((g / (lambd*R)) + 1)))
            Vel =  tas[j] *0.51444
            Fuel_out_weight = (lh[j] +  rh[j])*0.453592
            Aircraft_weight = TOW - Fuel_out_weight
    return Aircraft_weight 















