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


reference_data = loadmat('Reference_data.mat')
'''
The mat file structure has 48 options. Each option is a parameter that is measured. Each option is split up into three options: units, data
and description. When selecting data one should adhere this format:

reference_data["flightdata"]["desired parameter to be read"]["data"]

This returns an array with the data
'''

#getting the kets
#print(reference_data.keys())
dictlist = reference_data["flightdata"].keys()
# for i in dictlist:
#     print(i,"  ",   reference_data["flightdata"][i]["description"],reference_data["flightdata"][i]["units"] )
#print()

time_in_secs_utc = reference_data["flightdata"]["Gps_utcSec"]["data"]
yvalues = reference_data["flightdata"]["Dadc1_alt"]["data"]
#y2values = reference_data["flightdata"]["Dadc1_alt"]["data"]

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

def ref_time_to_utc(min,sec):
    time = min*60+sec +30832
    return time

ref_time_utc_list = [31989,32129,32258,32396,32619,32751]

cl_list = []
aoa_list = []
cd_list = []

for i in ref_time_utc_list:
    cl,aoa = cl__values(i)
    cd = cd_values(i)
    cl_list.append(cl)
    aoa_list.append(aoa)
    cd_list.append(cd)

#print(cl_list,aoa_list,cd_list)
#
# plt.plot(cd_list,cl_list)
# plt.show()
# plt.plot(aoa_list,cl_list)
# plt.show()

#plt.plot(time_in_secs_utc,yvalues)
#plt.show()


