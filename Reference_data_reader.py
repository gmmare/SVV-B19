import scipy.io as spio
import matplotlib.pyplot as plt
from Cit_par import *
import numpy as np

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
for i in dictlist:
    print(i,"  ",   reference_data["flightdata"][i]["description"])
print()

time_in_secs_utc = reference_data["flightdata"]["Gps_utcSec"]["data"]
yvalues = reference_data["flightdata"]["Dadc1_alt"]["data"]
#y2values = reference_data["flightdata"]["Dadc1_alt"]["data"]

TOW = 6689 # Weight for reference data

CL_list = []
Alpha_list = []

print(reference_data["flightdata"]["lh_engine_FU"]["units"])

for j in range(len(reference_data["flightdata"]["Gps_utcSec"]["data"])):
    if reference_data["flightdata"]["Gps_utcSec"]["data"][j]>32000 and reference_data["flightdata"]["Gps_utcSec"]["data"][j] < 32500:
        altitude = reference_data["flightdata"]["Dadc1_alt"]["data"][j]
        hp0  = altitude *0.3048
        rho    = rho0 * pow( ((1+(lambd * hp0 / Temp0))), (-((g / (lambd*R)) + 1)))
        Vel =  reference_data["flightdata"]["Dadc1_tas"]["data"][j] *0.51444
        Fuel_out_weight = (reference_data["flightdata"]["lh_engine_FU"]["data"][j] +  reference_data["flightdata"]["rh_engine_FU"]["data"][j])*0.453592

        Aircraft_weight = TOW - Fuel_out_weight
        Aircraft_weight_newton = Aircraft_weight * g
        CL = 2 * Aircraft_weight_newton / (rho * Vel ** 2 * S)
        aoa = reference_data["flightdata"]["vane_AOA"]["data"][j]
        if CL>1:
            print(altitude, rho, Vel, Fuel_out_weight, CL)
        CL_list.append(CL)
        Alpha_list.append(aoa)


plt.plot(time_in_secs_utc,yvalues)
plt.show()

plt.plot(Alpha_list,CL_list)
plt.show()