import matplotlib.pyplot as plt
from Cit_par import *
import numpy as np
import Reference_data_reader as ref_data
reference_data = ref_data.reference_data

dictlist = reference_data["flightdata"].keys()
for i in dictlist:
    print(i,"  ",   reference_data["flightdata"][i]["description"])
print()

time_in_secs_utc = reference_data["flightdata"]["Gps_utcSec"]["data"]
yvalues = reference_data["flightdata"]["vane_AOA"]["data"]
#y2values = reference_data["flightdata"]["Dadc1_alt"]["data"]

TOW = 6500 # THIS VALUE IS GUESSED, WE SHOULD FIND THE REAL VALUE FOR REFERENCE DATA,AND OUR TEST

CL_list = []
Alpha_list = []

for j in range(len(reference_data["flightdata"]["Gps_utcSec"]["data"])):
    if reference_data["flightdata"]["Gps_utcSec"]["data"][j]>32000 and reference_data["flightdata"]["Gps_utcSec"]["data"][j] < 32500:
        altitude = reference_data["flightdata"]["Dadc1_alt"]["data"][j]
        hp0  = altitude
        rho    = rho0 * pow( ((1+(lambd * hp0 / Temp0))), (-((g / (lambd*R)) + 1)))
        Vel =  reference_data["flightdata"]["Dadc1_tas"]["data"][j]
        Fuel_out_weight = reference_data["flightdata"]["lh_engine_FU"]["data"][j] +  reference_data["flightdata"]["rh_engine_FU"]["data"][j]



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

time = ref_data.get_data("Reference_data.mat","time")
print(time)