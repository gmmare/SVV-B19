import scipy.io as spio
import numpy as np

#name of the data file
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

#Reading the AC data
reference_data = loadmat('Reference_data.mat')

#possible parameters
#print(reference_data["flightdata"].keys())
'''
The mat file structure has 48 options. Each option is a parameter that is measured. Each option is split up into three options: units, data
and description. When selecting data one should adhere this format:

reference_data["flightdata"]["desired parameter to be read"]["data"]

This returns an array with the data
'''
def get_data(dataset, measurement, detail = "data"):
    reference_data = loadmat(dataset)
    data_list = reference_data["flightdata"][measurement][detail]

<<<<<<< HEAD
    return data_list

#getting the data
test_list = get_data("Reference_data.mat", "lh_engine_itt", detail = "data")
print(test_list[0:10])
=======
#getting the kets
print(reference_data.keys())
dictlist = reference_data["flightdata"].keys()
for i in dictlist:
    print(i,"  ",   reference_data["flightdata"][i]["description"])
print()
test_list = reference_data["flightdata"]["lh_engine_FMF"]["description"]
print(test_list)
>>>>>>> 311729f209a20b13c1def9cd2f1799cb2e00e918
