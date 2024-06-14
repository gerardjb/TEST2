import numpy as np
import os
import scipy.io as sio

def open_Janelia_1(j_path):
    all_data = sio.loadmat(j_path)
    data = all_data['dff']
    time = all_data['time_stamps']

    return time, data

filename = os.path.join("gt_data","jGCaMP8f_ANM471994_cell01.mat")
time,data = open_Janelia_1(filename)

time1 = np.float64(time[0,:])
print(time1.shape)
print(time1[0])
print(time1[1])