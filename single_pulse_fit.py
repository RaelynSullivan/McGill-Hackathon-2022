#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import Dispersion_sweep
from scipy import signal

def activation_metric(prod,sig_val):
    activation = np.sum(prod,axis=0)
    metric = np.sum(activation > sig_val)/1024
    return metric
def convolve(data,kernel):
    time_kernel = kernel.shape[1]
    time_data = data.shape[1]
    iters = time_data//time_kernel
    correlate = []
    for i in range(time_data-time_kernel):
        prod = kernel*data[:,i:i+time_kernel]
        am = activation_metric(prod,20)
        prod = prod*am
        correlate.append(np.trapz(np.trapz(prod)))
    return correlate
    #input is the data and kernel, output will be

DM_trials = np.linspace(10,30,20)
max_correlate = []
data,metadata = Dispersion_sweep.generate_data(p_dm=20,time=10,toa_peak=5)
data = Dispersion_sweep.generate_rfi(data)
DM_trials=[20]
for dm in DM_trials:
    kernel,meta_kern = Dispersion_sweep.generate_data(p_dm=dm,time=0,toa_peak=0)
    correlate = convolve(data,kernel)
    max_correlate.append(max(correlate))
    plt.plot(correlate)
    plt.show()

plt.plot(DM_trials,max_correlate)
plt.show()
