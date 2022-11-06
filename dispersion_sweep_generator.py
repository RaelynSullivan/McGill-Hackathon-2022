#!/usr/bin/env python3
import numpy as np

def generate_data(p_dm = 50,plot=False):
    DM_CONST = 4.149377593360996  # dispersion constant

    def time_to_bin(t, sample_rate=327e-6):
        """Return time as a bin number provided a sampling time"""
        return np.round(t / sample_rate).astype(int)

    def dm_delay(dm, f1, f2):
        """Return DM delay in seconds"""
        return DM_CONST * dm * (1.0 / f2 ** 2 - 1.0 / f1 ** 2)*1e-3

    n_tsamples = 20000
    data = np.zeros((1024,n_tsamples))
    #this is your DM
    freqs = np.linspace(0.8,0.4,1024)
    max_dm_delay = dm_delay(p_dm, max(freqs), min(freqs))
    print(f"max dm delay {max_dm_delay}")
    #time sample rate is 327us
    tsamp = 327e-6
    max_dm_delay_bins = time_to_bin(max_dm_delay, tsamp)
    #width is the width of the gaussian pulse in ms
    width = 5e-3
    width_bins = time_to_bin(width, tsamp)
    #time sample of first bin
    toa_bin_top = 1000
    dm_delays = dm_delay(p_dm, freqs[0], freqs)
    per_chan_toa_bins = toa_bin_top + time_to_bin(dm_delays, tsamp)
    nbins_to_sim = 2 * max_dm_delay_bins
    x = np.linspace(0, nbins_to_sim, nbins_to_sim)
    pulse_wf = np.exp(
        -((x - per_chan_toa_bins[:, np.newaxis]) ** 2) / (2 * width_bins ** 2)
    )
    per_chan_inject_pow = 0.5
    pulse_wf /= pulse_wf.max(axis=1)[:, np.newaxis]
    pulse_wf *= per_chan_inject_pow

    end_bin = toa_bin_top+nbins_to_sim
    data[:,toa_bin_top:end_bin] +=pulse_wf

    metadata = {
        'tsamp':tsamp,
        'freqs':freqs,
        'times':np.arange(0,data.shape[1])*tsamp
    }
    if(plot):
        import matplotlib.pyplot as plt
        plt.imshow(data,aspect="auto")
        plt.show()
    return data, metadata 


if(__name__ == "__main__"):
    generate_data(plot=True)