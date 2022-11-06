#!/usr/bin/env python3
import numpy as np

def generate_data(p_dm = 50,plot=False,time=5,toa_peak=2.5,width=5e-3):
    DM_CONST = 4.149377593360996  # dispersion constant

    def time_to_bin(t, sample_rate=327e-6):
        """Return time as a bin number provided a sampling time"""
        return np.round(t / sample_rate).astype(int)

    def dm_delay(dm, f1, f2):
        """Return DM delay in seconds"""
        return DM_CONST * dm * (1.0 / f2 ** 2 - 1.0 / f1 ** 2)*1e-3


    #this is your DM
    freqs = np.linspace(0.8,0.4,1024)
    max_dm_delay = dm_delay(p_dm, max(freqs), min(freqs))
    print(f"max dm delay {max_dm_delay}")
    #time sample rate is 327us
    tsamp = 5e-3
    max_dm_delay_bins = time_to_bin(max_dm_delay, tsamp)
    nbins_to_sim = max_dm_delay_bins

    toa_bin_top = int(toa_peak/tsamp)
    n_tsamples = int(time/tsamp)
    if n_tsamples<nbins_to_sim:
        n_tsamples = nbins_to_sim
    data = np.zeros((1024,n_tsamples))
    #width is the width of the gaussian pulse in ms
    width_bins = time_to_bin(width, tsamp)
    #time sample of first bin

    dm_delays = dm_delay(p_dm, freqs[0], freqs)
    per_chan_toa_bins = time_to_bin(dm_delays, tsamp)
    x = np.linspace(0, nbins_to_sim, nbins_to_sim)
    pulse_wf = np.exp(
        -((x - per_chan_toa_bins[:, np.newaxis]) ** 2) / (2 * width_bins ** 2)
    )
    per_chan_inject_pow = 10
    pulse_wf /= pulse_wf.max(axis=1)[:, np.newaxis]
    pulse_wf *= per_chan_inject_pow

    end_bin = toa_bin_top+nbins_to_sim
    data[:,toa_bin_top:end_bin] +=pulse_wf

    metadata = {
        'tsamp':tsamp, #in seconds
        'freqs':freqs, #in freqs
        'times':np.arange(0,data.shape[1])*tsamp #times in seconds
    }
    if(plot):
        import matplotlib.pyplot as plt
        plt.imshow(data,aspect="auto",extent=(0,tsamp*data.shape[1],np.min(freqs),max(freqs)))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.colorbar()
        plt.show()
    return data, metadata 


def generate_rfi(data,bw=70,instances=10,rfi_length = 10,plot=True):
    #BW in channels, instances is the number of isntances of RFI
    t_shape = data.shape[1]
    freq_shape = data.shape[0]
    time_rng_g = np.random.default_rng(seed=1337)
    width_rng_g = np.random.default_rng(seed=1338)
    freq_rng_g = np.random.default_rng(seed=1339)
    for i in range(instances):
        time_rng = int(time_rng_g.random()*t_shape)
        width_rng = int(width_rng_g.random()*50)
        freq_rng = int(freq_rng_g.random()*freq_shape)
        data[freq_rng:freq_rng+width_rng,time_rng:time_rng+rfi_length]+=100
    if plot:
        import matplotlib.pyplot as plt
        plt.imshow(data,aspect="auto",extent=(0,10,0.4,0.8))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.colorbar()
        plt.show()
    return data

if(__name__ == "__main__"):
    data,metadata = generate_data(p_dm=1000,plot=True)
    data_with_rfi = generate_rfi(data)
    #generate ony rfi
    rfi_data = generate_rfi(np.zeros((1024,80000)))
