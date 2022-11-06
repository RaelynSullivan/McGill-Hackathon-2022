#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import Dispersion_sweep as ds



data,metadata=ds.generate_data(p_dm=60,time=10,toa_peak=5)
data60=ds.generate_rfi(data)

#data60,metadata60=ds.generate_data(p_dm=60,time=10,toa_peak=5,plot=True)

# print(data40)
# plt.imshow(data40,aspect="auto")
# plt.colorbar()
# plt.show()
# plt.imshow(data60,aspect="auto")
# plt.colorbar()
# plt.show()

# dat=dat

# #create kernel? in map space maybe? plot to make sure it looks right

kernel,meta_kern = ds.generate_data(p_dm=60,time=0,toa_peak=0,plot=False)

# kern=kern

# ##go to fourier space with both the kernel and the map




# plt.imshow(np.abs(ftimage40),aspect="auto")
# plt.colorbar()
# plt.show()

ftdat60 = np.fft.fft2(data60.T)
fkern = np.fft.fft2(kernel.T, s=ftdat60.shape)
ftimage60 = np.fft.fftshift(ftdat60.T)
plt.imshow(np.abs(ftimage60),aspect="auto")
plt.colorbar()
plt.show()
# # ftkern=np.fft.fft2(kern)

# ftdat=fft.fft2(dat)
# ftkern=fft.fft2(kern)


# #convolve/multiply kernel(s) and data

ftdat_conv=ftdat60*fkern
ftimage = np.fft.fftshift(ftdat_conv)
plt.imshow(np.abs(ftimage),aspect="auto")
plt.colorbar()
plt.show()
# ##go back to map space to apply the weights
# dat_conv40=fft.ifft2(fkern).T
# # plt.imshow(np.abs(dat_conv40),aspect="auto")
# # plt.colorbar()
# # plt.show()

# dat_conv60=fft.ifft2(ftdat60).T

# # plt.imshow(np.abs(dat_conv60),aspect="auto")
# # plt.colorbar()
# # plt.show()

dat_conv4060=fft.ifft2(ftdat_conv).T
plt.imshow(np.abs(dat_conv4060),aspect="auto")
print(np.where(dat_conv4060>=1))
plt.colorbar()
plt.show()