import numpy as np

import Dispersion_sweep

data, metadata = Dispersion_sweep.generate_data()

if True:
    Dispersion_sweep.generate_rfi(data, plot=False)

import matplotlib.pyplot as plt


if True:
    plt.imshow(
        data,
        extent=(
            metadata["times"][0],
            metadata["times"][-1],
            metadata["freqs"][-1],
            metadata["freqs"][0],
        ),
        aspect="auto",
    )
    plt.show()


times, freqs = metadata["times"], metadata["freqs"]
times_mesh, freq_mesh = np.meshgrid(times, freqs)

# I want to try and downsample the data, just so that the next few functions run faster? We can iterate on that afterwards.
print("start coarsening")
data_coarsened = np.zeros(shape=(100, 1000))
times_coarsened_edges = np.linspace(
    times[0], times[-1], data_coarsened.shape[1] + 1, endpoint=True
)
freqs_coarsened_edges = np.linspace(
    freqs[-1], freqs[0], data_coarsened.shape[0] + 1, endpoint=True
)
weighted = np.histogram2d(
    freq_mesh.flatten(),
    times_mesh.flatten(),
    bins=(freqs_coarsened_edges, times_coarsened_edges),
    weights=data.flatten(),
)
counts = np.histogram2d(
    freq_mesh.flatten(),
    times_mesh.flatten(),
    bins=(freqs_coarsened_edges, times_coarsened_edges),
)
data_coarsened[:, :] = weighted[0] / counts[0]
times_coarsened = 0.5 * (times_coarsened_edges[1:] + times_coarsened_edges[:-1])
freqs_coarsened = 0.5 * (freqs_coarsened_edges[1:] + freqs_coarsened_edges[:-1])
if True:
    plt.imshow(
        data_coarsened,
        extent=(
            times_coarsened[0],
            times_coarsened[-1],
            freqs_coarsened[-1],
            freqs_coarsened[0],
        ),
        aspect="auto",
    )
    plt.show()
print("fin coarsening")


# we want to pattern match against the dispersion sweep function
def t_delay_freq(const, f):
    """Return delay, with a generic constant out front, in units of time"""
    return const * (1.0 / f**2)


def surrogate_function(t, f, t_offset, amplitude, pulse_width, delay_const):
    offwid_factor = 4.
    return amplitude * (
        np.exp(
            -((t - t_offset - t_delay_freq(delay_const, f)) ** 2)
            / (2 * pulse_width**2)
        )
        / np.sqrt(2 * np.pi * pulse_width**2)
        - np.exp(
            -((t - t_offset - t_delay_freq(delay_const, f)) ** 2)
            / (2 * (offwid_factor * pulse_width) ** 2)
        )
        / np.sqrt(2 * np.pi * (offwid_factor * pulse_width) ** 2)
    )


times_mesh_coarse, freq_mesh_coarse = np.meshgrid(times_coarsened, freqs_coarsened)
surrogate_data = surrogate_function(
    times_mesh_coarse, freq_mesh_coarse, 0, 1.0, 20e-3, 200e-3
)


plt.imshow(
    surrogate_data,
    extent=(
        times_coarsened[0],
        times_coarsened[-1],
        freqs_coarsened[-1],
        freqs_coarsened[0],
    ),
    aspect="auto",
)
plt.show()


dms = np.geomspace(1e-3, 1000e-3)
toffsets = times_coarsened

# dm_grid,toffset_grid = np.meshgrid(dms,toffsets)


def compute_gini_for_fit(surrogate_data, data, freqs, plot=False):
    frequency_contributions = np.sum(surrogate_data * data_coarsened, axis=1)
    freqs_normed = (freqs - freqs[0]) / (freqs[-1] - freqs[0])
    if plot:
        plt.plot(
            freqs_normed,
            np.cumsum(frequency_contributions) / np.sum(frequency_contributions),
        )
        plt.show()
    gini_index = np.trapz(
        np.abs(
            freqs_normed
            - np.cumsum(frequency_contributions) / np.sum(frequency_contributions)
        ),
        freqs_normed,
    )
    return gini_index


fit_quality = np.zeros(shape=(dms.size, toffsets.size))
ginis = np.zeros(shape=fit_quality.shape)
for i, dm in enumerate(dms):
    for j, toff in enumerate(toffsets):
        print(i, j)
        surrogate_data = surrogate_function(
            times_mesh_coarse, freq_mesh_coarse, toff, 1.0, 20e-3, dm
        )
        fit_quality[i, j] = np.sum(data_coarsened * surrogate_data)
        ginis[i, j] = compute_gini_for_fit(
            surrogate_data, data_coarsened, freqs_coarsened, plot=False
        )


weighted_quality = fit_quality
mask = np.isnan(ginis)
weighted_quality[mask] = 0.0
weighted_quality[np.logical_not(mask)] /= ginis[np.logical_not(mask)]

# plt.imshow(fit_quality)
fig, ax = plt.subplots(1, 3)
ax[0].imshow(fit_quality, extent=(dms[0], dms[-1], toffsets[0], toffsets[-1]))
ax[0].set_xscale("log")
ax[1].imshow(ginis, extent=(dms[0], dms[-1], toffsets[0], toffsets[-1]))
ax[1].set_xscale("log")
ax[2].imshow(weighted_quality, extent=(dms[0], dms[-1], toffsets[0], toffsets[-1]))
ax[2].set_xscale("log")
plt.show()

fit_inds = np.unravel_index(weighted_quality.argmax(), fit_quality.shape)
print(fit_inds)
# fit_inds = np.unravel_index(fit_quality.argmax(),fit_quality.shape)
dm_opt, toffset_opt = dms[fit_inds[0]], toffsets[fit_inds[1]]
print("best:", dm_opt, toffset_opt)

fig, ax = plt.subplots(1, 2, dpi=150)
surrogate_data = surrogate_function(
    times_mesh, freq_mesh, toffset_opt, 1.0, 20e-3, dm_opt
)
ax[0].imshow(
    data,
    extent=(
        times[0],
        times[-1],
        freqs[-1],
        freqs[0],
    ),
    aspect="auto",
)
ax[1].imshow(
    surrogate_data,
    extent=(
        times[0],
        times[-1],
        freqs[-1],
        freqs[0],
    ),
    aspect="auto",
)
plt.show()


# idea; let's look at the gini index
# how many frequency bins participate meaningfully in the fit?


surrogate_data = surrogate_function(
    times_mesh_coarse, freq_mesh_coarse, toffset_opt, 1.0, 20e-3, dm_opt
)
gini_index = compute_gini_for_fit(
    surrogate_data, data_coarsened, freqs_coarsened, plot=True
)

print("gini index: ", gini_index)
plt.show()
