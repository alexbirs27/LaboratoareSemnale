import numpy as np
import matplotlib.pyplot as plt

# same frequencies from exercise 2
freq = 15
fs_old = 8  # sub-Nyquist from ex2

# new Nyquist sampling frequency
fs_nyquist = 200

# same 3 as exercise 2
signal_original = lambda time: np.sin(2 * np.pi * freq * time)
signal_alias1 = lambda time: np.sin(2 * np.pi * (freq + fs_old) * time)
signal_alias2 = lambda time: np.sin(2 * np.pi * (freq + 3 * fs_old) * time)

time_axis = np.linspace(0, 1, 1200)


sample_points = np.linspace(0, 1, fs_nyquist)[:-1]

fig, axs = plt.subplots(3)

# 1
vals_1 = np.vectorize(signal_original)(time_axis)
axs[0].plot(time_axis, vals_1)
samples_1 = np.vectorize(signal_original)(sample_points)
axs[0].scatter(sample_points, samples_1, color='red')

# 2
vals_2 = np.vectorize(signal_alias1)(time_axis)
axs[1].plot(time_axis, vals_2)
samples_2 = np.vectorize(signal_alias1)(sample_points)
axs[1].scatter(sample_points, samples_2, color='red')

# 3
vals_3 = np.vectorize(signal_alias2)(time_axis)
axs[2].plot(time_axis, vals_3)
samples_3 = np.vectorize(signal_alias2)(sample_points)
axs[2].scatter(sample_points, samples_3, color='red')

plt.tight_layout()
plt.savefig(fname='Laborator4/exercitiul3figura1.pdf')
plt.show()