import numpy as np
import matplotlib.pyplot as plt

# param for aliasing demonstration
freq = 15
fs = 8

# 3 signals: original and two aliases
signal_original = lambda time: np.sin(2 * np.pi * freq * time)
signal_alias1 = lambda time: np.sin(2 * np.pi * (freq + fs) * time)
signal_alias2 = lambda time: np.sin(2 * np.pi * (freq + 2 * fs) * time)

time_axis = np.linspace(0, 1, 1200)
sample_points = np.linspace(0, 1, fs + 1)

figure, subplots = plt.subplots(3)

# 1
values_signal1 = np.vectorize(signal_original)(time_axis)
subplots[0].plot(time_axis, values_signal1)
samples1 = np.vectorize(signal_original)(sample_points)
subplots[0].scatter(sample_points, samples1, color='red')

# 2
values_signal2 = np.vectorize(signal_alias1)(time_axis)
subplots[1].plot(time_axis, values_signal2)
samples2 = np.vectorize(signal_alias1)(sample_points)
subplots[1].scatter(sample_points, samples2, color='blue')

# 3
values_signal3 = np.vectorize(signal_alias2)(time_axis)
subplots[2].plot(time_axis, values_signal3)
samples3 = np.vectorize(signal_alias2)(sample_points)
subplots[2].scatter(sample_points, samples3, color='green')

plt.savefig(fname='Laborator4/exercitiul2figura1.pdf')
plt.show()
