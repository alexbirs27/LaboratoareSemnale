import numpy as np
import matplotlib.pyplot as plt

signal = lambda t, phi : np.sin(4*np.pi*t + phi)
t = np.linspace(0, 10, 500)

fig, axs = plt.subplots(4)
for i in range(4):
    axs[i].plot(t, signal(t, i*np.pi/2))  # phases: 0, pi/2, pi, 3pi/2
plt.savefig('Laborator2/exercitiul2figura1.pdf')
plt.show()

# Part 2: Signals with noise at different SNR levels
clean_signal = lambda t : np.sin(2*np.pi*t)
noisy_signal = lambda x, gamma, z : x + gamma * z

fig, axs = plt.subplots(4)
x = clean_signal(t)  # Generate the clean signal first

for i, SNR in enumerate([0.1, 1, 10, 100]):
    z = np.random.normal(size=500)
    gamma = np.sqrt(np.linalg.norm(x)**2 / (SNR * np.linalg.norm(z)**2))
    # Calculate gamma from SNR formula: SNR = ||x||^2 / (gamma ^ 2 * ||z||^2)
    # Rearranging: gamma = sqrt(||x||^2 / (SNR * ||z||^2)
    axs[i].plot(t, noisy_signal(x, gamma, z))
    
plt.savefig('Laborator2/exercitiul2figura2.pdf')
plt.show()