import numpy as np
import matplotlib.pyplot as plt

N = 100

# a. Semnal aleator
x = np.random.rand(N)

fig, axs = plt.subplots(4, 1, figsize=(10, 10))

axs[0].plot(x)
axs[0].set_title('Original')

x1 = np.convolve(x, x)
axs[1].plot(x1)
axs[1].set_title('Dupa 1 convolutie')

x2 = np.convolve(x1, x1)
axs[2].plot(x2)
axs[2].set_title('Dupa 2 convolutii')

x3 = np.convolve(x2, x2)
axs[3].plot(x3)
axs[3].set_title('Dupa 3 convolutii')

plt.tight_layout()
plt.savefig('exercitiul2figura1.pdf')
plt.show()

# b. Semnal bloc rectangular
x_bloc = np.zeros(N)
x_bloc[40:60] = 1  # bloc de la 40 la 60

fig, axs = plt.subplots(4, 1, figsize=(10, 10))

axs[0].plot(x_bloc)
axs[0].set_title('Bloc original')

xb1 = np.convolve(x_bloc, x_bloc)
axs[1].plot(xb1)
axs[1].set_title('Dupa 1 convolutie')

xb2 = np.convolve(xb1, xb1)
axs[2].plot(xb2)
axs[2].set_title('Dupa 2 convolutii')

xb3 = np.convolve(xb2, xb2)
axs[3].plot(xb3)
axs[3].set_title('Dupa 3 convolutii')

plt.tight_layout()
plt.savefig('exercitiul2figura2.pdf')
plt.show()

print("Observatii:")
print("- Semnalul devine din ce in ce mai lat (suportul creste)")
print("- Semnalul devine din ce in ce mai neted (trece spre Gaussiana)")
print("- Pentru bloc: forma devine triunghiulara apoi Gaussiana")
