from scipy import datasets
import numpy as np
import matplotlib.pyplot as plt

X = datasets.face(gray=True)

Y = np.fft.fft2(X)
freq_db = 20 * np.log10(abs(Y))

# prag SNR
prag = 100
Y[freq_db < prag] = 0

X_comp = np.fft.ifft2(Y).real

plt.subplot(1, 2, 1)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(X_comp, cmap=plt.cm.gray)
plt.title(f'Comprimat (prag={prag}dB)')

plt.tight_layout()
plt.savefig('exercitiul2.pdf')
plt.show()
