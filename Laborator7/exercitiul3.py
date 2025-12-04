from scipy import datasets
import numpy as np
import matplotlib.pyplot as plt

X = datasets.face(gray=True)

# adaugam zgomot
pixel_noise = 50
noise = np.random.randint(-pixel_noise, pixel_noise+1, size=X.shape)
X_noisy = X + noise

# SNR inainte
SNR_inainte = np.linalg.norm(X) / np.linalg.norm(X - X_noisy)
print(f'SNR inainte: {SNR_inainte:.4f}')

# filtram - eliminam frecventele mici (zgomot)
Y = np.fft.fft2(X_noisy)
freq_db = 20 * np.log10(abs(Y))
prag = 100
Y[freq_db < prag] = 0
X_filtrat = np.fft.ifft2(Y).real

# SNR dupa
SNR_dupa = np.linalg.norm(X) / np.linalg.norm(X - X_filtrat)
print(f'SNR dupa: {SNR_dupa:.4f}')

plt.subplot(1, 3, 1)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title(f'Zgomot (SNR={SNR_inainte:.2f})')

plt.subplot(1, 3, 3)
plt.imshow(X_filtrat, cmap=plt.cm.gray)
plt.title(f'Filtrat (SNR={SNR_dupa:.2f})')

plt.tight_layout()
plt.savefig('exercitiul3.pdf')
plt.show()
