import numpy as np
import matplotlib.pyplot as plt

# a) x[n1, n2] = sin(2*pi*n1 + 3*pi*n2)
N = 100
n1 = np.linspace(0, 1, N)
n2 = np.linspace(0, 1, N)

X = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        X[i][j] = np.sin(2*np.pi*n1[i] + 3*np.pi*n2[j])

plt.subplot(1, 2, 1)
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Img')

Y = np.fft.fft2(X)
freq_db = 20 * np.log10(abs(Y))

plt.subplot(1, 2, 2)
plt.imshow(freq_db)
plt.colorbar()
plt.title('Spectru')

plt.tight_layout()
plt.savefig('exercitiul1a.pdf')
plt.show()


# b) x[n1, n2] = sin(4*pi*n1) + cos(6*pi*n2)
X2 = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        X2[i][j] = np.sin(4*np.pi*n1[i]) + np.cos(6*np.pi*n2[j])

plt.subplot(1, 2, 1)
plt.imshow(X2, cmap=plt.cm.gray)
plt.title('Img')

Y2 = np.fft.fft2(X2)
freq_db2 = 20 * np.log10(abs(Y2))

plt.subplot(1, 2, 2)
plt.imshow(freq_db2)
plt.colorbar()
plt.title('Spectru')

plt.tight_layout()
plt.savefig('exercitiul1b.pdf')
plt.show()


# functie pentru c, d, e
def plot_spectru(Y, filename):
    X = np.fft.ifft2(Y)
    X = np.real(X)

    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap=plt.cm.gray)
    plt.title('Img')

    freq_db = 20 * np.log10(abs(Y) + 1)

    plt.subplot(1, 2, 2)
    plt.imshow(freq_db)
    plt.colorbar()
    plt.title('Spectru')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# c) Y[0,5] = Y[0, N-5] = 1
Y3 = np.zeros((N, N), dtype=complex)
Y3[0, 5] = 1
Y3[0, N-5] = 1
plot_spectru(Y3, 'exercitiul1c.pdf')

# d) Y[5,0] = Y[N-5, 0] = 1
Y4 = np.zeros((N, N), dtype=complex)
Y4[5, 0] = 1
Y4[N-5, 0] = 1
plot_spectru(Y4, 'exercitiul1d.pdf')

# e) Y[5,5] = Y[N-5, N-5] = 1
Y5 = np.zeros((N, N), dtype=complex)
Y5[5, 5] = 1
Y5[N-5, N-5] = 1
plot_spectru(Y5, 'exercitiul1e.pdf')
