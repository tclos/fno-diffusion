import h5py
import numpy as np
import matplotlib.pyplot as plt



with h5py.File("data/snl/snl_dataset.h5", "r") as f:
    X = f["X"][:]      # n(omega, theta)
    Y = f["Y"][:]      # SNL(omega, theta)
    omega = f["omega"][:]
    theta = f["theta"][:]

idx = 0
n = X[idx, ..., 0]
snl = Y[idx, ..., 0]



f = omega / (2 * np.pi)

Theta, F = np.meshgrid(theta, f)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="polar")

pcm = ax.pcolormesh(
    Theta,
    F,
    snl,
    shading="auto",
)

ax.set_title("SNL(f, theta) – Polar representation")
ax.set_rlabel_position(135)

plt.colorbar(pcm, ax=ax, pad=0.1)
plt.show()
