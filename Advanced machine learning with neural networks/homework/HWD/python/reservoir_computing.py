import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def lorenz_step(t, x, y, z, dt, sigma, rho, beta):
    """Calculate the next step in the Lorenz system."""
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt

    return t + dt, x + dx, y + dy, z + dz


def lorenz(t0, x0, y0, z0, dt, sigma, rho, beta, iter_num):
    """Calculate the evolution of the Lorenz system."""
    t = np.zeros(iter_num + 1)
    x = np.zeros(iter_num + 1)
    y = np.zeros(iter_num + 1)
    z = np.zeros(iter_num + 1)

    t[0], x[0], y[0], z[0] = t0, x0, y0, z0
    for i in range(iter_num):
        t[i + 1], x[i + 1], y[i + 1], z[i + 1] = (
            lorenz_step(t[i], x[i], y[i], z[i], dt, sigma, rho, beta)
        )

    return t, x, y, z


def sigmoid(x):
    """Compute the sigmoid function for the input array."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

dim_system = 3
dim_reservoir = 300
edge_probability = 0.1
scaling_factor = 1.1
iter_num = 10_000
split_ratio = 0.50
regularization_factor = 0.0001

W_in = 2 * edge_probability * (np.random.rand(dim_reservoir, dim_system) - .5)
reservoir_state = np.zeros(dim_reservoir)

graph = nx.gnp_random_graph(dim_reservoir, edge_probability)
graph = nx.to_numpy_array(graph)
A = 2 * (np.random.rand(dim_reservoir, dim_reservoir) - 0.5) * graph
eigenvalues, _ = np.linalg.eig(A)
A = A / np.absolute(np.amax(eigenvalues)) * scaling_factor

W_out = np.zeros((dim_system, dim_reservoir))

t, x, y, z = lorenz(t0=0, x0=1, y0=1, z0=1, dt=0.01, sigma=10, rho=28, beta=8 / 3, iter_num=iter_num)

xyz = list(zip(x, y, z))

split_idx = int(iter_num * split_ratio)
t_train, xyz_train = t[:split_idx], np.array(xyz[:split_idx])
t_val, xyz_val = t[split_idx:], np.array(xyz[split_idx:])

R = np.zeros((dim_reservoir, xyz_train.shape[0]))
for i in range(xyz_train.shape[0]):
    R[:, i] = reservoir_state
    reservoir_state = sigmoid(np.dot(A, reservoir_state)
                              + np.dot(W_in, xyz_train[i]))

Rt = np.transpose(R)
regularization_factor = 0.0001
inverse_part = np.linalg.inv(np.dot(R, Rt)
                             + regularization_factor * np.identity(R.shape[0]))
W_out = np.dot(np.dot(xyz_train.T, Rt), inverse_part)

step_to_be_predicted = len(xyz_val)

xyz_pred = np.zeros((step_to_be_predicted, dim_system))
for i in range(step_to_be_predicted):
    xyz_pred[i] = np.dot(W_out, reservoir_state)
    reservoir_state = sigmoid(np.dot(A, reservoir_state)
                              + np.dot(W_in, xyz_pred[i]))

x_val, y_val, z_val = xyz_val[:, 0], xyz_val[:, 1], xyz_val[:, 2]
x_pred, y_pred, z_pred = xyz_pred[:, 0], xyz_pred[:, 1], xyz_pred[:, 2]

fig, axs = plt.subplots(3, 1, figsize=(8, 6))

axs[0].plot(t_val, x_val, label="x Lorenz")
axs[0].plot(t_val, x_pred, label="x prediction")
axs[0].set_ylabel("x")
axs[0].legend()

axs[1].plot(t_val, y_val, label="y Lorenz")
axs[1].plot(t_val, y_pred, label="y prediction")
axs[1].set_ylabel("y")
axs[1].legend()

axs[2].plot(t_val, z_val, label="z Lorenz")
axs[2].plot(t_val, z_pred, label="z prediction")
axs[2].set_ylabel("z")
axs[2].legend()

axs[2].set_xlabel("Time Step")

fig.tight_layout()
plt.show()
