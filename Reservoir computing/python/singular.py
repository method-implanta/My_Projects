import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d

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


def find_divergence_point(array1, array2, threshold):
    # Calculate the absolute difference between the arrays
    differences = np.abs(array1 - array2)

    # Find the first index where the difference exceeds the threshold
    divergent_indices = np.where(differences > threshold)[0]

    if divergent_indices.size > 0:
        return divergent_indices[0]
    else:
        return -1  # Return -1 if no divergence is found


dim_system = 1
dim_reservoir = 300
edge_probability = 0.1
scaling_factor = 1.1
iter_num = 10_000
split_ratio = 0.50
regularization_factor = 0.0001
repetitions = 30  # 每个奇异值重复的次数
loop_time = 30

time_to_divergence = np.zeros(loop_time)
desired_max_singular_value_all = np.zeros(loop_time)
divergence_times = np.zeros((loop_time, repetitions))  # 存储每次重复的divergence time

t, x, y, z = lorenz(t0=0, x0=1, y0=1, z0=1, dt=0.01, sigma=10, rho=28, beta=8 / 3, iter_num=iter_num)

for j in range(loop_time):
    desired_max_singular_value = j * 0.1
    desired_max_singular_value_all[j] = desired_max_singular_value

    for rep in range(repetitions):  # 添加的循环来重复计算
        # 输入权重矩阵
        W_in = 2 * edge_probability * (np.random.rand(dim_reservoir, dim_system) - 0.5)
        reservoir_state = np.zeros(dim_reservoir)

        # 创建图并转换为邻接矩阵
        graph = nx.gnp_random_graph(dim_reservoir, edge_probability)
        graph = nx.to_numpy_array(graph)

        # 生成原始矩阵A
        A = 2 * (np.random.rand(dim_reservoir, dim_reservoir) - 0.5) * graph
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S = S / S.max() * desired_max_singular_value
        A = np.dot(U, np.dot(np.diag(S), Vt))

        W_out = np.zeros((dim_system, dim_reservoir))

        split_idx = int(iter_num * split_ratio)
        t_train, x_train = t[:split_idx], np.array(x[:split_idx])
        t_val, x_val = t[split_idx:], np.array(x[split_idx:])

        R = np.zeros((dim_reservoir, x_train.shape[0]))
        for i in range(x_train.shape[0]):
            R[:, i] = reservoir_state
            reservoir_state = sigmoid(np.dot(A, np.expand_dims(reservoir_state, axis=1))
                                      + np.dot(W_in, x_train[i]))
            reservoir_state = reservoir_state[:, 0]

        Rt = np.transpose(R)
        inverse_part = np.linalg.inv(np.dot(R, Rt)
                                     + regularization_factor * np.identity(R.shape[0]))
        W_out = np.dot(np.dot(x_train.T, Rt), inverse_part)

        step_to_be_predicted = len(x_val)
        x_pred = np.zeros((step_to_be_predicted, dim_system))
        for i in range(step_to_be_predicted):
            x_pred[i] = np.dot(W_out, reservoir_state)
            reservoir_state = sigmoid(np.dot(A, reservoir_state)
                                      + np.dot(W_in, x_pred[i]))
        x_pred_squeezed = x_pred.squeeze()

        divergence_times[j, rep] = find_divergence_point(x_val, x_pred_squeezed, 3)

    time_to_divergence[j] = np.mean(divergence_times[j])
    print(f"Epoch {j + 1}/{loop_time} completed")

smoothed_data = uniform_filter1d(time_to_divergence, size=5)  # 使用均匀滤波进行平滑
plt.plot(desired_max_singular_value_all, smoothed_data)
plt.xlabel('Desired Max Singular Value')
plt.ylabel('Average Divergence Time')
plt.xscale('log', base=10)  # 使用 base 参数设置底数
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))  # 自定义对数刻度的数量
plt.show()

