import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def plot(N, P, x_opt, u_opt, param):

    # Plot reference path and optimal trajectory
    plt.figure()
    plt.plot(P[:, 0], P[:, 1], 'o-', label='Reference Path')
    plt.plot(x_opt[:, 0], x_opt[:, 1], "r-", label="Optimal trajectory")

    # Plot obstacles
    num_obstacles = len(param) // 2
    for i in range(num_obstacles):
        x_obst, y_obst = param[2 * i], param[2 * i + 1]
        plt.plot(x_obst, y_obst, "ro", label=f"Obstacle {i + 1}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()

    # New code for additional graphs
    time_values = np.arange(0, N+1)
    x_opt[-1, 3] = 0.003

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(time_values, P[:, 2], "c-", label="Desired Velocity")
    plt.plot(time_values, x_opt[:, 2], "r-", label="Velocity")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(time_values, P[:, 3], "c-", label="Desired Angle")
    plt.plot(time_values, x_opt[:, 3], "r-", label="Angle")
    plt.xlabel("Time")
    plt.ylabel("Theta")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.stem(time_values[:N], u_opt[:, 0], linefmt="c-", markerfmt="co", basefmt="k-", label="Acceleration")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.stem(time_values[:N], u_opt[:, 1], linefmt="y-", markerfmt="yo", basefmt="k-", label="Angular Velocity")
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Adjust the layout to prevent overlap
    plt.show()
