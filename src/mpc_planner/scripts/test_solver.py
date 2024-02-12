from create_solver import create_ocp_solver
from utils import plot
import numpy as np


def test_solver(obstacle_coords):
    # --------------PARAMETERS-----------
    # Create optimal control problem solver
    ocp, solver = create_ocp_solver(obstacle_coords)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    N = ocp.dims.N

    # ----------------INITIALIZE-------------------
    # Get desired trajectory
    P, R = desired_trajectory(N)

    # Initialize Optimal Trajectory
    x_opt = np.zeros((ocp.dims.N + 1, nx))
    u_opt = np.zeros((ocp.dims.N, nu))

    # Convert the obstacle coordinates to a NumPy array
    obstacle_coords_array = np.array(obstacle_coords).astype(float)

    # Set yref for each stage in the prediction horizon
    cost = np.zeros(1)
    for i in range(N):
        yref = np.concatenate((P[i], R[i], cost))
        solver.set(i, 'p', obstacle_coords_array)
        solver.set(i, "yref", yref)
        solver.set(i, "x", np.transpose(yref[:4]))

    # Set terminal cost
    solver.set(N, 'p', obstacle_coords_array)
    solver.set(N, 'yref', np.concatenate((P[-1], cost)))
    solver.set(N, "x", P[-1])
    # ---------------------SOLVE-----------------

    # Solve optimal control problem
    status = solver.solve()
    if status != 0:
        raise Exception("Solver failed! Status: ", status)

    for i in range(ocp.dims.N+1):
        x = solver.get(i, "x")
        x_opt[i, :] = x
        if i < ocp.dims.N:
            u_opt[i, :] = solver.get(i, "u")

    # ----------------OUTPUTS------------------

    print("---------------------------")
    print("Cost : ", solver.get_cost())
    print("Time: ", format(solver.get_stats('time_tot')))
    print("---------------------------")
    plot(N, P, x_opt, u_opt, obstacle_coords_array)
    # ---------------------PLOT---------------------------


def desired_trajectory(N):

    # Define parameters
    initial_position = (0, 0)
    final_position = (5, 5)

    # Linear interpolation for positions
    x = np.linspace(initial_position[0], final_position[0], N+1)
    y = np.linspace(initial_position[1], final_position[1], N+1)

    # Calculate v and theta for each pair of x and y
    v = 0.3 * np.ones(N+1)
    theta = np.arctan2(y, x)
    v[0] = 0
    v[N] = 0
    theta[N] = 0

    # Combine into p
    p = np.vstack((x, y, v, theta)).T

    a = np.zeros(len(x))  # Acceleration
    w = np.zeros(len(x))  # Angular velocity

    # Combine control variables into matrix r
    r = np.vstack((a, w)).T

    return p, r
