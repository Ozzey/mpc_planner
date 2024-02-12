import numpy as np
from robot_model import mobile_robot_model
from acados_template import AcadosOcp, AcadosOcpSolver


def create_ocp_solver(obstacle_coords):
    """
    Create Acados solver for trajectory optimization.
    """

    # Create AcadosOcp object
    ocp = AcadosOcp()

    # Set up the optimization problem
    model = mobile_robot_model(obstacle_coords)
    ocp.model = model

    # --------------------PARAMETERS--------------
    # constants
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 30

    # Setting initial conditions
    ocp.dims.N = N
    ocp.dims.nx = nx
    ocp.dims.nu = nu

    # Set initial condition for the robot
    ocp.constraints.x0 = np.array([0, 0, 0, 0])

    # ---------------------CONSTRAINTS------------------
    # Define constraints on states and control inputs
    ocp.constraints.idxbu = np.array([0, 1])  # indices 0 & 1 of u
    ocp.constraints.idxbx = np.array([0, 1, 2, 3])  # indices 0...3 of x
    ocp.constraints.lbu = np.array([-0.1, -0.3])  # Lower bounds on control inputs
    ocp.constraints.ubu = np.array([0.1, 0.3])  # Upper bounds on control inputs
    ocp.constraints.lbx = np.array([-100, -100, 0, -10])  # Lower bounds on states
    ocp.constraints.ubx = np.array([100, 100, 1, 10])  # Upper bounds on states
    # ---------------------COSTS--------------------------
    # Set up the cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.model.cost_y_expr = ocp.model.cost_y_expr
    ocp.model.cost_y_expr_e = ocp.model.cost_y_expr_e

    ocp.cost.yref = np.zeros(nx + nu + 1)
    ocp.cost.yref_e = np.zeros(nx + 1)

    W_x = np.array([5, 5, 35, 5, 0.001, 0.00001, 15])
    W = np.diag(W_x)
    W_xe = np.array([300, 300, 50, 50, 10])
    W_e = np.diag(W_xe)

    ocp.cost.W = W  # State weights
    ocp.cost.W_e = W_e  # Terminal state weights

    # Define the number of obstacles
    num_obstacles = len(obstacle_coords)
    ocp.parameter_values = np.zeros((num_obstacles))
    # ---------------------SOLVER-------------------------
    ocp.solver_options.tf = 30  # higher for more accurate
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = 10
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.levenberg_marquardt = 3.0
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_tol_stat = 1e2
    ocp.solver_options.nlp_solver_tol_eq = 2e-2

    # Set up Acados solver
    acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
    return ocp, acados_solver
