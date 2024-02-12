import casadi as ca
from acados_template import AcadosModel


def mobile_robot_model(obstacle_coords):
    """
    Define a simple mobile robot model.
    """

    model_name = 'mobile_robot'

    # Define symbolic variables (states)
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    v = ca.MX.sym('v')
    theta = ca.MX.sym('theta')

    x_dot = ca.MX.sym('x_dot')
    y_dot = ca.MX.sym('y_dot')
    theta_dot = ca.MX.sym('theta_dot')
    v_dot = ca.MX.sym('v_dot')

    # Control
    a = ca.MX.sym('a')  # acceleration
    w = ca.MX.sym('w')  # angular velocity

    # Define state and control vectors
    states = ca.vertcat(x, y, v, theta)
    controls = ca.vertcat(a, w)

    rhs = ca.vertcat(states[2] * ca.cos(states[3]),
                     states[2] * ca.sin(states[3]),
                     controls[0],
                     controls[1])

    x_dot = ca.vertcat(x_dot, y_dot, v_dot, theta_dot)

    f_impl = x_dot - rhs

    # Define parameter for obstacle coordinates
    num_obstacles = len(obstacle_coords) // 2  # Since each obstacle has two coordinates (x, y)

    # Define parameter for obstacle coordinates
    p = ca.MX.sym('p', 2, num_obstacles)  # 2D coordinates for each obstacle

    # Obstacle avoidance cost
    J_obst = 0
    b = 0.4
    w = 25
    for i in range(num_obstacles):
        x_obst = p[0, i]
        y_obst = p[1, i]
        d = ((x - x_obst) ** 2 / b ** 2) + ((y - y_obst) ** 2 / b ** 2)
        J_obst += (ca.pi / 2) + ca.atan(w - d * w)


    model = AcadosModel()

    model.cost_y_expr = ca.vertcat(states, controls, J_obst)
    model.cost_y_expr_e = ca.vertcat(states, J_obst)
    model.f_expl_expr = rhs
    model.f_impl_expr = f_impl
    model.x = states
    model.xdot = x_dot
    model.u = controls
    model.p = p
    model.name = model_name

    return model
