#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <iostream>
#include <sstream>
#include <fstream>
#include <ios>

#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"

#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#include "mobile_robot_model/mobile_robot_model.h"
#include "acados_solver_mobile_robot.h"

using namespace Eigen;
using std::ofstream;
using std::cout;
using std::endl;
using std::fixed;
using std::showpos;

#define N 30
#define NX 4
#define NU 2
#define NY 6
#define NYN 4

class NMPC {
    enum states {
        x = 0,
        y = 1,
        v = 2,
        theta = 3,
    };

    enum controls {
        u = 0,
        w = 1,
    };

    struct solver_output {
        double u0[NU];
        double x1[NX];
    };

    struct solver_input {
        double x0[NX];
        double yref[NY];
        double yref_e[NYN];
    };


    mobile_robot_solver_capsule *capsule;

    // ROS components
    ros::Publisher cmd_vel_pub;
    ros::Subscriber odometry_sub;

    // ACADOS variables
    int acados_status;
    solver_input acados_in;
    solver_output acados_out;

public:
    NMPC(ros::NodeHandle &n) {

        // Allocate memory for the capsule
        capsule = mobile_robot_acados_create_capsule();

        // Initialize the solver
        int status = mobile_robot_acados_create(capsule);
        if (status != 0) {
            ROS_ERROR("ACADOS initialization failed with status: %d", status);
            exit(1); // Consider a more graceful exit or error handling
        }

        // MODIFIED: Initialize the state (x0) with the initial position and orientation
        acados_in.x0[x] = 0.0; // Initial x
        acados_in.x0[y] = 0.0; // Initial y
        acados_in.x0[theta] = 0.0; // Initial theta
        acados_in.x0[v] = 0.0; // Initial velocity

        // MODIFIED: Set reference (yref) for the final desired state
        double target_x = 5.0, target_y = 5.0; // Target position
        acados_in.yref[0] = target_x; // Desired x
        acados_in.yref[1] = target_y; // Desired y
        acados_in.yref[2] = 0.0; // Desired velocity
        acados_in.yref[3] = 0.0; // Desired theta

    }

    // fix
    void odometryCallback(const nav_msgs::Odometry::ConstPtr &msg) {
        // Assuming the odometry provides position and orientation in a quaternion
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;

        // Convert quaternion to Euler angles (specifically yaw for theta)
        double siny_cosp = 2 * (qw * qz + qx * qy);
        double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
        double theta = std::atan2(siny_cosp, cosy_cosp);

        // Update the NMPC state with current position and orientation
        acados_in.x0[x] = msg->pose.pose.position.x;
        acados_in.x0[y] = msg->pose.pose.position.y;
        acados_in.x0[theta] = theta;

        // If velocity is available, update it as well
        acados_in.x0[v] = std::sqrt(std::pow(msg->twist.twist.linear.x, 2) + std::pow(msg->twist.twist.linear.y, 2));
    }

    //fix
    void control() {

        // Set the initial condition
        ocp_nlp_constraints_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "lbx", acados_in.x0);
        ocp_nlp_constraints_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, 0, "ubx", acados_in.x0);

        // Set the reference for all stages
        for (int i = 0; i < N; ++i) {
            ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, "yref", acados_in.yref);
        }

        // Set the reference for the terminal stage
        ocp_nlp_cost_model_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, N, "yref", acados_in.yref_e);

        // Solve the NMPC problem
        acados_status = mobile_robot_acados_solve(capsule);
        if (acados_status != 0) {
            ROS_ERROR("Acados solve failed with status: %d", acados_status);
            return;
        }

        // Extract the optimal control inputs from the solution
        ocp_nlp_out_get(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_out, 0, "u", acados_out.u0);

        // Publish the control commands
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = acados_out.u0[u]; // Assuming 'u' is linear velocity
        cmd_vel.angular.z = acados_out.u0[w]; // Assuming 'w' is angular velocity
        cmd_vel_pub.publish(cmd_vel);
    }


    ~NMPC() {
        // Free resources associated with the capsule
        mobile_robot_acados_free_capsule(capsule);
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "nmpc_controller");
    ros::NodeHandle n;
    NMPC nmpc(n);

    ros::Rate loop_rate(20); // Control loop rate

    while (ros::ok()) {
        nmpc.control();
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
