cmake_minimum_required(VERSION 3.0.2)
project(mpc_planner)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        cv_bridge
        std_msgs
        tf
        sensor_msgs
        )

find_package(
        Eigen3 REQUIRED
)


set(acados_include "/mnt/c/Users/91919/acados/include")
set(acados_lib "/mnt/c/Users/91919/acados/lib")
set(model_path "/mnt/c/Users/91919/mpc_planner/src/mpc_planner/scripts/c_generated_code")

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        scripts/c_generated_code
        ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/mobile_robot_model
        ${PROJECT_SOURCE_DIR}/scripts/c_generated_code/mobile_robot_cost
        ${PROJECT_SOURCE_DIR}/scripts/c_generated_code
        ${acados_include}
        ${acados_include}/blasfeo/include/
        ${acados_include}/hpipm/include/
        ${acados_include}/acados/
        ${acados_include}/qpOASES_e/
)

catkin_package(

)

link_directories(${PROJECT_SOURCE_DIR}/scripts/c_generated_code)

add_executable(mpc_planner src/main.cpp)

target_link_libraries(mpc_planner
        ${model_path}/libacados_ocp_solver_mobile_robot.so
        ${acados_lib}/libblasfeo.so
        ${acados_lib}/libacados.so
        ${catkin_LIBRARIES}
        )
