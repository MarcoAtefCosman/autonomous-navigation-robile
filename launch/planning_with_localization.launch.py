#!/usr/bin/env python3
"""
Launch file for planning + localization integrated.

Launches:
  1. Gazebo with Robile (includes RViz)
  2. A* Planner
  3. Planner Coordinator
  4. Potential Field Planner
  5. Particle Filter (provides map->odom TF)

NOTE: NO static map->odom transform — the particle filter handles this.

Pre-requisites (run in separate terminal before launching):
  ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$HOME/ros2_ws/src/amr_perception/maps/sim_map.yaml -p use_sim_time:=true
  ros2 lifecycle set /map_server configure
  ros2 lifecycle set /map_server activate

Usage:
  ros2 launch amr_perception planning_with_localization.launch.py

Then drive with teleop first to let particles converge:
  ros2 run teleop_twist_keyboard teleop_twist_keyboard

Once converged, click '2D Goal Pose' in RViz for autonomous navigation.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('amr_perception')
    config_file = os.path.join(pkg_share, 'config', 'planner_params.yaml')

    # 1. Gazebo with Robile + RViz 
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robile_gazebo'),
                'launch',
                'gazebo_4_wheel.launch.py'
            )
        )
    )

    static_tf_fallback = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 2. Particle Filter (provides map->odom TF) 
    particle_filter_node = Node(
        package='amr_perception',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    # 3. A* Planner 
    astar_node = Node(
        package='amr_perception',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
        
    )

    # 4. Planner Coordinator 
    coordinator_node = Node(
        package='amr_perception',
        executable='planner_coordinator',
        name='planner_coordinator',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    # 5. Potential Field Planner 
    pf_node = Node(
        package='amr_perception',
        executable='potential_field_planner',
        name='potential_field_planner',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo_launch,
        static_tf_fallback,
        particle_filter_node,
        astar_node,
        coordinator_node,
        pf_node,
    ])
