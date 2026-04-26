#!/usr/bin/env python3
"""
Launch file for planning only (no localization).
Uses static identity map->odom TF.

Pre-requisites:
  ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$HOME/ros2_ws/src/amr_perception/maps/sim_map.yaml -p use_sim_time:=true
  ros2 lifecycle set /map_server configure
  ros2 lifecycle set /map_server activate

Usage:
  ros2 launch amr_perception planning.launch.py
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

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robile_gazebo'),
                'launch',
                'gazebo_4_wheel.launch.py'
            )
        )
    )

    astar_node = Node(
        package='amr_perception',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[config_file]
    )

    coordinator_node = Node(
        package='amr_perception',
        executable='planner_coordinator',
        name='planner_coordinator',
        output='screen',
        parameters=[config_file]
    )

    pf_node = Node(
        package='amr_perception',
        executable='potential_field_planner',
        name='potential_field_planner',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        gazebo_launch,
        astar_node,
        coordinator_node,
        pf_node,
    ])
