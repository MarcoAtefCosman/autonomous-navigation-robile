#!/usr/bin/env python3
"""
Launch file for testing the Potential Field Planner.

Launches:
  1. Gazebo with Robile (uses your existing gazebo launch)
  2. Potential Field Planner node with params from YAML
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    #  Gazebo with Robile 
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robile_gazebo'),
                'launch',
                'gazebo_4_wheel.launch.py'
            )
        )
    )

    # Potential Field Planner 
    planner_node = Node(
        package='amr_perception',
        executable='potential_field_planner',
        name='potential_field_planner',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('amr_perception'),
                'config',
                'planner_params.yaml'
            )
        ]
    )

    return LaunchDescription([
        gazebo_launch,
        planner_node,
    ])
