#!/usr/bin/env python3
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

    particle_filter_node = Node(
        package='amr_perception',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        gazebo_launch,
        particle_filter_node,
    ])
