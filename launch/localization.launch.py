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


    static_tf_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0','0','0','0','0','0','map','odom'],
        output = 'screen'
    )

    particle_filter_node = Node(
        package='amr_perception',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        # gazebo_launch,
        static_tf_map,
        particle_filter_node,
    ])
