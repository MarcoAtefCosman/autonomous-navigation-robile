#!/usr/bin/env python3
"""
Launch file for testing Monte Carlo Localization.

Launches:
  1. Gazebo with Robile (includes RViz)
  2. Particle Filter node

Pre-requisites (run in separate terminal before launching):
  ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$HOME/ros2_ws/src/amr_perception/maps/sim_map.yaml -p use_sim_time:=true
  ros2 lifecycle set /map_server configure
  ros2 lifecycle set /map_server activate

RViz setup:
  - Add Map display        -> topic: /map
  - Add PoseArray display   -> topic: /particle_cloud
  - Add PoseStamped display -> topic: /mcl_pose
  - Add LaserScan display   -> topic: /scan
  - Set Fixed Frame to 'map'
  - Drive with teleop: ros2 run teleop_twist_keyboard teleop_twist_keyboard

Usage:
  ros2 launch amr_perception localization.launch.py
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

    # ── 1. Gazebo with Robile + RViz ─────────────────────────────
    # gazebo_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('robile_gazebo'),
    #             'launch',
    #             'gazebo_4_wheel.launch.py'
    #         )
    #     )
    # )

    # ── 2. Particle Filter ───────────────────────────────────────
    particle_filter_node = Node(
        package='amr_perception',
        executable='particle_filter',
        name='particle_filter',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        # gazebo_launch,
        particle_filter_node,
    ])
