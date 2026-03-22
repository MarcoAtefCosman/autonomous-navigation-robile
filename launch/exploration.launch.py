#!/usr/bin/env python3
"""
Launch file for autonomous frontier-based exploration.

Launches:
  1. Gazebo with Robile (includes RViz)
  2. SLAM Toolbox in online async mode (builds map live)
  3. Static TF fallback: map -> odom (until SLAM provides it)
  4. A* Planner
  5. Planner Coordinator
  6. Potential Field Planner
  7. Frontier Explorer

NO pre-built map needed — SLAM builds it as the robot explores.
NO manual goal clicking — frontier explorer selects goals autonomously.

The flow:
  SLAM builds /map → Frontier Explorer detects frontiers →
  Selects best frontier → Sends /exploration_goal → Coordinator →
  /plan_request → A* plans path → /waypoints → Coordinator →
  /waypoint_goal → PF Planner → /cmd_vel → Robot moves →
  SLAM updates /map → Repeat until no frontiers remain

RViz setup:
  - Add Map display         -> topic: /map
  - Add Path display        -> topic: /planned_path
  - Add Path display        -> topic: /waypoints
  - Add MarkerArray display -> topic: /frontier_markers
  - Add MarkerArray display -> topic: /waypoint_markers
  - Add MarkerArray display -> topic: /coordinator_markers
  - Add LaserScan display   -> topic: /scan
  - Set Fixed Frame to 'map'

Usage:
  ros2 launch amr_perception exploration.launch.py
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

    # 2. SLAM Toolbox (online async — builds map live) 
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_async_launch.py'
            )
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # 3. Static TF fallback: map -> odom 

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # 4. A* Planner 
    astar_node = Node(
        package='amr_perception',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    # 5. Planner Coordinator 
    coordinator_node = Node(
        package='amr_perception',
        executable='planner_coordinator',
        name='planner_coordinator',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    # 6. Potential Field Planner 
    pf_node = Node(
        package='amr_perception',
        executable='potential_field_planner',
        name='potential_field_planner',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    # 7. Frontier Explorer 
    explorer_node = Node(
        package='amr_perception',
        executable='frontier_explorer',
        name='frontier_explorer',
        output='screen',
        parameters=[config_file, {'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo_launch,
        slam_launch,
        static_tf,
        astar_node,
        coordinator_node,
        pf_node,
        explorer_node,
    ])
