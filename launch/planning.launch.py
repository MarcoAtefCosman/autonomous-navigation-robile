#!/usr/bin/env python3
"""
Launch file for the full planning pipeline.

Launches:
  1. Gazebo with Robile (includes RViz)
  2. Goal relay: /goal_pose -> /clicked_goal (so coordinator intercepts RViz clicks)
  3. Static TF: map -> odom
  4. A* Planner
  5. Planner Coordinator
  6. Potential Field Planner

Pre-requisites (run in separate terminal before launching):
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

    # ── 1. Gazebo with Robile + RViz ─────────────────────────────
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robile_gazebo'),
                'launch',
                'gazebo_4_wheel.launch.py'
            )
        )
    )

    # ── 2. Goal relay: forward /goal_pose to /clicked_goal ───────
    # RViz publishes on /goal_pose when you click '2D Goal Pose'.
    # The coordinator listens on /clicked_goal.
    # This relay bridges the two so the coordinator intercepts
    # every RViz goal click.
    # goal_relay = Node(
    #     package='topic_tools',
    #     executable='relay',
    #     name='goal_relay',
    #     output='screen',
    #     arguments=['/goal_pose', '/clicked_goal']
    # )

    # ── 3. Static TF: map -> odom (identity) ────────────────────
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # ── 4. A* Planner ────────────────────────────────────────────
    astar_node = Node(
        package='amr_perception',
        executable='astar_planner',
        name='astar_planner',
        output='screen',
        parameters=[config_file]
    )

    # ── 5. Planner Coordinator ───────────────────────────────────
    coordinator_node = Node(
        package='amr_perception',
        executable='planner_coordinator',
        name='planner_coordinator',
        output='screen',
        parameters=[config_file]
    )

    # ── 6. Potential Field Planner ───────────────────────────────
    pf_node = Node(
        package='amr_perception',
        executable='potential_field_planner',
        name='potential_field_planner',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        gazebo_launch,
        static_tf,
        astar_node,
        coordinator_node,
        pf_node,
    ])