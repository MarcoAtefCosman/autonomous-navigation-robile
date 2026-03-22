#!/usr/bin/env python3
"""
Frontier-Based Exploration for Robile Platform.

Reads the continuously updating map from SLAM Toolbox,
detects frontiers (free cells adjacent to unknown cells),
clusters them, ranks them, and sends the best frontier
as a goal to the planner coordinator via /exploration_goal.

The coordinator already subscribes to /exploration_goal and
forwards it to A* for path planning.

Topics:
    Subscribes: /map (OccupancyGrid from SLAM), /goal_reached (Bool)
    Publishes:  /exploration_goal (PoseStamped to coordinator),
                /frontier_markers (MarkerArray for RViz)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion
import tf2_ros
import numpy as np
import math
from collections import deque

from amr_perception.utils.map_utils import MapUtils


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # Parameters 
        self.declare_parameter('min_frontier_size', 5)
        self.declare_parameter('exploration_rate', 3.0)
        self.declare_parameter('distance_weight', 1.0)
        self.declare_parameter('size_weight', 2.0)
        self.declare_parameter('goal_reached_tolerance', 0.5)
        self.declare_parameter('blacklist_radius', 0.5)
        self.declare_parameter('max_failed_attempts', 3)
        self.declare_parameter('min_goal_distance', 0.3)
        self.declare_parameter('navigation_timeout', 30.0)

        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.exploration_rate = self.get_parameter('exploration_rate').value
        self.distance_weight = self.get_parameter('distance_weight').value
        self.size_weight = self.get_parameter('size_weight').value
        self.goal_reached_tolerance = self.get_parameter('goal_reached_tolerance').value
        self.blacklist_radius = self.get_parameter('blacklist_radius').value
        self.max_failed_attempts = self.get_parameter('max_failed_attempts').value
        self.min_goal_distance = self.get_parameter('min_goal_distance').value
        self.navigation_timeout = self.get_parameter('navigation_timeout').value

        # TF2 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers 
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_reached_sub = self.create_subscription(
            Bool, '/goal_reached', self.goal_reached_callback, 10)

        # Publishers 
        # Publishes to /exploration_goal — coordinator already listens to this
        self.goal_pub = self.create_publisher(PoseStamped, '/exploration_goal', 10)
        self.frontier_marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10)

        # State 
        self.map_utils = None
        self.current_goal = None
        self.is_exploring = False
        self.waiting_for_navigation = False
        self.goal_sent_time = None
        self.failed_goals = []  # list of (x, y, attempts)
        self.exploration_complete = False

        # Exploration timer 
        self.create_timer(self.exploration_rate, self.exploration_cycle)

        self.get_logger().info('Frontier Explorer initialized')
        self.get_logger().info(f'  min_frontier={self.min_frontier_size}, '
                               f'rate={self.exploration_rate}s')
        self.get_logger().info('Waiting for /map from SLAM...')

    def get_robot_pose(self):
        """Get robot (x, y) in map frame via TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
            return t.transform.translation.x, t.transform.translation.y
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return None

    # CALLBACKS
    def map_callback(self, msg):
        """Receive the evolving map from SLAM Toolbox."""
        self.map_utils = MapUtils(msg)
        if not self.is_exploring:
            self.is_exploring = True
            self.get_logger().info(
                f'Map received: {self.map_utils.width}x{self.map_utils.height}. '
                f'Exploration started!')

    def goal_reached_callback(self, msg):
        """The planner reports the current navigation goal was reached."""
        if msg.data and self.waiting_for_navigation:
            self.get_logger().info('Frontier goal reached! Selecting next...')
            self.waiting_for_navigation = False
            self.current_goal = None

    # Exploration cycle
    def exploration_cycle(self):
        """Main exploration loop — runs every exploration_rate seconds."""
        if self.map_utils is None or self.exploration_complete:
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        curr_x, curr_y = robot_pose

        # Check navigation timeout
        if self.waiting_for_navigation and self.current_goal is not None:
            # Backup: check if close enough
            dist = math.hypot(
                self.current_goal[0] - curr_x,
                self.current_goal[1] - curr_y)
            if dist < self.goal_reached_tolerance:
                self.waiting_for_navigation = False
                self.current_goal = None
            elif self.goal_sent_time is not None:
                elapsed = (self.get_clock().now() - self.goal_sent_time).nanoseconds / 1e9
                if elapsed > self.navigation_timeout:
                    self.get_logger().warn(
                        f'Navigation timeout ({elapsed:.0f}s). Blacklisting frontier.')
                    self.blacklist_frontier(self.current_goal[0], self.current_goal[1])
                    self.waiting_for_navigation = False
                    self.current_goal = None

        if self.waiting_for_navigation:
            return

        # Detect frontiers 
        frontier_cells = self.detect_frontiers()

        if not frontier_cells:
            self.get_logger().info('='*50)
            self.get_logger().info('  NO FRONTIERS — EXPLORATION COMPLETE!')
            self.get_logger().info('='*50)
            self.exploration_complete = True
            return

        # Cluster frontiers 
        clusters = self.cluster_frontiers(frontier_cells)
        clusters = [c for c in clusters if len(c) >= self.min_frontier_size]

        if not clusters:
            self.get_logger().info('No significant frontiers — exploration complete!')
            self.exploration_complete = True
            return

        self.get_logger().info(f'Found {len(clusters)} frontier clusters')

        # Select best frontier 
        best = self.select_best_frontier(clusters, curr_x, curr_y)

        if best is None:
            self.get_logger().warn('All frontiers blacklisted — exploration complete!')
            self.exploration_complete = True
            return

        #  Send goal 
        self.send_exploration_goal(best)

        # Visualize 
        self.publish_frontier_markers(clusters, best)

    # Frontier Detection
    def detect_frontiers(self):
        """Find free cells adjacent to at least one unknown cell."""
        frontiers = []
        for y in range(1, self.map_utils.height - 1):
            for x in range(1, self.map_utils.width - 1):
                if not self.map_utils.is_free(x, y):
                    continue

                for dx, dy in self.map_utils.DIRS_4:
                    nx, ny = x + dx, y + dy
                    if self.map_utils.is_unknown(nx, ny):
                        frontiers.append((x, y))
                        break

        return frontiers

    # Frontier clustering
    def cluster_frontiers(self, frontier_cells):
        """Group nearby frontier cells into clusters."""
        frontier_set = set(frontier_cells)
        visited = set()
        clusters = []

        for cell in frontier_cells:
            if cell in visited:
                continue

            cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                current = queue.popleft()
                cluster.append(current)

                cx, cy = current
                for dx, dy in self.map_utils.DIRS_8:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    # Frontier ranking and selection
    def select_best_frontier(self, clusters, curr_x, curr_y):
        """
        Rank frontiers by: size_weight * norm_size - distance_weight * norm_distance.
        Skip blacklisted and too-close frontiers.
        """
        candidates = []

        for cluster in clusters:
            # Centroid in world coordinates
            cx = sum(c[0] for c in cluster) / len(cluster)
            cy = sum(c[1] for c in cluster) / len(cluster)
            wx, wy = self.map_utils.grid_to_world(int(cx), int(cy))

            if self.is_blacklisted(wx, wy):
                continue

            dist = math.hypot(wx - curr_x, wy - curr_y)
            if dist < self.min_goal_distance:
                continue

            candidates.append({
                'x': wx, 'y': wy,
                'size': len(cluster),
                'distance': dist
            })

        if not candidates:
            return None

        max_size = max(c['size'] for c in candidates)
        max_dist = max(c['distance'] for c in candidates)
        if max_size == 0:
            max_size = 1
        if max_dist == 0:
            max_dist = 1

        best_score = float('-inf')
        best = None

        for c in candidates:
            score = (self.size_weight * c['size'] / max_size -
                     self.distance_weight * c['distance'] / max_dist)
            if score > best_score:
                best_score = score
                best = (c['x'], c['y'])

        return best

    # Goal management
    def send_exploration_goal(self, frontier):
        """Send frontier centroid as /exploration_goal to coordinator."""
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = frontier[0]
        msg.pose.position.y = frontier[1]
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)
        self.current_goal = frontier
        self.waiting_for_navigation = True
        self.goal_sent_time = self.get_clock().now()

        self.get_logger().info(
            f'Exploring frontier at ({frontier[0]:.2f}, {frontier[1]:.2f})')

    def is_blacklisted(self, x, y):
        """Check if a frontier location has been tried too many times."""
        for bx, by, attempts in self.failed_goals:
            if math.hypot(x - bx, y - by) < self.blacklist_radius:
                if attempts >= self.max_failed_attempts:
                    return True
        return False

    def blacklist_frontier(self, x, y):
        """Record a failed frontier attempt."""
        for i, (bx, by, attempts) in enumerate(self.failed_goals):
            if math.hypot(x - bx, y - by) < self.blacklist_radius:
                self.failed_goals[i] = (bx, by, attempts + 1)
                return
        self.failed_goals.append((x, y, 1))

    
    # Visualization
    def publish_frontier_markers(self, clusters, selected):
        """Publish frontier clusters and selected goal as RViz markers."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Clear old
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        # Each cluster as colored points
        for i, cluster in enumerate(clusters):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'frontiers'
            m.id = i
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = m.scale.y = 0.05

            np.random.seed(i * 7)
            m.color.r = float(np.random.uniform(0.3, 1.0))
            m.color.g = float(np.random.uniform(0.3, 1.0))
            m.color.b = float(np.random.uniform(0.3, 1.0))
            m.color.a = 0.7

            for gx, gy in cluster:
                wx, wy = self.map_utils.grid_to_world(gx, gy)
                p = Point()
                p.x = wx
                p.y = wy
                p.z = 0.05
                m.points.append(p)

            markers.markers.append(m)

        # Selected frontier goal (green cylinder)
        if selected is not None:
            g = Marker()
            g.header.frame_id = 'map'
            g.header.stamp = stamp
            g.ns = 'exploration_goal'
            g.id = 999
            g.type = Marker.CYLINDER
            g.action = Marker.ADD
            g.pose.position.x = selected[0]
            g.pose.position.y = selected[1]
            g.pose.position.z = 0.25
            g.pose.orientation.w = 1.0
            g.scale.x = g.scale.y = 0.4
            g.scale.z = 0.5
            g.color.r = 0.0
            g.color.g = 1.0
            g.color.b = 0.0
            g.color.a = 0.8
            markers.markers.append(g)

        # Frontier count text
        txt = Marker()
        txt.header.frame_id = 'map'
        txt.header.stamp = stamp
        txt.ns = 'frontier_info'
        txt.id = 998
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        robot_pose = self.get_robot_pose()
        if robot_pose:
            txt.pose.position.x = robot_pose[0]
            txt.pose.position.y = robot_pose[1]
        txt.pose.position.z = 1.5
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.25
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f'Frontiers: {len(clusters)}'
        markers.markers.append(txt)

        self.frontier_marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
