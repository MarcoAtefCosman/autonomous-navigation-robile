#!/usr/bin/env python3
"""
Frontier-Based Exploration for Robile Platform.

Reads the continuously updating map from SLAM Toolbox,
detects frontiers (free cells adjacent to unknown cells),
clusters them, ranks them, and sends the best frontier
as a goal to the planner coordinator.

Topics:
    Subscribes: /map (OccupancyGrid from SLAM), /odom (Odometry)
    Publishes:  /goal_pose (PoseStamped to coordinator),
                /frontier_markers (MarkerArray for RViz)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
import numpy as np
import math
from collections import deque

from amr_perception.utils.map_utils import MapUtils


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter('min_frontier_size', 5)        # min cells to count as a frontier
        self.declare_parameter('exploration_rate', 2.0)       # seconds between exploration cycles
        self.declare_parameter('distance_weight', 1.0)        # weight for distance in ranking
        self.declare_parameter('size_weight', 2.0)            # weight for frontier size in ranking
        self.declare_parameter('goal_reached_tolerance', 0.5) # meters to consider frontier reached
        self.declare_parameter('blacklist_radius', 0.5)       # meters — don't revisit failed frontiers
        self.declare_parameter('max_failed_attempts', 3)      # attempts before blacklisting a frontier

        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.exploration_rate = self.get_parameter('exploration_rate').value
        self.distance_weight = self.get_parameter('distance_weight').value
        self.size_weight = self.get_parameter('size_weight').value
        self.goal_reached_tolerance = self.get_parameter('goal_reached_tolerance').value
        self.blacklist_radius = self.get_parameter('blacklist_radius').value
        self.max_failed_attempts = self.get_parameter('max_failed_attempts').value

        # ── Subscribers ──────────────────────────────────────────
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.goal_reached_sub = self.create_subscription(
            Bool, '/goal_reached', self.goal_reached_callback, 10)

        # ── Publishers ───────────────────────────────────────────
        self.goal_pub = self.create_publisher(PoseStamped, '/exploration_goal', 10)
        self.frontier_marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10)

        # ── State ────────────────────────────────────────────────
        self.map_utils = None
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.current_goal = None
        self.is_exploring = False
        self.waiting_for_goal_reached = False
        self.failed_goals = []  # list of (x, y, attempts)

        # ── Exploration timer ────────────────────────────────────
        self.explore_timer = self.create_timer(
            self.exploration_rate, self.exploration_cycle)

        self.get_logger().info('Frontier Explorer initialized')
        self.get_logger().info(f'  min_frontier_size={self.min_frontier_size}, '
                               f'rate={self.exploration_rate}s')
        self.get_logger().info('Waiting for /map from SLAM...')

    # ═══════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════

    def map_callback(self, msg):
        """Receive the evolving map from SLAM Toolbox."""
        self.map_utils = MapUtils(msg)
        if not self.is_exploring:
            self.is_exploring = True
            self.get_logger().info(
                f'Map received: {self.map_utils.width}x{self.map_utils.height}. '
                f'Exploration started!')

    def odom_callback(self, msg):
        """Track current robot position."""
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y

    def goal_reached_callback(self, msg):
        """The planner reports the current goal was reached."""
        if msg.data and self.waiting_for_goal_reached:
            self.get_logger().info('Frontier goal reached! Will select next frontier...')
            self.waiting_for_goal_reached = False
            self.current_goal = None

    # ═══════════════════════════════════════════════════════════════
    # EXPLORATION CYCLE
    # ═══════════════════════════════════════════════════════════════

    def exploration_cycle(self):
        """Main exploration loop — runs periodically."""
        if self.map_utils is None:
            return

        # Don't select a new frontier if still navigating to one
        if self.waiting_for_goal_reached:
            # Check if we're close enough (backup check)
            if self.current_goal is not None:
                dist = math.hypot(
                    self.current_goal[0] - self.curr_x,
                    self.current_goal[1] - self.curr_y)
                if dist < self.goal_reached_tolerance:
                    self.waiting_for_goal_reached = False
                    self.current_goal = None
            else:
                return

        if self.waiting_for_goal_reached:
            return

        # ── Detect frontiers ─────────────────────────────────────
        frontier_cells = self.detect_frontiers()

        if not frontier_cells:
            self.get_logger().info('No frontiers found — exploration complete!')
            self.is_exploring = False
            return

        # ── Cluster frontiers ────────────────────────────────────
        clusters = self.cluster_frontiers(frontier_cells)

        # Filter small clusters
        clusters = [c for c in clusters if len(c) >= self.min_frontier_size]

        if not clusters:
            self.get_logger().info('No significant frontiers — exploration complete!')
            self.is_exploring = False
            return

        self.get_logger().info(f'Found {len(clusters)} frontier clusters')

        # ── Rank and select best frontier ────────────────────────
        best_frontier = self.select_best_frontier(clusters)

        if best_frontier is None:
            self.get_logger().warn('All frontiers blacklisted — exploration complete!')
            self.is_exploring = False
            return

        # ── Send goal ────────────────────────────────────────────
        self.send_exploration_goal(best_frontier)

        # ── Visualize ────────────────────────────────────────────
        self.publish_frontier_markers(clusters, best_frontier)

    # ═══════════════════════════════════════════════════════════════
    # FRONTIER DETECTION
    # ═══════════════════════════════════════════════════════════════

    def detect_frontiers(self):
        """
        Find frontier cells: free cells adjacent to at least one unknown cell.

        Returns:
            List of (grid_x, grid_y) frontier cells.
        """
        frontiers = []

        for y in range(1, self.map_utils.height - 1):
            for x in range(1, self.map_utils.width - 1):
                # Must be a free cell
                if not self.map_utils.is_free(x, y):
                    continue

                # Check if any neighbor is unknown
                has_unknown_neighbor = False
                for dx, dy in self.map_utils.DIRS_4:
                    nx, ny = x + dx, y + dy
                    if self.map_utils.is_unknown(nx, ny):
                        has_unknown_neighbor = True
                        break

                if has_unknown_neighbor:
                    frontiers.append((x, y))

        return frontiers

    # ═══════════════════════════════════════════════════════════════
    # FRONTIER CLUSTERING
    # ═══════════════════════════════════════════════════════════════

    def cluster_frontiers(self, frontier_cells):
        """
        Group nearby frontier cells into clusters using flood fill (BFS).

        Args:
            frontier_cells: List of (grid_x, grid_y) frontier cells.

        Returns:
            List of clusters, each cluster is a list of (grid_x, grid_y).
        """
        frontier_set = set(frontier_cells)
        visited = set()
        clusters = []

        for cell in frontier_cells:
            if cell in visited:
                continue

            # BFS to find connected frontier cells
            cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                current = queue.popleft()
                cluster.append(current)

                # Check 8-connected neighbors
                cx, cy = current
                for dx, dy in self.map_utils.DIRS_8:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    # ═══════════════════════════════════════════════════════════════
    # FRONTIER RANKING AND SELECTION
    # ═══════════════════════════════════════════════════════════════

    def select_best_frontier(self, clusters):
        """
        Rank frontier clusters and select the best one.

        Score = size_weight * normalized_size - distance_weight * normalized_distance

        Larger frontiers are preferred (more information gain).
        Closer frontiers are preferred (less travel cost).

        Args:
            clusters: List of frontier clusters.

        Returns:
            (world_x, world_y) centroid of the best frontier, or None.
        """
        candidates = []

        for cluster in clusters:
            # Compute centroid in world coordinates
            cx = sum(c[0] for c in cluster) / len(cluster)
            cy = sum(c[1] for c in cluster) / len(cluster)
            wx, wy = self.map_utils.grid_to_world(int(cx), int(cy))

            # Check if blacklisted
            if self.is_blacklisted(wx, wy):
                continue

            # Distance from robot
            dist = math.hypot(wx - self.curr_x, wy - self.curr_y)

            # Skip frontiers that are too close (probably already explored)
            if dist < 0.3:
                continue

            candidates.append({
                'x': wx,
                'y': wy,
                'size': len(cluster),
                'distance': dist,
                'cluster': cluster
            })

        if not candidates:
            return None

        # Normalize size and distance for scoring
        max_size = max(c['size'] for c in candidates)
        max_dist = max(c['distance'] for c in candidates)

        if max_size == 0:
            max_size = 1
        if max_dist == 0:
            max_dist = 1

        best_score = float('-inf')
        best_frontier = None

        for c in candidates:
            norm_size = c['size'] / max_size
            norm_dist = c['distance'] / max_dist

            score = self.size_weight * norm_size - self.distance_weight * norm_dist

            if score > best_score:
                best_score = score
                best_frontier = (c['x'], c['y'])

        return best_frontier

    # ═══════════════════════════════════════════════════════════════
    # GOAL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def send_exploration_goal(self, frontier):
        """Send a frontier centroid as a navigation goal."""
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = frontier[0]
        goal_msg.pose.position.y = frontier[1]
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)
        self.current_goal = frontier
        self.waiting_for_goal_reached = True

        self.get_logger().info(
            f'Exploring frontier at ({frontier[0]:.2f}, {frontier[1]:.2f})')

    def is_blacklisted(self, x, y):
        """Check if a frontier location has been blacklisted."""
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

    # ═══════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════

    def publish_frontier_markers(self, clusters, selected_frontier):
        """Publish frontier clusters and selected goal as RViz markers."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Clear old markers
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        # Draw each cluster as points
        for i, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'frontiers'
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.05

            # Different color per cluster
            np.random.seed(i)
            marker.color.r = float(np.random.uniform(0.3, 1.0))
            marker.color.g = float(np.random.uniform(0.3, 1.0))
            marker.color.b = float(np.random.uniform(0.3, 1.0))
            marker.color.a = 0.7

            from geometry_msgs.msg import Point
            for gx, gy in cluster:
                wx, wy = self.map_utils.grid_to_world(gx, gy)
                p = Point()
                p.x = wx
                p.y = wy
                p.z = 0.05
                marker.points.append(p)

            markers.markers.append(marker)

        # Highlight selected frontier goal
        if selected_frontier is not None:
            goal_marker = Marker()
            goal_marker.header.frame_id = 'map'
            goal_marker.header.stamp = stamp
            goal_marker.ns = 'exploration_goal'
            goal_marker.id = 999
            goal_marker.type = Marker.CYLINDER
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = selected_frontier[0]
            goal_marker.pose.position.y = selected_frontier[1]
            goal_marker.pose.position.z = 0.25
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.4
            goal_marker.scale.y = 0.4
            goal_marker.scale.z = 0.5
            goal_marker.color.r = 0.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 0.8
            markers.markers.append(goal_marker)

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
