#!usr/bin/env python3

"""
- A* planner for robile platform

- Subscribes to /map and /plan_request (goal), runs A* on the occupancy grid.
- extract waypoints using RDP Simplification and publishes results.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid,Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import heapq
import math

from amr_perception.utils.map_utils import MapUtils

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        # Declare parameters
        self.declare_parameter('inflation_radius', 3)   # cells to inflate around obstacles
        self.declare_parameter('rdp_epsilon', 0.3)  # RDP simplification tolerance 
        self.declare_parameter('max_waypoint_spacing', 1.5) # max distance between waypoints
        self.declare_parameter('use_eight_connected', True)

        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.rdp_epsilon = self.get_parameter('rdp_epsilon').value
        self.max_waypoint_spacing = self.get_parameter('max_waypoint_spacing').value
        self.use_eight_connected = self.get_parameter('use_eight_connected').value

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/plan_request', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.raw_path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.waypoint_path_pub = self.create_publisher(Path, '/waypoints', 10)
        self.waypoint_marker_pub = self.create_publisher(MarkerArray, '/waypoint_markers', 10)

        # State
        self.map_utils = None
        self.has_odom = False
        self.curr_x = 0.0
        self.curr_y = 0.0

        self.get_logger().info('A* Planner node initialized')

    # Callbacks
    def odom_callback(self,msg):
        """Update current pose from odometry."""
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.has_odom = True
    
    def map_callback(self,msg):
        """Recieve and process the occupancy grid map"""
        self.map_utils = MapUtils(msg)
        self.map_utils.inflate_obstacles(self.inflation_radius)
        self.get_logger().info(f'Map received: {self.map_utils.width}x{self.map_utils.height}, resolution={self.map_utils.resolution}m')
    
    def goal_callback(self,msg):
        """Recieve a goal and plan a path to it"""
        if self.map_utils is None or not self.has_odom:
            self.get_logger().warn('No map recieved / no odomoetry information available...')
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        # Grid coordinates
        start_grid = self.map_utils.world_to_grid(self.curr_x, self.curr_y)
        goal_grid = self.map_utils.world_to_grid(goal_x, goal_y)
        
        # Validate start and goal
        if not self.map_utils.is_in_bounds(*start_grid):
            self.get_logger().error('Start position is outside map bounds')
            return
        if not self.map_utils.is_in_bounds(*goal_grid):
            self.get_logger().error('Goal position is outside map bounds')
            return
        if self.map_utils.is_occupied(*goal_grid):
            self.get_logger().error('Goal is inside an obstacle')
            return
    
        # Run A*
        grid_path = self.astar(start_grid, goal_grid)

        if grid_path is None:
            self.get_logger().error('A* failed to find a path')
            return
        
        self.get_logger().info(f'A* found a path with length {len(grid_path)}')

        # Convert back to world coordinates
        world_path = [self.map_utils.grid_to_world(*cell) for cell in grid_path]

        # Publish raw world_path
        self.publish_path(world_path, self.raw_path_pub)

        # Extract waypoints
        waypoints = self.extract_waypoints(world_path)
        self.get_logger().info(f'Extracted {len(waypoints)} waypoints from {len(world_path)} path points')

        # Publish waypoints
        self.publish_path(waypoints, self.waypoint_path_pub)
        self.publish_waypoint_markers(waypoints)

    # A*
    def astar(self, start, goal):
        """
        A* search on the inflated occupancy grid.

        Args:
            start: (grid_x, grid_y) start cell.
            goal: (grid_x, grid_y) goal cell.

        Returns:
            List of (grid_x, grid_y) from start to goal, or None if no path.
        """
        open_set = []  # priority queue: (f_cost, counter, node)
        counter = 0    # tie-breaker for equal f_costs
        heapq.heappush(open_set, (0, counter, start))

        came_from = {}
        g_cost = {start: 0}
        f_cost = {start: self.heuristic(start, goal)}
        closed_set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor in self.map_utils.get_neighbors_inflated(
                    current[0], current[1], self.use_eight_connected):

                if neighbor in closed_set:
                    continue

                tentative_g = g_cost[current] + self.map_utils.get_cost(current, neighbor)

                if tentative_g < g_cost.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_cost[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return None  # no path found

    @staticmethod
    def heuristic(a, b):
        """Euclidean distance heuristic."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def reconstruct_path(came_from, current):
        """Trace back from goal to start to build the path."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path   

    # Waypoint Extraction
    def extract_waypoints(self, world_path):
        """
        Reduce a dense path to key waypoints using RDP simplification,
        then ensure no two consecutive waypoints are too far apart.

        Args:
            world_path: List of (x, y) world coordinates.

        Returns:
            List of (x, y) waypoints.
        """
        if len(world_path) <= 2:
            return world_path

        # Step 1: RDP simplification
        simplified = self.rdp_simplify(world_path, self.rdp_epsilon)

        # Step 2: Ensure max spacing between consecutive waypoints
        final_waypoints = [simplified[0]]
        for i in range(1, len(simplified)):
            prev = final_waypoints[-1]
            curr = simplified[i]
            dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])

            if dist > self.max_waypoint_spacing:
                # Insert intermediate waypoints
                num_intermediates = int(dist / self.max_waypoint_spacing)
                for j in range(1, num_intermediates + 1):
                    t = j / (num_intermediates + 1)
                    interp_x = prev[0] + t * (curr[0] - prev[0])
                    interp_y = prev[1] + t * (curr[1] - prev[1])
                    final_waypoints.append((interp_x, interp_y))

            final_waypoints.append(curr)

        return final_waypoints

    @staticmethod
    def rdp_simplify(points, epsilon):
        """
        Ramer-Douglas-Peucker line simplification algorithm.

        Keeps points where the path changes direction significantly.
        Removes redundant points along straight sections.

        Args:
            points: List of (x, y) tuples.
            epsilon: Maximum perpendicular distance tolerance.

        Returns:
            Simplified list of (x, y) tuples.
        """
        if len(points) <= 2:
            return list(points)

        # Find the point farthest from the line between start and end
        start = points[0]
        end = points[-1]
        max_dist = 0
        max_idx = 0

        for i in range(1, len(points) - 1):
            dist = AStarPlanner.perpendicular_distance(points[i], start, end)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        # If max distance exceeds epsilon, split and recurse
        if max_dist > epsilon:
            left = AStarPlanner.rdp_simplify(points[:max_idx + 1], epsilon)
            right = AStarPlanner.rdp_simplify(points[max_idx:], epsilon)
            return left[:-1] + right  
        else:
            return [start, end]

    @staticmethod
    def perpendicular_distance(point, line_start, line_end):
        """
        Calculate perpendicular distance from a point to a line segment.
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        dx = x2 - x1
        dy = y2 - y1
        line_len_sq = dx * dx + dy * dy

        if line_len_sq < 1e-10:
            return math.hypot(px - x1, py - y1)

        # Project point onto line, clamped to segment
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / line_len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.hypot(px - proj_x, py - proj_y)    
    
    # Waypoint publish
    def publish_path(self, world_path, publisher):
        """Publish a list of (x,y) as a nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for wx, wy in world_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        publisher.publish(path_msg)

    def publish_waypoint_markers(self, waypoints):
        """Publish waypoints as numbered sphere markers in RViz."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Clear old markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        for i, (wx, wy) in enumerate(waypoints):
            # Sphere marker
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp
            marker.ns = 'waypoints'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.2
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15

            # First waypoint green, last red, middle yellow
            if i == 0:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            elif i == len(waypoints) - 1:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.8, 0.0
            marker.color.a = 1.0

            markers.markers.append(marker)

            # Text label with waypoint number
            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = stamp
            text_marker.ns = 'waypoint_labels'
            text_marker.id = i + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = wx
            text_marker.pose.position.y = wy
            text_marker.pose.position.z = 0.5
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f'W{i}'
            markers.markers.append(text_marker)

        self.waypoint_marker_pub.publish(markers)    

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()    