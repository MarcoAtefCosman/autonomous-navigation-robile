#!usr/bin/env python3
"""
- Potential Field Planner for Robile Platform

- Subscribes to /scan for obstacle avoidance and /goal_pose for navigation goals.
- Computes attractive + repulsive forces and publishes /cmd_vel.
- Works standalone (RViz) or with the planner coordinator (waypoints).

- If the Robile is omnidirectional : use linear.x and linear.y
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, Pose, PoseArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion
import numpy as np
import math

class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__('potential_field_planner')

        # Declare parameters
        self.declare_parameter('ka', 1.2)               # attractive gain
        self.declare_parameter('kr', 0.8)               # repulsive gain
        self.declare_parameter('k_angular', 0.8)        # angular correction gain
        self.declare_parameter('rho_0', 0.8)            # obstacle influence distance (m)
        self.declare_parameter('goal_tolerance', 0.25)   # goal reached threshold (m)
        self.declare_parameter('max_linear_speed', 0.35) # max linear velocity (m/s)
        self.declare_parameter('max_angular_speed', 0.8) # max angular velocity (rad/s)
        self.declare_parameter('stuck_threshold', 60)   # stuck threshold
        self.declare_parameter('scan_subsample', 3)     # use every n-th laser beam
        self.declare_parameter('use_omnidirectional', True) # use linear.y for omni drive
        self.declare_parameter('pub_markers', False) # publish attractive and repulsive arrows

        # Load prameters
        self.ka = self.get_parameter('ka').value
        self.kr = self.get_parameter('kr').value
        self.k_angular = self.get_parameter('k_angular').value
        self.rho_0 = self.get_parameter('rho_0').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.stuck_threshold = self.get_parameter('stuck_threshold').value
        self.scan_subsample = self.get_parameter('scan_subsample').value
        self.use_omnidirectional = self.get_parameter('use_omnidirectional').value
        self.pub_markers = self.get_parameter('pub_markers').value

        # Subscribers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT) # ROS2 QoS profile
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/waypoint_goal', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
        if self.pub_markers:
            self.marker_pub = self.create_publisher(MarkerArray, '/potential_field_markers', 10)    

        # State
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_theta = 0.0
        self.goal = None
        self.has_odom = False
        self.latest_scan = None

        # Stukc to local minima
        self.stuck_counter = 0
        self.prev_x = 0
        self.prev_y = 0

        # Timer for control loop to publish velocity at fixed rate 10 hz
        self.control_timer = self.create_timer(0.1, self.control_loop) 
        self.get_logger().info('Potential Field Planner node initialized')

    # Callbacks
    def odom_callback(self, msg):
        """Update current pose from odometry."""
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, self.curr_theta = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.has_odom = True

    def scan_callback(self, msg):
        """Store latest laser scan for the control loop."""
        self.latest_scan = msg        
    
    def goal_callback(self, msg):
        """Set new navigation goal."""
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New goal received: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
    
    def control_loop(self):
        """Main control loop: compute forces and publish cmd_vel, runnin at fixed rate."""
        if self.goal is None or not self.has_odom or self.latest_scan is None:
            return  
        
        goal_x, goal_y = self.goal

        # Check if goal is reached
        distance_to_goal = math.hypot(goal_x - self.curr_x, goal_y - self.curr_y)
        if distance_to_goal < self.goal_tolerance:
            self.get_logger().info(f'Goal reached!, distance to goal {distance_to_goal:.2f} m')
            self.stop_robot()
            self.goal_reached_pub.publish(Bool(data=True))
            self.goal = None
            return
        
        # Apply random walks to overcome local minima
        # movement = math.hypot(self.curr_x - self.prev_x, self.curr_y - self.prev_y)
        # self.prev_x = self.curr_x
        # self.prev_y = self.curr_y
        
        # if movement < 0.01:
        #     self.stuck_counter += 1
        # else:
        #     self.stuck_counter = 0
        
        # if self.stuck_counter > self.stuck_threshold:
        #     self.get_logger().warn('Stuck detected! apply random perturbation')
        #     cmd_vel = Twist()

        #     cmd_vel.linear.x = float(np.random.uniform(-0.2, 0.2))
        #     cmd_vel.linear.y = float(np.random.uniform(-0.2, 0.2))
        #     cmd_vel.angular.z = float(np.random.uniform(-0.5, 0.5))
        #     self.cmd_pub.publish(cmd_vel)
        #     self.stuck_counter = 0
        #     return

        # Compute attractive and repulsive forces
        f_attr_x, f_attr_y = self.compute_attractive_force(goal_x, goal_y)
        f_rep_x, f_rep_y = self.compute_repulsive_force(self.latest_scan)

        f_tot_x = f_attr_x + f_rep_x
        f_tot_y = f_attr_y + f_rep_y

        # Transform to robot velocity
        cmd_vel = Twist()
        cos_theta = math.cos(self.curr_theta)
        sin_theta = math.sin(self.curr_theta)

        if self.use_omnidirectional:
            
            local_x = f_tot_x * cos_theta + f_tot_y * sin_theta
            local_y = -f_tot_x * sin_theta + f_tot_y * cos_theta

            cmd_vel.linear.x = float(np.clip(local_x, -self.max_linear_speed, self.max_linear_speed))
            cmd_vel.linear.y = float(np.clip(local_y, -self.max_linear_speed, self.max_linear_speed))

            desired_heading = math.atan2(goal_y - self.curr_y, goal_x - self.curr_x)
            heading_error = self.normalize_angle(desired_heading - self.curr_theta)
            cmd_vel.angular.z = float(np.clip(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed))

        else:
            # Turn toward force direction
            force_direction = math.atan2(f_tot_y, f_tot_x)
            heading_error = self.normalize_angle(force_direction - self.curr_theta)

            if abs(heading_error) > 0.5:
                cmd_vel.linear.x = 0.05
                cmd_vel.linear.y = 0.0
                cmd_vel.angular.z = float(np.clip(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed)) 
            else:
                force_magnitude = math.hypot(f_tot_x, f_tot_y)
                cmd_vel.linear.x = float(np.clip(force_magnitude, -self.max_linear_speed, self.max_linear_speed))
                cmd_vel.linear.y = 0.0
                cmd_vel.angular.z = float(np.clip(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed))
        
        self.cmd_pub.publish(cmd_vel)
        if self.pub_markers:
            self.publish_markers(f_attr_x, f_attr_y, f_rep_x, f_rep_y, f_tot_x, f_tot_y)

    # Forces computations
    def compute_attractive_force(self, goal_x, goal_y):
        """
        Attractive force: pulls the robot toward the goal.
        Uses a unit vector scaled by ka (constant magnitude)
        so the robot doesn't accelerate wildly when far from the goal.
        """
        dx = goal_x - self.curr_x
        dy = goal_y - self.curr_y
        dist = math.hypot(dx, dy)

        if dist < 1e-3:
            return 0.0, 0.0

        # Constant-magnitude attractive force (unit vector * gain)
        f_attr_x = self.ka * (dx / dist)
        f_attr_y = self.ka * (dy / dist)

        return f_attr_x, f_attr_y    

    def compute_repulsive_force(self, scan):
        """
        Repulsive force: pushes the robot away from nearby obstacles.
        Works in the robot's laser frame, then rotates to odom frame.
        Subsamples the scan for performance.
        """
        f_rep_x = 0.0
        f_rep_y = 0.0

        for i in range(0,len(scan.ranges),self.scan_subsample):
            r = scan.ranges[i]
            if math.isinf(r) or math.isnan(r) or r < scan.range_min or r > self.rho_0:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            obs_local_x = r * math.cos(angle)
            obs_local_y = r * math.sin(angle)

            # Transform to odom frame
            cos_theta = math.cos(self.curr_theta)
            sin_theta = math.sin(self.curr_theta)
            obs_odom_x = self.curr_x + (obs_local_x * cos_theta - obs_local_y * sin_theta)
            obs_odom_y = self.curr_y + (obs_local_x * sin_theta + obs_local_y * cos_theta)

            dx = self.curr_x - obs_odom_x
            dy = self.curr_y - obs_odom_y
            dist = math.hypot(dx,dy)

            if dist < 1e-3:
                continue

            # Repulsive force magnitude: kr * (1/dist - 1/rho_0) * (1/dist^2)
            magnitude = self.kr * (1.0/dist - 1.0/self.rho_0) * (1.0 / dist**2)

            f_rep_x += magnitude * (dx / dist)
            f_rep_y += magnitude * (dy / dist)

        return f_rep_x, f_rep_y

    # Visualization
    def publish_markers(self, f_attr_x, f_attr_y, f_rep_x, f_rep_y, f_total_x, f_total_y):
        """Publish force vectors as RViz markers for debugging."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Attractive force (green arrow)
        markers.markers.append(
            self.make_arrow_marker(0, stamp, f_attr_x, f_attr_y,
                                   r=0.0, g=1.0, b=0.0, ns='attractive'))

        # Repulsive force (red arrow)
        markers.markers.append(
            self.make_arrow_marker(1, stamp, f_rep_x, f_rep_y,
                                   r=1.0, g=0.0, b=0.0, ns='repulsive'))

        # Total force (blue arrow)
        markers.markers.append(
            self.make_arrow_marker(2, stamp, f_total_x, f_total_y,
                                   r=0.0, g=0.4, b=1.0, ns='total'))

        # Goal marker (yellow sphere)
        if self.goal is not None:
            goal_marker = Marker()
            goal_marker.header.frame_id = 'odom'
            goal_marker.header.stamp = stamp
            goal_marker.ns = 'goal'
            goal_marker.id = 3
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = self.goal[0]
            goal_marker.pose.position.y = self.goal[1]
            goal_marker.pose.position.z = 0.3
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.3
            goal_marker.scale.y = 0.3
            goal_marker.scale.z = 0.3
            goal_marker.color.r = 1.0
            goal_marker.color.g = 1.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 0.8
            markers.markers.append(goal_marker)

        self.marker_pub.publish(markers)

    def make_arrow_marker(self, marker_id, stamp, fx, fy, r, g, b, ns='force'):
        """Create an arrow marker from the robot's position in the force direction."""
        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow from robot position to robot position + force vector
        scale = 2.0  # visual scaling factor
        marker.points = []

        from geometry_msgs.msg import Point
        start = Point()
        start.x = self.curr_x
        start.y = self.curr_y
        start.z = 0.3

        end = Point()
        end.x = self.curr_x + fx * scale
        end.y = self.curr_y + fy * scale
        end.z = 0.3

        marker.points.append(start)
        marker.points.append(end)

        marker.scale.x = 0.05  # shaft diameter
        marker.scale.y = 0.1   # head diameter
        marker.scale.z = 0.1   # head length

        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.9

        return marker  

    # Utilies functions
    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()