#!usr/bin/env python3
"""
- Potential Field Planner for Robile Platform

- Subscribes to /scan for obstacle avoidance and /waypoint_goal for navigation goals.
- Computes attractive + repulsive forces and publishes /cmd_vel.
- Uses TF to get robot pose in map frame

Topics:
    Subscribes: /scan (LaserScan), /waypoint_goal (PoseStamped)
    Publishes:  /cmd_vel (Twist), /goal_reached (Bool)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf_transformations import euler_from_quaternion
import numpy as np
import math

class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__('potential_field_planner')

        # Declare parameters
        self.declare_parameter('ka', 1.5)                   # attractive gain
        self.declare_parameter('kr', 0.3)                   # repulsive gain
        self.declare_parameter('k_angular', 0.4)            # angular correction gain
        self.declare_parameter('rho_0', 0.5)                # obstacle influence distance (m)
        self.declare_parameter('goal_tolerance', 0.25)      # goal reached threshold (m)
        self.declare_parameter('smoothing_factor', 0.3)     # velocity smoothing
        self.declare_parameter('max_linear_speed', 0.3)     # max linear velocity (m/s)
        self.declare_parameter('max_angular_speed', 0.5)    # max angular velocity (rad/s)
        self.declare_parameter('scan_subsample', 3)         # use every n-th laser beam
        self.declare_parameter('use_omnidirectional', True) # use linear.y for omni drive

        self.ka = self.get_parameter('ka').value
        self.kr = self.get_parameter('kr').value
        self.k_angular = self.get_parameter('k_angular').value
        self.rho_0 = self.get_parameter('rho_0').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.smoothing_factor = self.get_parameter('smoothing_factor').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.scan_subsample = self.get_parameter('scan_subsample').value
        self.use_omnidirectional = self.get_parameter('use_omnidirectional').value
        
        # Subscribers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT) # ROS2 QoS profile
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/waypoint_goal', self.goal_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)     
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)
 
        # State
        self.curr_x = None
        self.curr_y = None
        self.curr_theta = None
        self.has_pose = False
        self.goal = None
        self.latest_scan = None

        # Velocity smoother
        self.prev_cmd_x = 0.0
        self.prev_cmd_y = 0.0
        self.prev_cmd_z = 0.0

        # Timer for control loop to publish velocity at fixed rate 10 hz
        self.control_timer = self.create_timer(0.1, self.control_loop) 
        
        self.get_logger().info('Potential Field Planner node initialized')
    
    # Callbacks    
    def odom_callback(self, msg):
        """Process odometry."""
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_theta = theta    
        self.has_pose = True
    
    def scan_callback(self, msg):
        """Store latest laser scan for the control loop."""
        self.latest_scan = msg        
    
    def goal_callback(self, msg):
        """Set new navigation goal."""
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New goal received: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
    
    def control_loop(self):
        """Main control loop: compute forces and publish cmd_vel, runnin at fixed rate."""
        if self.goal is None or not self.has_pose or self.latest_scan is None:
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

        # Compute attractive and repulsive forces
        f_attr_x, f_attr_y = self.compute_attractive_force(goal_x, goal_y)
        f_rep_x, f_rep_y = self.compute_repulsive_force(self.latest_scan)

        # Reduce repulsion influence near the goal
        goal_proximity_scale = min(1.0, distance_to_goal / (2.0 * self.goal_tolerance))
        f_tot_x = f_attr_x + goal_proximity_scale * f_rep_x
        f_tot_y = f_attr_y + goal_proximity_scale * f_rep_y

        # Transform to robot velocity
        cmd_vel = Twist()
        cos_theta = math.cos(self.curr_theta)
        sin_theta = math.sin(self.curr_theta)

        if self.use_omnidirectional:
            
            local_x = f_tot_x * cos_theta + f_tot_y * sin_theta
            local_y = -f_tot_x * sin_theta + f_tot_y * cos_theta

            raw_x = float(np.clip(local_x, -self.max_linear_speed, self.max_linear_speed))
            raw_y = float(np.clip(local_y, -self.max_linear_speed, self.max_linear_speed))

            desired_heading = math.atan2(f_tot_y, f_tot_x)
            heading_error = self.normalize_angle(desired_heading - self.curr_theta)
            raw_z = float(np.clip(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed))

        else:
            # Turn toward force direction
            force_direction = math.atan2(f_tot_y, f_tot_x)
            heading_error = self.normalize_angle(force_direction - self.curr_theta)

            if abs(heading_error) > 0.3:
                raw_x = 0.05
                raw_y = 0.0
            else:
                force_magnitude = math.hypot(f_tot_x, f_tot_y)
                raw_x = float(np.clip(force_magnitude, -self.max_linear_speed, self.max_linear_speed))
                raw_y = 0.0
            raw_z = float(np.clip(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed))
        
        alpha = self.smoothing_factor
        cmd_vel.linear.x = alpha * self.prev_cmd_x + (1 - alpha) * raw_x
        cmd_vel.linear.y = alpha * self.prev_cmd_y + (1 - alpha) * raw_y
        cmd_vel.angular.z = alpha * self.prev_cmd_z + (1 - alpha) * raw_z
 
        self.prev_cmd_x = cmd_vel.linear.x
        self.prev_cmd_y = cmd_vel.linear.y
        self.prev_cmd_z = cmd_vel.angular.z
 
        self.cmd_pub.publish(cmd_vel)


    # Forces computations
    def compute_attractive_force(self, goal_x, goal_y):
        """
        Attractive force: pulls the robot toward the goal.
        """
        dx = goal_x - self.curr_x
        dy = goal_y - self.curr_y
        dist = math.hypot(dx, dy)

        if dist < 1e-3:
            return 0.0, 0.0

        magnitude = min(self.ka * dist, self.ka)

        f_attr_x = magnitude * (dx / dist)
        f_attr_y = magnitude * (dy / dist)

        return f_attr_x, f_attr_y    

    def compute_repulsive_force(self, scan):
        """
        Repulsive force: pushes the robot away from nearby obstacles.
        """
        f_rep_x = 0.0
        f_rep_y = 0.0
        cos_theta = math.cos(self.curr_theta)
        sin_theta = math.sin(self.curr_theta)
        
        for i in range(0,len(scan.ranges),self.scan_subsample):
            r = scan.ranges[i]
            if math.isinf(r) or math.isnan(r) or r < scan.range_min or r > self.rho_0:
                continue

            angle = scan.angle_min + i * scan.angle_increment
            obs_local_x = r * math.cos(angle)
            obs_local_y = r * math.sin(angle)

            # Transform to odom frame
            obs_odom_x = self.curr_x + (obs_local_x * cos_theta - obs_local_y * sin_theta)
            obs_odom_y = self.curr_y + (obs_local_x * sin_theta + obs_local_y * cos_theta)

            dx = self.curr_x - obs_odom_x
            dy = self.curr_y - obs_odom_y
            dist = math.hypot(dx,dy)

            if dist < 1e-3:
                continue

            # Repulsive force magnitude: kr * (1/dist - 1/rho_0) * (1/dist^2)
            magnitude = self.kr * (1.0/dist - 1.0/self.rho_0) * (1.0 / dist**2)
            # magnitude = min(magnitude, 2.0 * self.ka)

            f_rep_x += magnitude * (dx / dist)
            f_rep_y += magnitude * (dy / dist)

        return f_rep_x, f_rep_y

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