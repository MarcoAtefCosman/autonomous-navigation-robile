#!/usr/bin/env python3
""""
- Planner coordinator for robile platform

- Recieves a navigation goal from RViz or Frontier explorer, forwards it to A* for planning,
  then feeds waypoints one at a time to the potential field.
- Uses TF to get robot pose in map frame

Topics:
    Subscribes: /goal_pose (RViz), /exploration_goal (frontier explorer),
                /waypoints (A* planner), /goal_reached (PF planner)
    Publishes:  /plan_request (to A*), /waypoint_goal (to PF planner),
                /coordinator_markers (RViz)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import math

class PlannerCoordinator(Node):
    def __init__(self):
        super().__init__('planner_coordinator')
    
        # Declare parameters
        self.declare_parameter('stuck_timeout', 15.0)
        self.declare_parameter('waypoint_reached_tolerance', 0.3)
        
        self.stuck_timeout = self.get_parameter('stuck_timeout').value
        self.waypoint_tolerance = self.get_parameter('waypoint_reached_tolerance').value
       

        # Subscribers
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.navigation_goal_callback, 10)
        self.exploration_goal_sub = self.create_subscription(PoseStamped, '/exploration_goal', self.exploration_goal_callback, 10)
        self.waypoints_sub = self.create_subscription(Path, '/waypoints', self.waypoints_callback, 10)
        self.reached_sub = self.create_subscription(Bool, '/goal_reached', self.waypoint_reached_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)     
        
        # Publishers
        self.plan_request_pub = self.create_publisher(PoseStamped, '/plan_request', 10)
        self.goal_pose_pub = self.create_publisher(PoseStamped, '/waypoint_goal', 10)
        self.status_marker_pub = self.create_publisher(MarkerArray, '/coordinator_markers', 10)

        # State
        self.curr_x = None
        self.curr_y = None
        self.curr_theta = None
        self.has_pose = False
        self.waypoints = []
        self.current_waypoint_index = 0
        self.final_goal = None
        self.is_navigating = False

        # Stuck detection
        self.last_progress_time = self.get_clock().now()
        self.last_distance = float('inf')

        # Monitor time : tune the timer
        self.create_timer(1, self.monitor_progress)

        self.get_logger().info('Planner Coordinator node initialized')
        
    # Callbacks
    def odom_callback(self, msg):
        """Process odometry."""
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_theta = theta    
        self.has_pose = True

    def navigation_goal_callback(self,msg):
        """
        Recieve a new navigation goal from RViz, forward it to A* as a plan request 
        """
        self.final_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New navigation goal: ({self.final_goal[0]:.2f} , {self.final_goal[1]:.2f})')

        # Reset state
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False

        # Forward to A* planner
        self.plan_request_pub.publish(msg)
        self.get_logger().info(f'Sent plan request to A* planner, according to RViz goal')

    def exploration_goal_callback(self, msg):
        """Handle exploration goals from frontier explorer, forward it to A* as a plan request"""
        self.final_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'New EXPLORATION goal: ({self.final_goal[0]:.2f}, {self.final_goal[1]:.2f})')

        # Reset state 
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False

        # Forward to A* planner
        self.plan_request_pub.publish(msg)
        self.get_logger().info(f'Sent plan request to A* planner, according to exploration goal')
    
    def waypoints_callback(self,msg):
        """
        Recieve waypoints from A* planner.
        Start sending them to the PF planner one by one
        """        
        self.waypoints = [
            (pose.pose.position.x, pose.pose.position.y) for pose in msg.poses
        ]
        if not self.waypoints:
            self.get_logger().warn('Receieved empty waypoints list')
            return
        
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        self.current_waypoint_index = 0
        self.is_navigating = True
        self.last_progress_time = self.get_clock().now()
        self.last_distance = float('inf')

        # Send the waypoint to the PF
        self.send_current_waypoint()
    
    def waypoint_reached_callback(self,msg):
        """
        The PF reports that it reached the current waypoint.
        advance to the next waypoint.
        """
        if not self.is_navigating or not msg.data:
            return
        
        self.get_logger().info(f'waypoint {self.current_waypoint_index+1}/{len(self.waypoints)} reached!')
        
        if self.current_waypoint_index == len(self.waypoints)-1:
            self.get_logger().info('NAVIGATION IS COMPLETE')
            self.is_navigating = False
            self.publish_status_markers()
            return
         
        self.current_waypoint_index += 1
        self.last_progress_time = self.get_clock().now()
        self.last_distance = float('inf')

        self.send_current_waypoint()

    
    # Waypoint management
    def send_current_waypoint(self):
        """
        Publish the current waypoint to /goal_pose for the PF planner
        """
        if self.current_waypoint_index >= len(self.waypoints):
            return
        
        wx, wy = self.waypoints[self.current_waypoint_index]
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.position.x = wx
        goal_msg.pose.position.y = wy
        
        heading = 0
        quat = quaternion_from_euler(0, 0, heading)
        goal_msg.pose.orientation.w = 1.0
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]
        
        self.get_logger().info(f'Sending waypoint {self.current_waypoint_index + 1}' f'/{len(self.waypoints)}: ({wx:.2f}, {wy:.2f})')
        self.goal_pose_pub.publish(goal_msg)
        
        self.publish_status_markers()
    
    # Progress monitoring
    def monitor_progress(self):
        """
        Check if the robot is making progress toward current waypoint
        """
        if not self.is_navigating or not self.waypoints:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            return
        
        if not self.has_pose:
            self.get_logger().warn('Cannot get robot pose in map frame, skipping plan request')
            return

        wx, wy = self.waypoints[self.current_waypoint_index]
        
        distance = math.hypot(wx - self.curr_x, wy - self.curr_y)
        now = self.get_clock().now()
        
        # Check if making progress
        if distance < self.last_distance - 0.05:
            self.last_progress_time = now
            self.last_distance = distance
        
        # Stuck detection
        elapsed = (now - self.last_progress_time).nanoseconds / 1e9
        if elapsed > self.stuck_timeout:
            self.get_logger().warn(f'Stuck for {elapsed:.1f}s! Skipping waypoint')
            self.current_waypoint_index += 1
            self.last_progress_time = now
            self.last_distance = float('inf')
            
            # Skip current waypoint
            if self.current_waypoint_index < len(self.waypoints):
                self.send_current_waypoint()
            else:
                self.get_logger().warn('No more waypoints. Replanning...')
                if self.final_goal is not None:
                    replan_msg = PoseStamped()
                    replan_msg.header.frame_id = 'map'
                    replan_msg.header.stamp = now.to_msg()
                    replan_msg.pose.position.x = self.final_goal[0]
                    replan_msg.pose.position.y = self.final_goal[1]
                    replan_msg.pose.orientation.w = 1.0
                    self.plan_request_pub.publish(replan_msg)
                    self.is_navigating = False          

    
    # Visualization
    def publish_status_markers(self):
        """Show current waypoint target and progress in RViz."""
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        if not self.waypoints:
            return

        # Highlight current target waypoint
        if self.current_waypoint_index < len(self.waypoints):
            wx, wy = self.waypoints[self.current_waypoint_index]

            target = Marker()
            target.header.frame_id = 'map'
            target.header.stamp = stamp
            target.ns = 'current_target'
            target.id = 0
            target.type = Marker.CYLINDER
            target.action = Marker.ADD
            target.pose.position.x = wx
            target.pose.position.y = wy
            target.pose.position.z = 0.1
            target.pose.orientation.w = 1.0
            target.scale.x = 0.4
            target.scale.y = 0.4
            target.scale.z = 0.2
            target.color.r = 0.0
            target.color.g = 1.0
            target.color.b = 1.0
            target.color.a = 0.6
            markers.markers.append(target)

        self.status_marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()    