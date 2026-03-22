#!/usr/bin/env python3
"""
Monte Carlo Localization (Particle Filter) for Robile Platform.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import math

from amr_perception.utils.map_utils import MapUtils
from amr_perception.utils.ray_casting import RayCaster


class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')

        # Parameters
        self.declare_parameter('num_particles', 1500)
        self.declare_parameter('num_subsampling_beams', 50)
        self.declare_parameter('max_laser_range', 10.0)
        self.declare_parameter('sigma_hit', 0.15)
        self.declare_parameter('alpha1', 0.05)
        self.declare_parameter('alpha2', 0.02)
        self.declare_parameter('alpha3', 0.05)
        self.declare_parameter('alpha4', 0.02)
        self.declare_parameter('resample_threshold', 0.6)
        self.declare_parameter('random_particle_ratio', 0.03)
        self.declare_parameter('update_min_distance', 0.03)
        self.declare_parameter('update_min_angle', 0.03)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.1)
        self.declare_parameter('init_spread_radius', 1.0)
        self.declare_parameter('convergence_boost_updates', 30)
        self.declare_parameter('z_short', 0.05)
        self.declare_parameter('z_max', 0.05)
        self.declare_parameter('lambda_short', 0.1)
        self.declare_parameter('local_particles', 0.5)
        self.declare_parameter('random_injection_fraction', 0.02)  # TUNE: 2% base injection
        self.declare_parameter('convergence_var_threshold', 0.1)   # variance below this = converged
        self.declare_parameter('min_scans', 3)
        self.declare_parameter('published_particles', 500)

        self.num_particles = self.get_parameter('num_particles').value
        self.num_subsampling_beams = self.get_parameter('num_subsampling_beams').value
        self.max_laser_range = self.get_parameter('max_laser_range').value
        self.sigma_hit = self.get_parameter('sigma_hit').value
        self.alpha1 = self.get_parameter('alpha1').value
        self.alpha2 = self.get_parameter('alpha2').value
        self.alpha3 = self.get_parameter('alpha3').value
        self.alpha4 = self.get_parameter('alpha4').value
        self.resample_threshold = self.get_parameter('resample_threshold').value
        self.random_particle_ratio = self.get_parameter('random_particle_ratio').value
        self.update_min_distance = self.get_parameter('update_min_distance').value
        self.update_min_angle = self.get_parameter('update_min_angle').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.z_short = self.get_parameter('z_short').value
        self.z_max = self.get_parameter('z_max').value
        self.lambda_short = self.get_parameter('lambda_short').value
        self.local_particles = self.get_parameter('local_particles').value
        self.init_spread_radius = self.get_parameter('init_spread_radius').value
        self.min_scans = self.get_parameter('min_scans').value
        self.convergence_boost_updates = self.get_parameter('convergence_boost_updates').value
        self.published_particles = self.get_parameter('published_particles').value
        self.random_injection_fraction = self.get_parameter('random_injection_fraction').value
        self.convergence_var_threshold = self.get_parameter('convergence_var_threshold').value

        # Subscribers
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, scan_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publishers
        self.particle_pub = self.create_publisher(PoseArray, '/particle_cloud', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/mcl_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # State
        self.map_utils = None
        self.ray_caster = None
        self.particles = None
        self.weights = None
        self.initialized = False
        self.latest_scan = None

        # Odometry tracking
        self.prev_odom_x = None
        self.prev_odom_y = None
        self.prev_odom_theta = None
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.first_odom_received = False
        self.init_odom_x = 0.0
        self.init_odom_y = 0.0
        self.init_odom_theta = 0.0

        # Estimated pose
        self.estimated_x = 0.0
        self.estimated_y = 0.0
        self.estimated_theta = 0.0

        # Scan geometry
        self.scan_angle_min = None
        self.scan_angle_max = None
        self.scan_angle_increment = None

        # Free cells cache
        self.free_cells = None

        # Map received flag (wait for both map and odom before init)
        self.map_received = False
        self.map_msg = None

        # Timers
        self.create_timer(0.2, self.publish_particles)

        self.get_logger().info('Particle Filter initialized')

    # Callbacks

    def map_callback(self, msg):
        """Receive the map. Wait for first odom before initializing."""
        self.map_msg = msg
        self.map_utils = MapUtils(msg)
        self.ray_caster = RayCaster(self.map_utils, self.max_laser_range)

        # Use inflated obstacles
        self.map_utils.inflate_obstacles()

        self.free_cells = []
        for y in range(self.map_utils.height):
            for x in range(self.map_utils.width):
                if self.map_utils.is_free_inflated(x, y):
                    self.free_cells.append((x, y))

        self.map_received = True
        self.get_logger().info(
            f'Map received: {self.map_utils.width}x{self.map_utils.height}'
            f'{len(self.free_cells)} free cells')

        # initialize particles
        if self.first_odom_received and not self.initialized:
            self.initialize_particles()

    def scan_callback(self, msg):
        """Store latest scan and its parameters."""
        self.latest_scan = msg
        if self.scan_angle_min is None:
            self.scan_angle_min = msg.angle_min
            self.scan_angle_max = msg.angle_max
            self.scan_angle_increment = msg.angle_increment
            self.get_logger().info(f'Scan configured: {len(msg.ranges)} beams')

    def odom_callback(self, msg):
        """Process odometry and trigger filter updates."""
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_theta = theta

        # Store initial odom for particle initialization
        if not self.first_odom_received:
            
            self.first_odom_received = True
            
            self.init_odom_x = self.odom_x
            self.init_odom_y = self.odom_y
            self.init_odom_theta = self.odom_theta
            
            self.prev_odom_x = self.odom_x
            self.prev_odom_y = self.odom_y
            self.prev_odom_theta = self.odom_theta

            # initialize particles
            if self.map_received and not self.initialized:
                self.initialize_particles()
            return

        if not self.initialized:
            return

        # Compute odometry delta
        dx = self.odom_x - self.prev_odom_x
        dy = self.odom_y - self.prev_odom_y
        dtheta = self.normalize_angle(self.odom_theta - self.prev_odom_theta)
        distance = math.hypot(dx, dy)

        if distance < self.update_min_distance and abs(dtheta) < self.update_min_angle:
            return

        # PREDICT
        self.motion_update(dx, dy, dtheta)

        self.prev_odom_x = self.odom_x
        self.prev_odom_y = self.odom_y
        self.prev_odom_theta = self.odom_theta

        # UPDATE
        if self.latest_scan is not None:
            self.sensor_update(self.latest_scan)

            n_eff = 1.0 / np.sum(self.weights ** 2)
            if int(n_eff) < self.resample_threshold * self.num_particles:
                self.resample()

        self.publish_estimated_pose()

    # Initialize particles, using initial odometry 
    def initialize_particles(self):
        """
        Initialize particles using odometry as a hint.
        """
        if not self.free_cells:
            self.get_logger().error('No free cells, cannot initialize!')
            return

        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Local particles near the odom starting position
        num_local = int(self.local_particles * self.num_particles)
        num_global = self.num_particles - num_local

        # Local particles: Gaussian spread around odom position
        for i in range(num_local):
            px = self.init_odom_x + np.random.normal(0, self.init_spread_radius)
            py = self.init_odom_y + np.random.normal(0, self.init_spread_radius)
            gx, gy = self.map_utils.world_to_grid(px, py)
            if self.map_utils.is_in_bounds(gx, gy) and self.map_utils.is_free_inflated(gx, gy):
                self.particles[i, 0] = px
                self.particles[i, 1] = py
                # Concentrate orientations near actual heading
                self.particles[i, 2] = self.init_odom_theta + np.random.normal(0, 0.5)
                self.particles[i, 2] = self.normalize_angle(self.particles[i, 2])
            else:
                idx = np.random.randint(0, len(self.free_cells))
                gx, gy = self.free_cells[idx]
                wx, wy = self.map_utils.grid_to_world(gx, gy)
                self.particles[i, 0] = wx
                self.particles[i, 1] = wy
                self.particles[i, 2] = np.random.uniform(-math.pi, math.pi)

        # Global particles: uniformly across free space
        indices = np.random.randint(0, len(self.free_cells), num_global)
        for i, idx in enumerate(indices):
            gx, gy = self.free_cells[idx]
            wx, wy = self.map_utils.grid_to_world(gx, gy)
            self.particles[num_local + i, 0] = wx
            self.particles[num_local + i, 1] = wy
            self.particles[num_local + i, 2] = self.init_odom_theta + np.random.normal(0, 0.5)
            self.particles[num_local + i, 2] = self.normalize_angle(self.particles[num_local + i, 2])

        self.initialized = True
        self.get_logger().info(
            f'Initialized {self.num_particles} particles '
            f'({num_local} local near ({self.init_odom_x:.2f}, {self.init_odom_y:.2f}), '
            f'{num_global} global)')
        
    
    # MOTION MODEL
    def motion_update(self, dx, dy, dtheta):
        """Odometry motion model: rot1 -> translation -> rot2 with noise."""
        distance = math.hypot(dx, dy)

        if distance > 0.01:
            move_angle = math.atan2(dy, dx)
            rot1 = self.normalize_angle(move_angle - self.prev_odom_theta)
            rot2 = self.normalize_angle(dtheta - rot1)
        else:
            rot1 = 0.0
            rot2 = dtheta

        for i in range(self.num_particles):
            rot1_noise = self.alpha1 * abs(rot1) + self.alpha2 * distance
            trans_noise = self.alpha3 * distance + self.alpha4 * (abs(rot1) + abs(rot2))
            rot2_noise = self.alpha1 * abs(rot2) + self.alpha2 * distance

            # noisy_rot1 = rot1 + np.random.normal(0, max(rot1_noise, 1e-4))
            # noisy_trans = distance + np.random.normal(0, max(trans_noise, 1e-4))
            # noisy_rot2 = rot2 + np.random.normal(0, max(rot2_noise, 1e-4))

            noisy_rot1 = rot1 + np.random.normal(0, rot1_noise)
            noisy_trans = distance + np.random.normal(0, trans_noise)
            noisy_rot2 = rot2 + np.random.normal(0, rot2_noise)

            if distance > 0.01:
                new_x = self.particles[i, 0] + noisy_trans * math.cos(self.particles[i, 2] + noisy_rot1)
                new_y = self.particles[i, 1] + noisy_trans * math.sin(self.particles[i, 2] + noisy_rot1)
                new_theta = self.particles[i, 2] + noisy_rot1 + noisy_rot2
                new_theta = self.normalize_angle(new_theta)

                gx, gy = self.map_utils.world_to_grid(new_x, new_y)
                if self.map_utils.is_in_bounds(gx, gy) and not self.map_utils.is_occupied(gx, gy):
                    self.particles[i, 0] = new_x
                    self.particles[i, 1] = new_y
                    self.particles[i, 2] = new_theta
            else:
                self.particles[i, 2] = self.particles[i, 2] + noisy_rot2
                self.particles[i, 2] = self.normalize_angle(self.particles[i, 2])
    
    # SENSOR MODEL
    def sensor_update(self, scan):
        """
        Complete beam-based sensor model with all four components:
        - P_hit: Gaussian around expected range
        - P_short: Exponential for unexpected obstacles
        - P_max: Max range readings
        - P_rand: Uniform random measurements
        
        Uses log-space accumulation to prevent underflow.
        """
        if self.latest_scan is None:
            return

        # Subsample beams for efficiency
        total_beams = len(scan.ranges)
        step = max(1, total_beams // self.num_subsampling_beams)
        beam_indices = list(range(0, total_beams, step))[:self.num_subsampling_beams]

        actual_ranges = []
        actual_angles = []
        valid_beams = 0

        for idx in beam_indices:
            r = scan.ranges[idx]
            
            # Handle invalid readings
            if math.isinf(r) or math.isnan(r):
                r = self.max_laser_range
            elif r < scan.range_min:
                r = scan.range_min
            elif r > self.max_laser_range:
                r = self.max_laser_range
                
            actual_ranges.append(r)
            actual_angles.append(scan.angle_min + idx * scan.angle_increment)
            
            if r < self.max_laser_range - 0.1:
                valid_beams += 1

        # In case of few valid beams, skip update
        if valid_beams < self.min_scans:
            self.get_logger().warn(f'Too few valid beams: {valid_beams}')
            return

        # Pre-compute constants
        gaussian_norm = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)
        log_min = math.log(1e-6)  # Floor to avoid log(0)
        
        # Verify mixing parameters sum to 1.0
        z_sum = self.z_hit + self.z_short + self.z_max + self.z_rand
        if abs(z_sum - 1.0) > 0.01:
            self.get_logger().warn(f'Mixing parameters sum to {z_sum:.2f}, should be 1.0')
            # Normalize to ensure sum = 1.0
            total = z_sum
            self.z_hit /= total
            self.z_short /= total
            self.z_max /= total
            self.z_rand /= total

        # Work in log space to prevent underflow
        log_weights = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            px = self.particles[i, 0]
            py = self.particles[i, 1]
            ptheta = self.particles[i, 2]

            # Check if particle is in valid area
            gx, gy = self.map_utils.world_to_grid(px, py)
            if not self.map_utils.is_in_bounds(gx, gy) or self.map_utils.is_occupied(gx, gy):
                log_weights[i] = -float('inf')
                continue

            # Calculate P(Z|x,m) = P_HIT + P_SHORT + P_MAX + P_RAND
            log_likelihood = 0.0

            for j, angle in enumerate(actual_angles):
                # Cast ray from particle pose, cast ray depends on 
                # the angle in world frame 
                world_angle = ptheta + angle
                
                # z_exp : from static world, z : from scan measurement
                expected_range = self.ray_caster.cast_ray(px, py, world_angle)
                z = actual_ranges[j]

                # 1. P_HIT: Gaussian around expected range
                diff = z - expected_range
                p_hit = gaussian_norm * math.exp(-0.5 * (diff / self.sigma_hit) ** 2)
                
                # 2. P_SHORT: Exponential for unexpected obstacles (closer than expected)
                if z < expected_range:
                    # Exponential distribution: lambda * exp(-lambda * z)
                    p_short = self.lambda_short * math.exp(-self.lambda_short * z)
                    p_short /= (1.0 - math.exp(-self.lambda_short * expected_range))
                else:
                    p_short = 0.0
                
                # 3. P_MAX: Max range readings
                if z >= self.max_laser_range - 0.1:
                    if expected_range >= self.max_laser_range - 0.1:
                        p_max = 1.0  # Both are max range
                    else:
                        p_max = 0.0  # Expected obstacle but got max range
                else:
                    p_max = 0.0
                
                # 4. P_RAND: Uniform random measurements
                p_rand = 1.0 / self.max_laser_range
                
                prob = self.z_hit * p_hit + self.z_short * p_short + self.z_max * p_max + self.z_rand * p_rand
                
                # Accumulate log probability
                log_likelihood += max(math.log(prob),log_min)

            log_weights[i] = log_likelihood

    
        finite_mask = np.isfinite(log_weights)    
        if not np.any(finite_mask):
            # All particles are invalid - reinitialize
            self.get_logger().warn('All particle weights are -inf! Reinitializing...')
            self.initialize_particles()
            return

        # Set infinite values to a very low number 
        min_finite = np.min(log_weights[finite_mask])
        log_weights[~finite_mask] = min_finite - 10.0

        # Shift for numerical stability (prevents exp overflow)
        log_weights -= np.max(log_weights)

        # Convert back to linear space
        new_weights = np.exp(log_weights)

        # Normalize
        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            self.weights = new_weights / weight_sum
        else:
            self.get_logger().warn('Weight sum is zero, keeping previous weights')


    # RESAMPLING
    def resample(self):
        """
        Low-variance resampling with variance-adaptive random particle injection.
        """
        new_particles = np.zeros_like(self.particles)

        # Systematic resampling
        positions = (np.random.random() + np.arange(self.num_particles)) / self.num_particles
        cumulative_sum = np.cumsum(self.weights)

        i = 0
        for j in range(self.num_particles):
            while positions[j] > cumulative_sum[i]:
                i += 1
            new_particles[j] = self.particles[i]

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    # Pose Estimation & TF
    def publish_estimated_pose(self):
        """Compute weighted mean pose, publish PoseStamped and TF."""
        if not self.initialized:
            return

        stamp = self.get_clock().now().to_msg()

        self.estimated_x = np.average(self.particles[:, 0], weights=self.weights)
        self.estimated_y = np.average(self.particles[:, 1], weights=self.weights)

        mean_cos = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        mean_sin = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        self.estimated_theta = math.atan2(mean_sin, mean_cos)

        # Estimated weighted pose
        self.get_logger().info(f'Estimated Robot pose : x = {self.estimated_x:.3f}, y = {self.estimated_y:.3f},'
                               f'heading = {(self.estimated_theta*180/np.pi):.3f}')
        
        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.estimated_x
        pose_msg.pose.position.y = self.estimated_y
        quat = quaternion_from_euler(0, 0, self.estimated_theta)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # TF: map -> odom
        dtheta = self.normalize_angle(self.estimated_theta - self.odom_theta)
        cos_dt = math.cos(dtheta)
        sin_dt = math.sin(dtheta)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = self.estimated_x - (cos_dt * self.odom_x - sin_dt * self.odom_y)
        t.transform.translation.y = self.estimated_y - (sin_dt * self.odom_x + cos_dt * self.odom_y)
        t.transform.translation.z = 0.0
        tf_quat = quaternion_from_euler(0, 0, dtheta)
        t.transform.rotation.x = tf_quat[0]
        t.transform.rotation.y = tf_quat[1]
        t.transform.rotation.z = tf_quat[2]
        t.transform.rotation.w = tf_quat[3]
        self.tf_broadcaster.sendTransform(t)

    def publish_particles(self):
        """Publish particle cloud for RViz."""
        if not self.initialized or self.particles is None:
            return

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        step = max(1, self.num_particles // self.published_particles)
        for i in range(0, self.num_particles, step):
            pose = Pose()
            pose.position.x = self.particles[i, 0]
            pose.position.y = self.particles[i, 1]
            quat = quaternion_from_euler(0, 0, self.particles[i, 2])
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            msg.poses.append(pose)

        self.particle_pub.publish(msg)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
