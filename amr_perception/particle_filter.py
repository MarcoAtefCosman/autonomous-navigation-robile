#!/usr/bin/env python3
"""
Monte Carlo Localization (Particle Filter) for Robile Platform.

Fixes applied:
1. Log-space weight accumulation to prevent underflow
2. Motion model: position rejection no longer corrupts theta
3. Correct map->odom TF with proper rotation applied to translation
4. Lower resample threshold to avoid over-resampling
5. Variance-adaptive random particle injection (inject less when converged)
6. Tighter sensor model (more beams, smaller sigma) for faster discrimination
7. Reduced motion noise so sensor updates dominate while moving
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
        self.declare_parameter('num_particles', 2000)
        self.declare_parameter('num_beams', 60)           # TUNE: more beams → sharper weights
        self.declare_parameter('max_laser_range', 10.0)
        self.declare_parameter('sigma_hit', 0.15)         # TUNE: tighter Gaussian → stronger discrimination
        self.declare_parameter('alpha_trans', 0.05)       # TUNE: less translational noise
        self.declare_parameter('alpha_rot', 0.1)          # TUNE: less rotational noise
        self.declare_parameter('resample_threshold', 0.25) # TUNE: resample less often
        self.declare_parameter('z_hit', 0.85)             # TUNE: trust sensor hits more
        self.declare_parameter('z_rand', 0.15)
        self.declare_parameter('update_min_distance', 0.05)
        self.declare_parameter('update_min_angle', 0.05)
        self.declare_parameter('random_injection_fraction', 0.02)  # TUNE: 2% base injection
        self.declare_parameter('convergence_var_threshold', 0.1)   # variance below this = converged

        self.num_particles = self.get_parameter('num_particles').value
        self.num_beams = self.get_parameter('num_beams').value
        self.max_laser_range = self.get_parameter('max_laser_range').value
        self.sigma_hit = self.get_parameter('sigma_hit').value
        self.alpha_trans = self.get_parameter('alpha_trans').value
        self.alpha_rot = self.get_parameter('alpha_rot').value
        self.resample_threshold = self.get_parameter('resample_threshold').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.update_min_distance = self.get_parameter('update_min_distance').value
        self.update_min_angle = self.get_parameter('update_min_angle').value
        self.random_injection_fraction = self.get_parameter('random_injection_fraction').value
        self.convergence_var_threshold = self.get_parameter('convergence_var_threshold').value

        # Subscribers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
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
        self.scan_angle_min = None
        self.scan_angle_max = None
        self.scan_angle_increment = None
        self.free_cells = []

        # Estimated pose (for TF broadcasting)
        self.estimated_x = 0.0
        self.estimated_y = 0.0
        self.estimated_theta = 0.0

        # Odometry tracking
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_odom_theta = 0.0
        self.first_odom = True

        # Statistics
        self.update_count = 0
        self.resample_count = 0

        # Timers
        self.create_timer(0.2, self.publish_particles)
        # self.create_timer(1.0, self.print_statistics)

        self.get_logger().info('Particle Filter initialized')

    # -------------------------------------------------------------------------
    # Map
    # -------------------------------------------------------------------------

    def map_callback(self, msg):
        """Initialize map and particles."""
        self.map_utils = MapUtils(msg)
        self.ray_caster = RayCaster(self.map_utils, self.max_laser_range)

        self.free_cells = []
        for y in range(self.map_utils.height):
            for x in range(self.map_utils.width):
                if self.map_utils.is_free(x, y):
                    self.free_cells.append((x, y))

        self.get_logger().info(f'Map received with {len(self.free_cells)} free cells')
        self.initialize_particles()

    def initialize_particles(self):##
        """Uniformly distribute particles in free space."""
        if not self.free_cells:
            self.get_logger().error('No free cells to initialize particles!')
            return

        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles

        indices = np.random.randint(0, len(self.free_cells), self.num_particles)
        for i, idx in enumerate(indices):
            gx, gy = self.free_cells[idx]
            wx, wy = self.map_utils.grid_to_world(gx, gy)
            self.particles[i, 0] = wx
            self.particles[i, 1] = wy
            self.particles[i, 2] = np.random.uniform(-math.pi, math.pi)

        self.initialized = True
        self.first_odom = True
        self.get_logger().info(f'Initialized {self.num_particles} particles')

    # -------------------------------------------------------------------------
    # Sensor / Odometry callbacks
    # -------------------------------------------------------------------------

    def scan_callback(self, msg):
        """Store latest scan."""
        self.latest_scan = msg
        if self.scan_angle_increment is None:
            self.scan_angle_increment = msg.angle_increment
            self.scan_angle_min = msg.angle_min
            self.scan_angle_max = msg.angle_max
            self.get_logger().info(f'Scan configured: {len(msg.ranges)} beams')

    def odom_callback(self, msg):
        """Process odometry and trigger filter updates."""
        q = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_theta = theta

        if self.first_odom:
            self.last_odom_x = self.odom_x
            self.last_odom_y = self.odom_y
            self.last_odom_theta = self.odom_theta
            self.first_odom = False
            return

        if not self.initialized:
            return

        # Compute odometry delta
        dx = self.odom_x - self.last_odom_x
        dy = self.odom_y - self.last_odom_y
        dtheta = self.normalize_angle(self.odom_theta - self.last_odom_theta)

        movement = math.hypot(dx, dy) #
        if movement < self.update_min_distance and abs(dtheta) < self.update_min_angle:
            return

        # Motion update
        self.motion_update(dx, dy, dtheta)

        # Measurement update
        if self.latest_scan is not None:
            self.measurement_update()

            n_eff = 1.0 / np.sum(self.weights ** 2)
            if n_eff < self.resample_threshold * self.num_particles:
                self.resample()
                self.resample_count += 1

        # Advance odometry reference
        self.last_odom_x = self.odom_x
        self.last_odom_y = self.odom_y
        self.last_odom_theta = self.odom_theta
        self.update_count += 1

        self.publish_estimated_pose()

    # -------------------------------------------------------------------------
    # Motion model
    # -------------------------------------------------------------------------

    def motion_update(self, dx, dy, dtheta):
        """
        Odometry motion model with noise.

        FIX #1: When a particle's new position is rejected (wall collision),
        the entire update (position AND orientation) is discarded so the
        particle state remains fully consistent.
        """
        distance = math.hypot(dx, dy)

        if distance > 0.01:
            move_angle = math.atan2(dy, dx)
            rot1 = self.normalize_angle(move_angle - self.last_odom_theta)
            rot2 = self.normalize_angle(dtheta - rot1)
        else:
            rot1 = 0.0
            rot2 = dtheta

        for i in range(self.num_particles):
            noisy_rot1 = rot1 + np.random.normal(0, self.alpha_rot * abs(rot1) + 0.01)
            noisy_rot2 = rot2 + np.random.normal(0, self.alpha_rot * abs(rot2) + 0.01)
            noisy_dist = distance + np.random.normal(0, self.alpha_trans * distance + 0.01)

            x = self.particles[i, 0]
            y = self.particles[i, 1]
            theta = self.particles[i, 2]

            if distance > 0.01:
                new_x = x + noisy_dist * math.cos(theta + noisy_rot1)
                new_y = y + noisy_dist * math.sin(theta + noisy_rot1)
                new_theta = self.normalize_angle(theta + noisy_rot1 + noisy_rot2)

                gx, gy = self.map_utils.world_to_grid(new_x, new_y)
                if self.map_utils.is_in_bounds(gx, gy) and not self.map_utils.is_occupied(gx, gy):
                    # FIX #1: only commit all three values together
                    self.particles[i, 0] = new_x
                    self.particles[i, 1] = new_y
                    self.particles[i, 2] = new_theta
                # else: leave particle completely unchanged
            else:
                # Pure rotation — always safe to apply
                self.particles[i, 2] = self.normalize_angle(theta + noisy_rot2)

    # -------------------------------------------------------------------------
    # Measurement model
    # -------------------------------------------------------------------------

    def measurement_update(self):
        """
        Update particle weights using a beam likelihood model.

        FIX #2: Accumulate log-probabilities to avoid floating-point underflow
        when multiplying many small probabilities together.
        """
        scan = self.latest_scan

        # Subsample beams
        total_beams = len(scan.ranges)
        step = max(1, total_beams // self.num_beams)
        beam_indices = list(range(0, total_beams, step))[:self.num_beams]

        actual_ranges = []
        angles = []

        for idx in beam_indices:
            r = scan.ranges[idx]
            if math.isinf(r) or math.isnan(r):
                r = self.max_laser_range
            elif r > self.max_laser_range:
                r = self.max_laser_range
            elif r < scan.range_min:
                r = scan.range_min

            actual_ranges.append(r)
            angles.append(scan.angle_min + idx * scan.angle_increment)

        gaussian_norm = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)
        log_min = math.log(1e-6)  # floor to avoid log(0)

        # FIX #2: work in log space
        log_weights = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            px = self.particles[i, 0]
            py = self.particles[i, 1]
            ptheta = self.particles[i, 2]

            gx, gy = self.map_utils.world_to_grid(px, py)
            if not self.map_utils.is_in_bounds(gx, gy) or self.map_utils.is_occupied(gx, gy):
                log_weights[i] = -np.inf
                continue

            log_likelihood = 0.0

            for j, angle in enumerate(angles):
                world_angle = ptheta + angle
                expected_range = self.ray_caster.cast_ray(px, py, world_angle)
                z = actual_ranges[j]

                if z >= self.max_laser_range - 0.1:
                    prob = 0.8 if expected_range >= self.max_laser_range - 0.1 else 0.2
                else:
                    diff = z - expected_range
                    gauss = gaussian_norm * math.exp(-0.5 * (diff / self.sigma_hit) ** 2)
                    prob = self.z_hit * gauss + self.z_rand / self.max_laser_range

                # FIX #2: accumulate log-probability with a floor
                log_likelihood += max(math.log(prob), log_min)

            log_weights[i] = log_likelihood

        # Normalize in log space for numerical stability, then exponentiate
        finite_mask = np.isfinite(log_weights)
        if not np.any(finite_mask):
            self.get_logger().warn('All particle weights are zero — keeping uniform weights')
            self.weights = np.ones(self.num_particles) / self.num_particles
            return

        log_weights[~finite_mask] = np.min(log_weights[finite_mask]) - 10.0
        log_weights -= np.max(log_weights)          # shift for numerical stability
        new_weights = np.exp(log_weights)
        self.weights = new_weights / np.sum(new_weights)

    # -------------------------------------------------------------------------
    # Resampling
    # -------------------------------------------------------------------------

    def resample(self):
        """
        Low-variance resampling with variance-adaptive random particle injection.

        FIX #5 (improved): Injection fraction scales down as the particle cloud
        converges (low positional variance), so random particles stop cluttering
        the visualization once the robot is localized. During global localization
        (high variance) the full injection fraction is used to maintain diversity.
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

        # Variance-adaptive injection: scale injection by normalized positional variance.
        # When var_xy < convergence_var_threshold the robot is considered localized
        # and injection drops to near-zero.
        est_x = np.average(self.particles[:, 0], weights=self.weights)
        est_y = np.average(self.particles[:, 1], weights=self.weights)
        var_x = np.average((self.particles[:, 0] - est_x) ** 2, weights=self.weights)
        var_y = np.average((self.particles[:, 1] - est_y) ** 2, weights=self.weights)
        var_xy = (var_x + var_y) / 2.0

        # Fraction scales linearly from 0 (converged) to random_injection_fraction (dispersed)
        scale = min(1.0, var_xy / self.convergence_var_threshold)
        adaptive_fraction = self.random_injection_fraction * scale
        n_random = int(adaptive_fraction * self.num_particles)

        if n_random > 0 and self.free_cells:
            rand_indices = np.random.randint(0, len(self.free_cells), n_random)
            for k, idx in enumerate(rand_indices):
                gx, gy = self.free_cells[idx]
                wx, wy = self.map_utils.grid_to_world(gx, gy)
                new_particles[-(k + 1), 0] = wx
                new_particles[-(k + 1), 1] = wy
                new_particles[-(k + 1), 2] = np.random.uniform(-math.pi, math.pi)

        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    # -------------------------------------------------------------------------
    # Pose estimation & publishing
    # -------------------------------------------------------------------------

    def publish_estimated_pose(self):
        """Compute and publish estimated pose with correct map->odom TF."""
        if not self.initialized:
            return

        # Weighted mean position
        self.estimated_x = np.average(self.particles[:, 0], weights=self.weights)
        self.estimated_y = np.average(self.particles[:, 1], weights=self.weights)

        # Circular mean for orientation
        mean_cos = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        mean_sin = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        self.estimated_theta = math.atan2(mean_sin, mean_cos)

        # Variance monitoring
        var_x = np.average((self.particles[:, 0] - self.estimated_x) ** 2, weights=self.weights)
        var_y = np.average((self.particles[:, 1] - self.estimated_y) ** 2, weights=self.weights)
        var_theta = np.average((self.particles[:, 2] - self.estimated_theta) ** 2, weights=self.weights)

        if self.update_count % 50 == 0:
            self.get_logger().info(
                f'Pose: ({self.estimated_x:.2f}, {self.estimated_y:.2f}, '
                f'{math.degrees(self.estimated_theta):.1f}°) '
                f'Var: ({var_x:.3f}, {var_y:.3f}, '
                f'{math.degrees(math.sqrt(var_theta)):.1f}°)'
            )

        stamp = self.get_clock().now().to_msg()

        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.estimated_x
        pose_msg.pose.position.y = self.estimated_y
        pose_msg.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, self.estimated_theta)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # ===== BROADCAST TF: map -> odom =====
        # FIX #3: properly rotate the odom origin into the map frame.
        # A 2D rigid-body transform T_map_odom satisfies:
        #   p_map = R(dtheta) * p_odom + t
        # Solving for t:
        #   t = p_map_estimated - R(dtheta) * p_odom_current
        dtheta = self.normalize_angle(self.estimated_theta - self.odom_theta)
        cos_dt = math.cos(dtheta)
        sin_dt = math.sin(dtheta)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = (
            self.estimated_x - (cos_dt * self.odom_x - sin_dt * self.odom_y)
        )
        t.transform.translation.y = (
            self.estimated_y - (sin_dt * self.odom_x + cos_dt * self.odom_y)
        )
        t.transform.translation.z = 0.0

        tf_quat = quaternion_from_euler(0, 0, dtheta)
        t.transform.rotation.x = tf_quat[0]
        t.transform.rotation.y = tf_quat[1]
        t.transform.rotation.z = tf_quat[2]
        t.transform.rotation.w = tf_quat[3]

        self.tf_broadcaster.sendTransform(t)

        # if self.update_count % 100 == 0:
        #     self.get_logger().info(
        #         f'map->odom TF: '
        #         f'trans=({t.transform.translation.x:.2f}, {t.transform.translation.y:.2f}) '
        #         f'rot={math.degrees(dtheta):.1f}°'
        #     )

    def publish_particles(self):
        """Publish particles for visualization."""
        if not self.initialized or self.particles is None:
            return

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        step = max(1, self.num_particles // 500)

        for i in range(0, self.num_particles, step):
            pose = Pose()
            pose.position.x = self.particles[i, 0]
            pose.position.y = self.particles[i, 1]
            pose.position.z = 0.0
            quat = quaternion_from_euler(0, 0, self.particles[i, 2])
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            msg.poses.append(pose)

        self.particle_pub.publish(msg)

    # def print_statistics(self):
    #     """Print filter statistics including convergence state."""
    #     if self.initialized and self.update_count > 0:
    #         n_eff = 1.0 / np.sum(self.weights ** 2)
    #         var_x = np.average((self.particles[:, 0] - self.estimated_x) ** 2, weights=self.weights)
    #         var_y = np.average((self.particles[:, 1] - self.estimated_y) ** 2, weights=self.weights)
    #         var_xy = (var_x + var_y) / 2.0
    #         converged = var_xy < self.convergence_var_threshold
    #         self.get_logger().info(
    #             f'Stats — Updates: {self.update_count}, '
    #             f'Resamples: {self.resample_count}, '
    #             f'N_eff: {n_eff:.1f}/{self.num_particles}, '
    #             f'Var_xy: {var_xy:.3f} '
    #             f'[{"CONVERGED" if converged else "localizing..."}]'
    #         )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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