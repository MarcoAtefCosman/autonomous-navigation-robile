#!/usr/bin/env python3
"""
- Ray Casting utility for Monte Carlo localization
- Simulates laser beams on the occupancy grid map.
Given a pose and an angle, traces a ray unitl it hits an occupied cell or reaches max. range
- Used by the particle filter's sensor model to compare expected vs actual laser readings
"""
import math

class RayCaster:
    def __init__(self, map_utils, max_range=10.0):
        self.map_utils = map_utils
        self.max_range = max_range
        self.resolution = map_utils.resolution

    def cast_ray(self, world_x, world_y, angle):
        """
        Cast a single ray from a world position at a given angle.

        Uses Bresenham-style stepping through the grid.

        Args:
            world_x, world_y: Ray origin in world coordinates (meters).
            angle: Ray direction in world frame (radians).

        Returns:
            Distance to the first occupied cell (meters), or max_range if nothing hit.
        """ 
        # Starting grid cell          
        gx, gy = self.map_utils.world_to_grid(world_x, world_y)
        
        # Ray direction in grid steps
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Step through the grid
        max_steps = int(self.max_range / self.resolution)
        for step in range(1, max_steps + 1):
            # Current position along the ray
            check_x = int(gx + dx * step)
            check_y = int(gy + dy * step)

            # Out of bounds = max range
            if not self.map_utils.is_in_bounds(check_x, check_y):
                return self.max_range

            # Hit an obstacle
            if self.map_utils.is_occupied(check_x, check_y):
                return step * self.resolution

        return self.max_range

    def simulate_scan(self, world_x, world_y, theta, angle_min, angle_max, angle_increment, num_beams=None):
        """
        Simulate a full laser scan from a given pose.

        Args:
            world_x, world_y: Position in world frame (meters).
            theta: Heading in world frame (radians).
            angle_min: Start angle of laser (radians, relative to robot heading).
            angle_max: End angle of laser (radians, relative to robot heading).
            angle_increment: Angle between consecutive beams (radians).
            num_beams: If provided, override the beam count (for subsampling).

        Returns:
            List of range values (meters) for each beam.
        """
        if num_beams is None:
            num_beams = int((angle_max - angle_min) / angle_increment) + 1
        
        # Compute actual angle to cast
        if num_beams <= 1:
            angles = [angle_min]
        else:
            step = (angle_max - angle_min) / (num_beams - 1)
            angles = [angle_min + i * step for i in range(num_beams)]

        ranges = []

        for local_angle in angles:
            # Convert local laser angle to world frame
            world_angle = theta + local_angle
            r = self.cast_ray(world_x, world_y, world_angle)
            ranges.append(r)
        
        return ranges          