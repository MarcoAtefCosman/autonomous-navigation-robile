#!/usr/bin/env python3
"""
Map utilitis for Occupancy Grid Operations
"""

import math

class MapUtils:
    def __init__(self, occupancy_grid_msg):
        self.resolution = occupancy_grid_msg.info.resolution
        self.width = occupancy_grid_msg.info.width
        self.height = occupancy_grid_msg.info.height
        self.origin_x = occupancy_grid_msg.info.origin.position.x
        self.origin_y = occupancy_grid_msg.info.origin.position.y
        self.data = list(occupancy_grid_msg.data)
        self.inflated_data = None

        # Neighbor lookup
        self.DIRS_4 = [(1,0),(-1,0),(0,1),(0,-1)]
        self.DIRS_8 = [
        (-1,-1), (-1,0) , (-1,1),
        (0,-1) ,          (0,1),
        (1,-1) , (1,0)  , (1,1)
        ]
    
    def world_to_grid(self, world_x, world_y):
        """
        Convert world coordinates (meters) to grid indices (pixels)

        Args:
            world_x, world_y : position in the map frame (m)
        Returns:
            grid_x, grid_y : column and row indices in the grid
        """
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x,grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices back to world coordinates 
        Args:
            grid_x, grid_y : column and row indices in the gird
        Returns:
            world_x, world_y : position in meters 
        """
        world_x = grid_x * self.resolution + self.origin_x + self.resolution / 2.0
        world_y = grid_y * self.resolution + self.origin_y + self.resolution / 2.0
        return world_x, world_y
    
    def is_in_bounds(self, grid_x, grid_y):
        """
        Check if gird coordinates are within the map boundaries
        """
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height
    
    def get_cell(self, grid_x, grid_y):
        """
        Get the occupancy value of cell
        Returns:
            0 -> free, 100 -> occupied, -1 -> unknown or out of boundaries
        """
        if not self.is_in_bounds(grid_x, grid_y):
            return -1
        index = grid_y * self.width + grid_x
        return self.data[index]
    
    def is_free(self, grid_x, grid_y):
        """
        Check if a cell is free
        """
        val = self.get_cell(grid_x, grid_y)
        return 0 <= val < 25
    
    def is_occupied(self, grid_x, grid_y):
        """
        Check if a cell is occupied
        """
        val = self.get_cell(grid_x, grid_y)
        return val >= 65
    
    def is_unknown(self, grid_x, grid_y):
        """
        Check if a cell is unknown
        """
        val = self.get_cell(grid_x, grid_y)
        return val == -1
    
    def get_neighbors(self, grid_x, grid_y, eight_connected=True):
        """
        Get valid, non-occupied neighbor cells

        Args:
            grid_x, grid_y : current cell
            eight_connected : diagonal neighbors
        Returns:
            List of (nx, ny): free/unknown neighbors
        """
        directions = self.DIRS_8 if eight_connected else self.DIRS_4
        neighbors = []
        for (dx,dy) in directions:
            nx, ny = grid_x + dx, grid_y + dy
            if self.is_in_bounds(nx, ny) and not self.is_occupied(nx, ny):
                neighbors.append((nx,ny))
        return neighbors
    
    def get_cost(self, from_cell, to_cell):
        """
        Movement cost between adjacent cells.
        Diagonal movement cost sqrt(2), cardinal costs 1.0
        """
        dx = abs(to_cell[0] - from_cell[0])
        dy = abs(to_cell[1] - from_cell[1])

        if dx+dy == 2:
            return math.sqrt(2)
        
        return 1.0
    
    # Inflated map for planning:
    def inflate_obstacles(self, inflation_radius_cells=3):
        """
        Create an inflated copy of the map where obstacles are expanded.

        Args:
            inflation_radius_cells: number of cells to inflate around each obstacle.
        Returns:
            A new list representing the inflated map data.
        """
        inflated = list(self.data)

        for y in range(self.height):
            for x in range(self.width):
                if self.is_occupied(x, y):
                    for dy in range(-inflation_radius_cells, inflation_radius_cells + 1):
                        for dx in range(-inflation_radius_cells, inflation_radius_cells + 1):
                            nx, ny = x + dx, y + dy
                            if (self.is_in_bounds(nx, ny) and dx * dx + dy * dy <= inflation_radius_cells * inflation_radius_cells):
                                idx = ny * self.width + nx
                                if inflated[idx] < 65:  
                                    inflated[idx] = 100

        self.inflated_data = inflated
        return inflated

    def is_free_inflated(self, grid_x, grid_y):
        """Check if a cell is free in the inflated map."""
        if self.inflated_data is None:
            return self.is_free(grid_x, grid_y)
        if not self.is_in_bounds(grid_x, grid_y):
            return False
        index = grid_y * self.width + grid_x
        return 0 <= self.inflated_data[index] < 25

    def get_neighbors_inflated(self, grid_x, grid_y, eight_connected=True):
        """Get neighbors using the inflated map."""
        directions = self.DIRS_8 if eight_connected else self.DIRS_4
        neighbors = []
        for dx, dy in directions:
            nx, ny = grid_x + dx, grid_y + dy
            if self.is_in_bounds(nx, ny) and self.is_free_inflated(nx, ny):
                neighbors.append((nx, ny))
        return neighbors