class Environment:
    def __init__(self, grid, robots_location, numberofrobots, finished):
        self.grid = grid
        self.numberOfRobots = numberofrobots
        self.finished = finished
        self.robots_location = robots_location # a matrix of the robots actual location in the grid

    def set_robot_location(self, value, index, x_or_y): # adds a number to the current value
        self.robots_location[index][x_or_y] += value