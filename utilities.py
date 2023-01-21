from enum import IntEnum


GRID_SIZE = 64
def grid_move_checker(grid, x1, y1, x2, y2): # checks if movement is legal
    if grid[x2, y2] != -1:
        return True
    return False

class Direction(IntEnum):
    RB = 0
    R = 1
    RT = 2
    T = 3
    LT = 4
    L = 5
    LB = 6
    D = 7


direction_dict = {Direction.RB: {"change_x": 1, "change_y": 1},  # right bottom corner
                  Direction.R: {"change_x": 0, "change_y": 1},  # right from center
                  Direction.RT: {"change_x": -1, "change_y": 1},  # right top corner
                  Direction.T: {"change_x": -1, "change_y": 0},  # top from center
                  Direction.LT: {"change_x": -1, "change_y": -1},  # left top corner
                  Direction.L: {"change_x": 0, "change_y": -1},  # left from center
                  Direction.LB: {"change_x": 1, "change_y": -1},  # left bottom corner
                  Direction.D: {"change_x": 1, "change_y": 0}}  # down from center

