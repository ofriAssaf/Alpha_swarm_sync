import threading
import time
from time import process_time_ns
from abc import ABC
from collections import deque
from random import randint
import numpy as np
from parameters import vals
from grid_world import get_grid
from utilities import grid_move_checker, direction_dict, Direction
from Enviorment import Environment

global robot_list
robot_list = []
global starting_time
starting_time = time.time()

class RobotBase(ABC):
    def __init__(self, environment_obj, robot_id: int, grid_size=vals.grid_size):
        self.x = 0
        self.y = 0
        robot_list.append(self)
        self.simulation_time = []
        self.accuracy = []
        self.grid_size = grid_size
        self.grid = environment_obj.grid
        self.environment_obj = environment_obj
        self.robot_id = robot_id
        self.phase = 0
        self.mine_done = deque()
        mine_locations = np.where(self.grid == vals.mine_value)
        self.num_of_mines = mine_locations[0].shape[0]

    # lab code
    def robots_communication_range(self, robot_type): # returns an array of the robots in range including self
        InRangeRobotsList = []
        for i in range(len(robot_list)):
            pos1x = robot_list[i].x + vals.starting_pos[0]
            pos2x = self.get_pos_on_grid()[0]
            dis_x = abs(pos1x - pos2x)
            pos1y = robot_list[i].y + vals.starting_pos[1]
            pos2y = self.get_pos_on_grid()[1]
            dis_y = abs(pos1y - pos2y)
            if dis_x <= self.get_view_range(robot_type) // 2 and dis_y <= self.get_view_range(robot_type) // 2:
                InRangeRobotsList.append(robot_list[i])

        return InRangeRobotsList

    # lab code
    def get_view_range(self, robot_type): # returns view range according to robot type and height
        if robot_type == 1:
            if self.grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 1 or self.grid[
                self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 2:
                return 3
            if self.grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 3 or self.grid[
                self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 4:
                return 5
            if self.grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 5:
                return 7
        elif robot_type == 2 or robot_type == 3:
            if self.grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 1 or self.grid[
                self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 2 or self.grid[
                self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 3:
                return 3
            if self.grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 4 or self.grid[
                self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == 5:
                return 5
        return 5

    # lab code
    def get_pos_on_grid(self): # return actual location of robot in grid
        x = np.clip(self.x + vals.starting_pos[0], 0, vals.grid_size - 1)
        y = np.clip(self.y + vals.starting_pos[1], 0, vals.grid_size - 1)
        return [x, y]

    def closest_mine(self,x, y, tasks_left): # returns the closest mine to location [x,y]
        min_distance = abs(x - tasks_left[0][0])  + abs(y - tasks_left[0][1])
        closest = tasks_left[0]
        for loc in tasks_left:
            dis = abs(x - loc[0]) + abs(y - loc[1])
            if min_distance > dis:
                min_distance = dis
                closest = loc
        return closest

class Navigator(RobotBase):
    def __init__(self, environment_obj, robot_id):
        self.robot_type = 1
        super().__init__(environment_obj, robot_id)
        self.x = 0
        self.y = 0 # relative to [0,0]:
        self.mine_found = deque()  # mine founds by the robots
        self.mine_done = deque()  # mine found and finished by the digger and shoveler
        self.mine_F_not_D = deque()  # mine found by the navigator and aren't done by the diggers (as far as the navigater knows)
        self.time_mine_done = []
        self.time_mine_found = []
        self.stamp = time.time()
        self.comm_mines_not_D_To_Digger = deque() # in order for the digger and navigator to make the same decisions they need to have the same reference point to the distance
        self.visit_order = deque()
        self.visit_order.append([0,0])
        self.comm_mines_not_D_To_Shov = deque()

    def one_step(self, direction, lock): # updates a single robot location values in a certain direction
        if lock.acquire():
            # relative location
            self.x += direction_dict[direction]["change_x"]
            self.y += direction_dict[direction]["change_y"]
            lock.release()
        # actual location in grid
        self.environment_obj.set_robot_location(direction_dict[direction]["change_x"], self.robot_id, 0)
        self.environment_obj.set_robot_location(direction_dict[direction]["change_y"], self.robot_id, 1)

    def find_next_mine(self, lock): # phase 2 - Navigator explores the grid until finding a new mine
        direction1 = randint(0, 7)
        while (True):
            direction1 = randint(0, 7)
            if [self.x, self.y] not in self.mine_found:
                if self.grid[self.environment_obj.robots_location[self.robot_id][0],
                             self.environment_obj.robots_location[self.robot_id][1]] == vals.mine_value:
                    break

            if grid_move_checker(self.grid, self.environment_obj.robots_location[self.robot_id][0],
                                 self.environment_obj.robots_location[self.robot_id][1],
                                 self.environment_obj.robots_location[self.robot_id][0] +
                                 direction_dict[direction1]["change_x"],
                                 self.environment_obj.robots_location[self.robot_id][1] +
                                 direction_dict[direction1][
                                     "change_y"]):
                self.one_step(direction1, lock)

        if len(self.mine_found) < vals.num_mines:
            self.mine_found.append([self.x, self.y])
            self.mine_F_not_D.append([self.x, self.y])
            self.time_mine_found.append(time.time() - self.stamp)


    def communicate_D_S(self, InRange: list): # Navigator communicate new mines found to digger and shoveler
        if len(InRange) > 1:
            for robot in InRange:
                if robot.robot_type == 2:
                    for place in self.mine_found:
                        if place not in robot.taskFound:
                            robot.taskLocationsQueue.append(place)
                            robot.taskFound.append(place)
                            self.comm_mines_not_D_To_Digger.append(place)
                if robot.robot_type == 3: # the shoveler will make the same decisions as the shoveler and will go to same mines
                    for place in self.comm_mines_not_D_To_Digger:
                        if place not in robot.taskFound:
                            robot.taskLocationsQueue.append(place)
                            robot.taskFound.append(place)
                            self.comm_mines_not_D_To_Shov.append(place)

    def reach_last_mine(self, lock):  # reach the last mine found to check if the digger and shoveler are there
        self.phase = 2  # communicate information
        if len(self.mine_F_not_D) <= 1 and len(self.mine_done) == 0: # No mines were found yet - return to starting point
            mine = [0, 0]
        else:
            mine = self.closest_mine(self.visit_order[-1][0], self.visit_order[-1][1], self.comm_mines_not_D_To_Shov)
        gap_x = abs(mine[0] - self.x)
        if (gap_x == 0):
            gap_x += 1
        for i in range(gap_x):
            for j in range(abs(mine[1] - self.y)):
                if mine[1] - self.y > 0:
                    self.one_step(Direction.R, lock)
                else:
                    self.one_step(Direction.L, lock)
            if (mine[0] - self.x < 0):
                self.one_step(Direction.T, lock)
            elif mine[0] - self.x > 0:
                self.one_step(Direction.D, lock)
        if (self.grid[
            self.environment_obj.robots_location[self.robot_id][0], self.environment_obj.robots_location[self.robot_id][
                1]] == vals.mine_val_finished):
            self.mine_done.append([self.x, self.y])
            self.mine_F_not_D.remove([self.x, self.y])  # mine is done
            self.comm_mines_not_D_To_Digger.remove([self.x, self.y])
            self.comm_mines_not_D_To_Shov.remove([self.x, self.y])
            self.time_mine_done.append(time.time() - self.stamp)
            self.visit_order.append([self.x, self.y])


    def run_all_navigator(self, lock): # performs the Navigator's behavior incorporating all phases
        print(self.num_of_mines)
        self.phase = 1
        staticT = time.time()
        start = process_time_ns()
        while len(self.mine_found) != self.num_of_mines:
            if self.phase == 1:
                if len(self.mine_found) != self.num_of_mines:
                    self.find_next_mine(lock)
                    self.phase = 2
                else:
                    self.phase = 2
            if self.phase == 2:
                self.reach_last_mine(lock)
                if len(self.robots_communication_range(self.robot_type)) > 1:
                    self.phase = 3
                else:
                    self.phase = 2
            if self.phase == 3: #communicate
                InRange = self.robots_communication_range(self.robot_type)
                InRange.sort(key=lambda x: x.robot_type)
                for robot in InRange:
                    if robot.robot_type != 1:
                        if robot.phase == 3 and robot.robot_type != 1:
                            break
                        else:
                            self.communicate_D_S(InRange)
                self.phase = 1

        while len(self.mine_F_not_D) != 0: # found all mines
            if len(self.robots_communication_range(self.robot_type)) > 1:
                InRange = self.robots_communication_range(self.robot_type)
                InRange.sort(key=lambda x: x.robot_type)
                for robot in InRange:

                    if (robot.phase != 3 and robot.robot_type != 1): # the robots didn't finish all their tasks
                        self.communicate_D_S(InRange)
                    elif robot.robot_type != 1:
                        if len(self.comm_mines_not_D_To_Digger) > 0:
                            self.reach_last_mine(lock)
                        elif len(self.mine_F_not_D) == 0:
                            break
            else:
                self.reach_last_mine(lock)

        print(self.time_mine_done, "time mine done")
        print(self.time_mine_found, "time mine found")
        print("!!!!!!!!!!!!!!done!!!!!!!!!!!")
        print("Simulation Duration: ", process_time_ns() - start)
        self.finished_bool = True


class Shoveler(RobotBase): # lab code
    def __init__(self, environment_obj, robot_id):
        self.robot_type = 3
        super().__init__(environment_obj, self.robot_type, robot_id)
        self.x = 0
        self.y = 0
        self.taskLocationsQueue = deque()
        self.shoveler_mine_done = 0
        self.taskFound = deque()
        self.count = 0

    # lab code
    def state_idle(self): # no new tasks
        if len(self.taskLocationsQueue) > 0:
            self.phase = 1
        else:
            pass

    # lab code
    def state_finish(self): # finished all tasks
        pass

    # lab code
    def state_move_to_mine(self, x, y, lock):  # move to next mine
        if self.x - x > 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0] - 1, self.get_pos_on_grid()[1]):
                if lock.acquire():
                    self.x -= 1
                    lock.release()
                self.pos = [self.x, self.y]
        elif self.x - x < 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0] + 1, self.get_pos_on_grid()[1]):
                if lock.acquire():
                    self.x += 1
                    lock.release()
                self.pos = [self.x, self.y]
        elif self.y - y > 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0], self.get_pos_on_grid()[1] - 1):
                if lock.acquire():
                    self.y -= 1
                    lock.release()
                self.pos = [self.x, self.y]
        elif self.y - y < 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0], self.get_pos_on_grid()[1] + 1):
                if lock.acquire():
                    self.y += 1
                    lock.release()
                self.pos = [self.x, self.y]

        if self.y == y and self.x == x:
            self.phase = 2

    # lab code
    def state_shovel(self): # perform shoveling task at mine
        taskTime = np.random.normal(vals.time_to_shovel, 0.3)
        tStatic = time.time()
        if self.grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] == vals.mine_val_digger_finished:
            while time.time() < tStatic + taskTime:
                time.sleep(1)
            else:
                self.taskLocationsQueue.remove(self.closest_mine(self.x, self.y, self.taskLocationsQueue))
                self.grid[self.x + vals.starting_pos[0], self.y + vals.starting_pos[1]] = 0
                self.phase = 0
                self.count += 1
        else:
            pass

    # lab code
    def run_all_shoveler(self, lock): # perform Shoveler's behavior
        while len(navigator_1.mine_done) != self.num_of_mines:
            if self.count == self.num_of_mines:
                self.phase = 3
                self.state_finish()
            if self.phase == 0:
                self.state_idle()
            elif self.phase == 1:
                self.state_move_to_mine(self.closest_mine(self.x, self.y, self.taskLocationsQueue)[0], self.closest_mine(self.x, self.y, self.taskLocationsQueue)[1], lock=lock)
            elif self.phase == 2:
                self.state_shovel()


class Digger(RobotBase): # lab code
    def __init__(self, environment_obj, robot_id):
        self.robot_type = 2
        super().__init__(environment_obj, self.robot_type, robot_id)
        self.x = 0
        self.y = 0
        self.taskLocationsQueue = deque()
        self.taskFound = deque()
        self.count = 0

    # lab code
    def state_idle(self): # no new tasks
        if len(self.taskLocationsQueue) > 0:
            self.phase = 1
        else:
            pass

    # lab code
    def state_finish(self): # finished all tasks
            pass

    # lab code
    def state_move_to_mine(self, x, y, lock): # move to next mine
        if self.x - x > 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0] - 1, self.get_pos_on_grid()[1]):
                if lock.acquire():
                    self.x -= 1
                    lock.release()
                self.pos = [self.x, self.y]
        elif self.x - x < 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0] + 1, self.get_pos_on_grid()[1]):
                if lock.acquire():
                    self.x += 1
                    lock.release()
                self.pos = [self.x, self.y]
        elif self.y - y > 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0], self.get_pos_on_grid()[1] - 1):
                if lock.acquire():
                    self.y -= 1
                    lock.release()
                self.pos = [self.x, self.y]
        elif self.y - y < 0:
            if grid_move_checker(self.grid, self.get_pos_on_grid()[0], self.get_pos_on_grid()[1],
                                 self.get_pos_on_grid()[0], self.get_pos_on_grid()[1] + 1):
                if lock.acquire():
                    self.y += 1
                    lock.release()
                self.pos = [self.x, self.y]

        if self.y == y and self.x == x:
            self.phase = 2

    # lab code
    def state_dig(self): # perform digging task at mine
        taskTime = np.random.normal(vals.time_to_dig, 0.3)
        tStatic = time.time()
        while time.time() < tStatic + taskTime:
            time.sleep(1)
            # pass
        else:
            if time.time() >= tStatic + taskTime:
                self.communicate_finished_digging()
                if len(self.taskLocationsQueue) != 0:
                    self.taskLocationsQueue.remove(self.closest_mine(self.x, self.y, self.taskLocationsQueue))
                    self.count += 1
                self.phase = 0

    # lab code
    def communicate_finished_digging(self): # communicate to shoveler done digging at mine
        InRangeRobotList = self.robots_communication_range(self.robot_type)
        for j in range(len(robot_list)):
            if robot_list[j].robot_type == 3:
                robot_list[j].grid[self.get_pos_on_grid()[0], self.get_pos_on_grid()[1]] = vals.mine_val_digger_finished

        if len(InRangeRobotList) != 0:
            for i in range(len(InRangeRobotList)):
                if InRangeRobotList[i].robot_type == 3:
                    if self.closest_mine(self.x, self.y, self.taskLocationsQueue) not in self.taskFound:
                        InRangeRobotList[i].taskLocationsQueue.append(self.closest_mine(self.x, self.y, self.taskLocationsQueue))
                        InRangeRobotList[i].taskFound.append(self.closest_mine(self.x, self.y, self.taskLocationsQueue))
        else:
            pass

    # lab code
    def run_all_digger(self, lock): # perform Digger's behavior
        tStatic = time.time()
        while len(navigator_1.mine_done) != self.num_of_mines:
            if self.count == self.num_of_mines:
                self.phase = 3
                self.state_finish()
            if self.phase == 0:
                self.state_idle()
            elif self.phase == 1:
                self.state_move_to_mine(self.closest_mine(self.x, self.y, self.taskLocationsQueue)[0], self.closest_mine(self.x, self.y, self.taskLocationsQueue)[1], lock=lock)
            elif self.phase == 2:
                self.state_dig()


if __name__ == "__main__":
    grid = get_grid(vals.grid_size, vals.pad_width, vals.N)
    numberOfRobots = 5
    finish = False
    robots_location = np.full((numberOfRobots, 2), vals.starting_pos)
    environment1 = Environment(grid, robots_location, numberOfRobots, finish)
    starting_time = time.time()

    shoveler_1 = Shoveler(environment1, 4)
    navigator_1 = Navigator(environment1, 0)
    digger_1 = Digger(environment1, 2)

    lock = threading.Lock()
    t1 = threading.Thread(target=navigator_1.run_all_navigator, args=[lock])
    t3 = threading.Thread(target=digger_1.run_all_digger, args=[lock])
    t5 = threading.Thread(target=shoveler_1.run_all_shoveler, args=[lock])

    t1.start()
    t3.start()
    t5.start()

