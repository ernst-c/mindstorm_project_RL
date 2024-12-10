"""
todo:
- improve collision detection
- improve load speed
- spawn radius
- localization probably not finished in time, try to get model working with only range finder data, possible other data as well

"""


from EOM.eom import dynamic_model_mindstorm, simple_dynamic_model
from EOM.rk4 import runge_kutta4
import random as r
import numpy as np
import gymnasium as gym
from os import path
from gymnasium.utils import seeding
from gymnasium import spaces
from math import pi, cos, sin, tan
from Reward.rewardfuncs import sparse_reward2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString
import pygame
class mindstormBot(gym.Env):
    metadata = {'render.modes': ['human']}

    # state = [x, z, xdot, zdot, theta], action = [Thrust, Theta_commanded], param = [mass, gain_const, time_const]

    def __init__(self, t_s, goal_state=np.array([0, 3.5, 0, 0, 0], dtype=float),
                 episode_steps=450, rewardfunc=sparse_reward2d, eom=dynamic_model_mindstorm,
                 param=np.array([0.3,0.1]), rk4=runge_kutta4):
        #params: [wheel base, wheel radius]
        self.wheel_base = param[0]
        self.wheel_radius = param[1]

        super(mindstormBot, self).__init__()

        self.viewer = None
        self.episode_steps = episode_steps
        self.param = param
        self.rewardfunc = rewardfunc
        self.EOM = eom
        self.RK4 = rk4
        self.T_s = t_s
        self.Timesteps = 0
        self.goal_state = goal_state

        #range finder
        self.max_range = 5

        # Define the wall's start and end points
        start_point = (-20,0)

        self.wall_length = 0.1
        # Create a LineString representing the wall's center line
        wall_center = LineString([start_point, (start_point[0]+self.wall_length,start_point[1])])

        # Define the wall's thickness (0.2 in this case)
        self.wall_thickness = 0.2

        # Create a polygon representing the wall by buffering the LineString
        self.polygons = [wall_center.buffer(self.wall_thickness / 2, cap_style=2), wall_center.buffer(self.wall_thickness / 2, cap_style=2), wall_center.buffer(self.wall_thickness / 2, cap_style=2)]
        #self.wall_polygon1 = wall_center.buffer(self.wall_thickness / 2, cap_style=2)
        #self.wall_polygon2 = wall_center.buffer(self.wall_thickness / 2, cap_style=2)
        #self.wall_polygon3 = wall_center.buffer(self.wall_thickness / 2, cap_style=2)
        
        #add long y-direction walls as border of field:
        self.polygons.append(LineString([(-3,-5),(-3,5)]).buffer(self.wall_thickness / 2, cap_style=2))
        self.polygons.append(LineString([(3,-5),(3,5)]).buffer(self.wall_thickness / 2, cap_style=2))
        #border behind goal:
        self.polygons.append(LineString([(-3,5),(3,5)]).buffer(self.wall_thickness / 2, cap_style=2))
        # Used for simulations
        self.episode_counter = 0
        self.goal_range = 1
        self.spawn_increment = 1/1500
        self.horizontal_spawn_radius = 0.25
        self.vertical_spawn_radius = 0.25
        #left wheel pwm, right wheel pwm;
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]), dtype=float)
        # States are: [x, z, x_dot. z_dot, Theta, Theta_dot]
        # new states are: [x,y, x_dot, z_dot, alpha]
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -20, -20, -2*pi,0]),
            high=np.array([5, 5, 20, 20, 2*pi,self.max_range]),
            dtype=float
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.agent_pos = []

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))  # Set screen size
        self.clock = pygame.time.Clock()

        self.reset()
        self.seed()
        self.counter = 0

    def wall_center(self,start_point):
        return LineString([start_point, (start_point[0]+self.wall_length,start_point[1])])

    def get_wall_polygon(self,wall_center):
        return wall_center.buffer(self.wall_thickness / 2, cap_style=2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cast_ray(self, robot_position, robot_orientation, polygon):
        # Ensure robot_position is a Shapely Point
        ray_start = Point(robot_position)

        # Use the x and y attributes of the Point for calculation
        ray_end = Point(
            ray_start.x + self.max_range * np.sin(robot_orientation),
            ray_start.y + self.max_range * np.cos(robot_orientation)
        )

        # Create the ray as a LineString
        ray = LineString([ray_start, ray_end])

        # Find intersections with the wall polygon
        intersection = ray.intersection(polygon)

        if intersection.is_empty:
            return self.max_range  # No intersection, return max range

        # If there's an intersection, handle it
        if isinstance(intersection, Point):
            return ray_start.distance(intersection)
        elif isinstance(intersection, LineString):
            # For a LineString, use the closest endpoint
            distances = [ray_start.distance(Point(pt)) for pt in intersection.coords]
            return min(distances)
        elif intersection.geom_type == 'MultiPoint':
            # Multiple intersection points, take the closest
            return min(ray_start.distance(point) for point in intersection.geoms)

        # Default fallback, no valid intersection
        return self.max_range

    def get_single_range_finder_reading(self, polygon):
        """
        Returns the distance to the nearest obstacle in the direction the robot is facing.
        """
        robot_position = (self.agent_pos[0], self.agent_pos[1])
        robot_orientation = self.agent_pos[4] #Robot's heading
        return self.cast_ray(robot_position, robot_orientation, polygon)

    def step(self, action):
        self.real_action = np.array([action[0], action[1]], dtype=float)
        #fix orientation value issues for training
        self.agent_pos[4] = (self.agent_pos[4] + np.pi) % (2 * np.pi) - np.pi
        self.agent_pos = self.agent_pos + self.EOM(self.agent_pos, self.real_action, self.param)#self.RK4(self.agent_pos, self.real_action, self.EOM, self.T_s)
        self.agent_pos = np.clip(self.agent_pos, self.observation_space.low, self.observation_space.high)

        for polygon in self.polygons:
            distance_of_object = self.get_single_range_finder_reading(polygon)
            if distance_of_object < self.agent_pos[5]:
                self.agent_pos[5] = distance_of_object
        
        observation = self.agent_pos
        reward, terminated = self.rewardfunc(observation, self.goal_state, self.observation_space, self.goal_range, self.polygons, self.wheel_base, self.wheel_radius)
        point = Point(self.agent_pos[0], self.agent_pos[1])
        self.counter += 1
        self.Timesteps += 1
        truncated = False
        if self.counter == self.episode_steps:
            truncated = True

        info = {}

        return observation, reward, terminated,truncated, info

    def reset(self, seed=None):

        self.episode_counter += 1
        """
        # Start episodes within a box around the goal state
        self.agent_pos = np.array([-0.5,
                                  -4,
                                   0, 0, pi/16,self.max_range], dtype=np.float32)
        """
        self.agent_pos = np.array([r.uniform(self.goal_state[0]-self.horizontal_spawn_radius,self.goal_state[0]+self.horizontal_spawn_radius),
                                   r.uniform(self.goal_state[1]-self.vertical_spawn_radius,self.goal_state[1]),
                                   0, 0, 0, self.max_range], dtype=float)

        while any(polygon.contains(Point(self.agent_pos[0], self.agent_pos[1])) for polygon in self.polygons):
            self.agent_pos = np.array(
                [np.clip(r.uniform(self.goal_state[0] - self.horizontal_spawn_radius,
                                self.goal_state[0] + self.horizontal_spawn_radius),
                        self.observation_space.low[0], self.observation_space.high[0]),
                np.clip(r.uniform(self.goal_state[1] - self.vertical_spawn_radius,
                                self.goal_state[1]),
                        self.observation_space.low[1], self.observation_space.high[1]),
                0, 0, 0, self.max_range],
                dtype=float)

        """
        while self.landing_polygon.contains(Point(self.agent_pos[0], self.agent_pos[1])):
            self.agent_pos = np.array([np.clip(r.uniform(self.goal_state[0] - self.horizontal_spawn_radius,
                                                 -1),
                                               self.observation_space.low[0], self.observation_space.high[0]),
                                       np.clip(r.uniform(self.goal_state[1] - self.vertical_spawn_radius,
                                                 self.goal_state[1] + self.vertical_spawn_radius),
                                               self.observation_space.low[1]+0.5, self.observation_space.high[1]), 0, 0, 0],
                                      dtype=np.float32)
        """
        # Spawn Radius Increase
        if self.horizontal_spawn_radius <= 2:
            self.horizontal_spawn_radius += self.spawn_increment
        if self.vertical_spawn_radius <= 7:
            self.vertical_spawn_radius += self.spawn_increment
        if self.wall_length < 3:
            self.wall_length += self.spawn_increment #self.spawn_increment
        # Gradually decrease the goal threshold
        #if 7500 >= self.episode_counter >= 2500 and self.goal_range >= 0.1:
        #    self.goal_range -= 0.15/5000
        if self.episode_counter > 1000:
            start_point = (-3, 1.5)
            self.polygons[0] = self.get_wall_polygon(self.wall_center(start_point))
        if self.episode_counter > 1750:
            start_point = (0,-0.5)
            self.polygons[1] = self.get_wall_polygon(self.wall_center(start_point))
        if self.episode_counter > 2500:
            start_point = (-3,-2)
            self.polygons[2] = self.get_wall_polygon(self.wall_center(start_point))
        
        # Clip position to be in the bounds of the field
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.observation_space.low[0] + 0.1,
                                        self.observation_space.high[0] - 0.1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.observation_space.low[1] + 1,
                                        self.observation_space.high[1] - 0.1)
        self.counter = 0
        info = {}

        return self.agent_pos, info

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))  # Fill screen with white color

        # Chassis (rectangle)
        l, r, t, b = -self.wheel_base / 2, self.wheel_base / 2, self.wheel_radius, -self.wheel_radius
        chassis_rect = pygame.Rect(self.agent_pos[0] * 100 + 400, self.agent_pos[1] * 100 + 400, self.wheel_base * 100, self.wheel_radius * 200)
        pygame.draw.rect(self.screen, (150, 150, 200), chassis_rect)  # Draw chassis

        # Wheels (rectangles)
        wheel_offset = self.wheel_base / 2
        wheel_left_pos = (self.agent_pos[0] + wheel_offset * np.cos(self.agent_pos[4])) * 100 + 400, \
                         (self.agent_pos[1] - wheel_offset * np.sin(self.agent_pos[4])) * 100 + 400
        wheel_right_pos = (self.agent_pos[0] - wheel_offset * np.cos(self.agent_pos[4])) * 100 + 400, \
                          (self.agent_pos[1] + wheel_offset * np.sin(self.agent_pos[4])) * 100 + 400
        pygame.draw.circle(self.screen, (0, 0, 0), (int(wheel_left_pos[0]), int(wheel_left_pos[1])), int(self.wheel_radius * 50))
        pygame.draw.circle(self.screen, (0, 0, 0), (int(wheel_right_pos[0]), int(wheel_right_pos[1])), int(self.wheel_radius * 50))

        # Goal (circle)
        goal_pos = (self.goal_state[0] * 100 + 400, self.goal_state[1] * 100 + 400)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(goal_pos[0]), int(goal_pos[1])), 20)

        # Add walls (polygon)
        for polygon in self.polygons:
            for i in range(len(polygon.exterior.coords) - 1):
                start_point = polygon.exterior.coords[i]
                end_point = polygon.exterior.coords[i + 1]
                pygame.draw.line(self.screen, (200, 50, 50),
                                 (int(start_point[0] * 100 + 400), int(start_point[1] * 100 + 400)),
                                 (int(end_point[0] * 100 + 400), int(end_point[1] * 100 + 400)), 2)

        # Update the display
        pygame.display.flip()

        if mode == 'human':
            self.clock.tick(60)  # Control the frame rate
            return None
        elif mode == 'rgb_array':
            # Get the screen as a numpy array (return image)
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




