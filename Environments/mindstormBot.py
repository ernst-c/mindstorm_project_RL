"""
todo:
- check if improved dynamic model with linear velocity actually works
- check if gamma should be changed
- check if subprocenv actually matters
- check if the reward function is correct, maybe distance based is better

"""


from EOM.eom import *
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
from shapely.strtree import STRtree

class mindstormBotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    # state = [x, z, xdot, zdot, theta], action = [Thrust, Theta_commanded], param = [mass, gain_const, time_const]

    def __init__(self, t_s=1/20, goal_state=np.array([0, 1.45, 0, 0, 0], dtype=float),
                 episode_steps=600, rewardfunc=sparse_reward2d, eom=discrete_model,
                 param=np.array([0.3,0.1]), rk4=runge_kutta4,render_mode=None):
        #params: [wheel base, wheel radius]
        self.wheel_base = param[0]
        self.wheel_radius = param[1]

        #rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_width = 200
        self.window_height = 270
        #super(mindstormBotEnv, self).__init__()

        self.episode_steps = episode_steps
        self.param = param
        self.rewardfunc = rewardfunc
        self.EOM = eom
        self.RK4 = rk4
        self.T_s = t_s
        self.Timesteps = 0
        self.goal_state = goal_state

        #range finder
        self.max_range = 1

        # Define the wall's length
        self.wall_length = 0.1

        # Define the wall's thickness (0.2 in this case)
        self.horizontal_spawn_radius = 0.1
        self.vertical_spawn_radius = 0.1
                
        #add long y-direction walls as border of field:
        
        # Used for simulations
        self.episode_counter = 0
        self.spawn_increment = 1/750
        if eom == discrete_model:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]), dtype=float)        
        # States are: [x, z, x_dot. z_dot, Theta, Theta_dot]
        # new states are: [x,y, x_dot, z_dot, alpha]
        self.observation_space = spaces.Box(
            low=np.array([-0.8, 0, -2*pi,0]),
            high=np.array([0.8, 2.5, 2*pi,self.max_range]),
            dtype=float
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.agent_pos = [0,0,0,0]

        #dynamic model
        if eom == linear_velocity_steering_angle:
            self.linear_velocity = 0
        elif eom == dynamic_model_mindstorm:
            self.wheel_velocity = [0,0]
        # optimization
        self.goal_range = 0.15
        #self.polygons = self.create_simple_map()
        self.polygons = self.create_large_map(0)
        self.spatial_index = STRtree(self.polygons)
        self.counter = 0
        #rendering
        self.ray = LineString([(0,0),(0,0)])
        #collision
        self.collision_range = 0.1
        self.reset()
        self.seed()

    def get_wall_line(self, start_point):
        return LineString([start_point, (start_point[0] + self.wall_length, start_point[1])])

    def create_simple_map(self, random=False):
        polygons = [0,0,0]
        if not random:
            start_point = (-3, 4)   
            polygons[0] = self.get_wall_line(start_point)
            start_point = (0, 2)
            polygons[1] = self.get_wall_line(start_point)
            start_point = (-3,0.5)
            polygons[2] = self.get_wall_line(start_point)
        else:
            polygons[0] = self.get_wall_line((r.choice([-0.4,0]),0.4))
            polygons[1] = self.get_wall_line((r.choice([-0.4,0]),0.8))
            polygons[2] = self.get_wall_line((r.choice([-0.4,0]),1.2))

        polygons.append(LineString([(-0.4,0),(-0.4,1.6)]))
        polygons.append(LineString([(0.4,0),(0.4,1.6)]))
        polygons.append(LineString([(-0.4,1.6),(0.4,1.6)]))

        return polygons
    
    def create_large_map(self, episode_counter):
        self.goal_state = np.array([-0.4, 0.35, 0, 0, 0], dtype=float) 
        polygons = [0,0,0,0,0,0,0]
        ###add border walls of field to polygons at x=-1 and x=1 and vertically to from y=0 to y=2.5
        polygons.append(LineString([(-0.8,0),(-0.8,2.5)]))
        polygons.append(LineString([(0.8,0),(0.8,2.5)]))
        polygons.append(LineString([(-0.8,0),(0.8,0)]))
        polygons.append(LineString([(-0.8,2.5),(0.8,2.5)]))
        ###after episode_counter = 500, add small wall in the middle of field which increases in length until episode counter = 1000
        if episode_counter > 500:
            wall_length = min(0.1 + (episode_counter-500)*0.001,1.5)
            polygons[6] = (LineString([(0,0.5),(0,0.5+wall_length)]))
        else:
            polygons[6] = (LineString([(0,0.5),(0,0.51)]))
        ###add randomized horizontal obstacle walls:
        polygons[0] = (LineString([(-0.8,0.6),(0,0.6)]))
        polygons[1] = self.get_wall_line((r.choice([-0.8,-0.4]),1.2))   
        polygons[2] = self.get_wall_line((r.choice([-0.8,-0.4]),1.8))
        polygons[3] = self.get_wall_line((r.choice([0,0.4]),0.6))
        polygons[4] = self.get_wall_line((r.choice([0,0.4]),1.2))
        polygons[5] = self.get_wall_line((r.choice([0,0.4]),1.8))

        return polygons

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def ray_caster(self):
        # Create a LineString representing the robot's position and orientation
        robot_position = (self.agent_pos[0], self.agent_pos[1])
        robot_orientation = self.agent_pos[2]
        ray_start = Point(robot_position)
        ray_end = Point(
            ray_start.x + self.max_range * np.sin(robot_orientation),
            ray_start.y + self.max_range * np.cos(robot_orientation)
        )
        ray = LineString([ray_start, ray_end])
        return ray

    def cast_ray(self, polygon ,ray):

        # Find intersections with the wall polygon
        intersection = ray.intersection(polygon)

        if intersection.is_empty:
            return self.max_range  # No intersection, return max range

        if isinstance(intersection, LineString):
            # For LineString intersections, find the closest endpoint
            ray_start = ray.coords[0]  # Starting point of the ray
            distances = [Point(ray_start).distance(Point(pt)) for pt in intersection.coords]
            return min(distances)

        if isinstance(intersection, Point):
            # Direct point intersection
            ray_start = ray.coords[0]  # Starting point of the ray
            return Point(ray_start).distance(intersection)

        # Default fallback, no valid intersection
        return self.max_range

    def step(self, action):
        #self.real_action = np.array([action[0], action[1]], dtype=float)
        #fix orientation value issues for training
        self.agent_pos[2] = (self.agent_pos[2] + np.pi) % (2 * np.pi) - np.pi
        #movement = self.EOM(self.agent_pos, self.real_action, self.param, self.wheel_velocity, self.T_s)
        #movement = self.EOM(self.agent_pos, self.real_action, self.param, self.linear_velocity, self.T_s)
        movement = self.EOM(self.agent_pos, action)
        ray = self.ray_caster()
        #set laser detected range to closest detected object
        collision = False   
        
        if (self.spatial_index.query_nearest(Point(self.agent_pos[0], self.agent_pos[1]), return_distance=True)[1][0] < self.collision_range):
            collision = True
            
        self.agent_pos[3] = self.max_range
        query_result = self.spatial_index.query(ray, predicate='intersects')
        if len(query_result) > 0:
            for i in query_result:
                closest_object = self.polygons[int(i)]
                closest_intersection = self.cast_ray(closest_object, ray)
                if closest_intersection < self.agent_pos[3]:
                    self.agent_pos[3] = closest_intersection
        self.ray = LineString([ray.coords[0], (ray.coords[0][0] + self.agent_pos[3] * np.sin(self.agent_pos[2]),
                                                ray.coords[0][1] + self.agent_pos[3] * np.cos(self.agent_pos[2]))])

        self.agent_pos[0] += movement[0]
        self.agent_pos[1] += movement[1]
        self.agent_pos[2] += movement[2]
        if self.EOM == dynamic_model_mindstorm:
            self.wheel_velocity[0] = movement[4]
            self.wheel_velocity[1] = movement[5]
        elif self.EOM == linear_velocity_steering_angle:
            self.linear_velocity = movement[4]
        self.agent_pos = np.clip(self.agent_pos, self.observation_space.low, self.observation_space.high)

        observation = self.agent_pos
        reward, terminated = self.rewardfunc(observation, self.goal_state, self.observation_space, self.goal_range, collision)
        self.counter += 1
        self.Timesteps += 1
        truncated = False
        if self.counter == self.episode_steps:
            truncated = True

        info = {}

        return observation, reward, terminated,truncated, info

    def reset(self, seed=None, options=None):

        self.episode_counter += 1
        """
        self.agent_pos = np.array([r.uniform(self.goal_state[0]-self.horizontal_spawn_radius,self.goal_state[0]+self.horizontal_spawn_radius),
                                   r.uniform(self.goal_state[1]-self.vertical_spawn_radius,self.goal_state[1]+self.vertical_spawn_radius),
                                   0, self.max_range], dtype=float)
        """
        self.agent_pos = np.array([r.uniform(self.goal_state[0]-self.horizontal_spawn_radius,self.goal_state[0]+self.horizontal_spawn_radius),
                            r.uniform(self.goal_state[1],self.goal_state[1]+self.vertical_spawn_radius),
                            0, self.max_range], dtype=float)

        # Spawn Radius Increase
        if self.horizontal_spawn_radius <= 0.8:
            self.horizontal_spawn_radius += self.spawn_increment
        if self.vertical_spawn_radius <= 2.5:
            self.vertical_spawn_radius += self.spawn_increment
        if self.wall_length < 0.4:
            self.wall_length += self.spawn_increment #self.spawn_increment

        #self.polygons = self.create_simple_map(random=True)
        self.polygons = self.create_large_map(self.episode_counter)

        self.spatial_index = STRtree(self.polygons)
        ##rewrite for more efficiency and buffer for objects
        while(self.spatial_index.query_nearest(Point(self.agent_pos[0], self.agent_pos[1]), return_distance=True)[1][0] < self.collision_range):
            self.agent_pos = np.array(
                [np.clip(r.uniform(self.goal_state[0] - self.horizontal_spawn_radius,
                                self.goal_state[0] + self.horizontal_spawn_radius),
                        self.observation_space.low[0], self.observation_space.high[0]),
                np.clip(r.uniform(self.goal_state[1] - self.vertical_spawn_radius,
                                self.goal_state[1]),
                        self.observation_space.low[1], self.observation_space.high[1]),
                0, self.max_range],
                dtype=float)
        """
        while any(polygon.intersects(Point(self.agent_pos[0], self.agent_pos[1])) for polygon in self.polygons):
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
        #dynamic model update
        if self.EOM == dynamic_model_mindstorm:
            self.wheel_velocity = [0,0]
        elif self.EOM == linear_velocity_steering_angle:
            self.linear_velocity = 0
        # Clip position to be in the bounds of the field
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.observation_space.low[0],
                                        self.observation_space.high[0])
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.observation_space.low[1],
                                        self.observation_space.high[1])
        self.counter = 0
        info = {}

        return self.agent_pos, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))  # Fill screen with white color

        robot_pos = (self.agent_pos[0] * 100+self.window_width/2, self.agent_pos[1] * 100)
        pygame.draw.circle(canvas, (255, 0, 0), (int(robot_pos[0]), int(robot_pos[1])), int(self.wheel_radius*100))

        # Goal (circle)
        goal_pos = (self.goal_state[0] * 100+self.window_width/2, self.goal_state[1] * 100)
        pygame.draw.circle(canvas, (0, 255, 0), (int(goal_pos[0]), int(goal_pos[1])), int(self.goal_range*100))

        # Add walls (polygon)
        for polygon in self.polygons:
            start_point = polygon.coords[0]
            end_point = polygon.coords[1]
            pygame.draw.line(canvas, (200, 50, 50),
                                (int(start_point[0] * 100)+self.window_width/2, int(start_point[1] * 100)),
                                (int(end_point[0] * 100)+self.window_width/2, int(end_point[1] * 100)), 2)
        
        #ray
        pygame.draw.line(canvas, (0, 0, 0), (int(self.ray.coords[0][0] * 100)+self.window_width/2, int(self.ray.coords[0][1] * 100)),
                         (int(self.ray.coords[1][0] * 100)+self.window_width/2, int(self.ray.coords[1][1] * 100)), 1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()




