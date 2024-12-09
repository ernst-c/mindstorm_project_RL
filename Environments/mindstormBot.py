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
import gym
from os import path
from gym.utils import seeding
from gym import spaces
from math import pi, cos, sin, tan
from Reward.rewardfuncs import sparse_reward2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString

class mindstormBot(gym.Env):
    metadata = {'render.modes': ['human']}

    # state = [x, z, xdot, zdot, theta], action = [Thrust, Theta_commanded], param = [mass, gain_const, time_const]

    def __init__(self, t_s, goal_state=np.array([0, 1, 0, 0, 0], dtype=float),
                 episode_steps=300, rewardfunc=sparse_reward2d, eom=dynamic_model_mindstorm,
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

        self.wall_length = 0
        # Create a LineString representing the wall's center line
        wall_center = LineString([start_point, (start_point[0]+self.wall_length,start_point[1])])

        # Define the wall's thickness (0.2 in this case)
        self.wall_thickness = 0.25

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
        self.goal_range = 0.5
        self.spawn_increment = 1/2000
        self.horizontal_spawn_radius = 0.25
        self.vertical_spawn_radius = 0.25
        #left wheel pwm, right wheel pwm;
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]), dtype=np.float)
        # States are: [x, z, x_dot. z_dot, Theta, Theta_dot]
        # new states are: [x,y, x_dot, z_dot, alpha]
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -5, -5, -2*pi,0]),
            high=np.array([5, 5, 5, 5, 2*pi,self.max_range]),
            dtype=np.float
        )
        self.reward_range = (-float("inf"), float("inf"))
        self.agent_pos = []

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
        reward, done = self.rewardfunc(observation, self.goal_state, self.observation_space, self.goal_range, self.polygons, self.wheel_base, self.wheel_radius)
        point = Point(self.agent_pos[0], self.agent_pos[1])
        self.counter += 1
        self.Timesteps += 1

        if self.counter == self.episode_steps:
            done = True

        info = {}

        return observation, reward, done, info

    def reset(self):

        self.episode_counter += 1
        """
        # Start episodes within a box around the goal state
        self.agent_pos = np.array([-0.5,
                                  -4,
                                   0, 0, pi/16,self.max_range], dtype=np.float32)
        """
        self.agent_pos = np.array([r.uniform(self.goal_state[0]-self.horizontal_spawn_radius,self.goal_state[0]+self.horizontal_spawn_radius),
                                   r.uniform(self.goal_state[1]-self.vertical_spawn_radius,self.goal_state[1]),
                                   0, 0, 0, self.max_range], dtype=np.float32)

        while any(polygon.contains(Point(self.agent_pos[0], self.agent_pos[1])) for polygon in self.polygons):
            self.agent_pos = np.array(
                [np.clip(r.uniform(self.goal_state[0] - self.horizontal_spawn_radius,
                                self.goal_state[0] + self.horizontal_spawn_radius),
                        self.observation_space.low[0], self.observation_space.high[0]),
                np.clip(r.uniform(self.goal_state[1] - self.vertical_spawn_radius,
                                self.goal_state[1]),
                        self.observation_space.low[1], self.observation_space.high[1]),
                0, 0, 0, self.max_range],
                dtype=np.float32)

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
        if self.horizontal_spawn_radius <= 4:
            self.horizontal_spawn_radius += self.spawn_increment
        if self.vertical_spawn_radius <= 4:
            self.vertical_spawn_radius += self.spawn_increment
        if self.wall_length < 1:
            self.wall_length += self.spawn_increment
        # Gradually decrease the goal threshold
        #if 7500 >= self.episode_counter >= 2500 and self.goal_range >= 0.1:
        #    self.goal_range -= 0.15/5000
        if self.episode_counter > 1000:
            start_point = (-0.5,0)
            self.polygons[0] = self.get_wall_polygon(self.wall_center(start_point))
        if self.episode_counter > 2000:
            start_point = (0,-1)
            self.polygons[1] = self.get_wall_polygon(self.wall_center(start_point))
        if self.episode_counter > 3000:
            start_point = (-0.5,-2)
            self.polygons[2] = self.get_wall_polygon(self.wall_center(start_point))
        
        # Clip position to be in the bounds of the field
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.observation_space.low[0] + 0.1,
                                        self.observation_space.high[0] - 0.1)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.observation_space.low[1] + 1,
                                        self.observation_space.high[1] - 0.1)
        self.counter = 0
        return self.agent_pos

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            # Initialize the viewer
            self.viewer = rendering.Viewer(740, 740)
            self.viewer.set_bounds(-5, 5, -5, 5)

            # Chassis (rectangle)
            l, r, t, b = -self.wheel_base / 2, self.wheel_base / 2, self.wheel_radius, -self.wheel_radius
            chassis = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            chassis.set_color(0.6, 0.6, 0.8)
            self.chassis_transform = rendering.Transform()
            chassis.add_attr(self.chassis_transform)

            # Wheels (rectangles)
            wl, wr, wt, wb = -self.wheel_radius / 2, self.wheel_radius / 2, self.wheel_radius, -self.wheel_radius
            wheel_left = rendering.FilledPolygon([(wl, wb), (wl, wt), (wr, wt), (wr, wb)])
            wheel_left.set_color(0.1, 0.1, 0.1)
            self.wheel_left_transform = rendering.Transform()
            wheel_left.add_attr(self.wheel_left_transform)

            wheel_right = rendering.FilledPolygon([(wl, wb), (wl, wt), (wr, wt), (wr, wb)])
            wheel_right.set_color(0.1, 0.1, 0.1)
            self.wheel_right_transform = rendering.Transform()
            wheel_right.add_attr(self.wheel_right_transform)

            # Goal (circle)
            goal = rendering.make_circle(self.goal_range)
            goal.set_color(0.3, 0.8, 0.3)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)

            # Add components to viewer
            self.viewer.add_geom(chassis)
            self.viewer.add_geom(wheel_left)
            self.viewer.add_geom(wheel_right)
            self.viewer.add_geom(goal)

            # Initialize a list to track dynamic geometries
            self.dynamic_geometries = []

        #add wall
        for polygon in self.polygons:
            wall_coords = list(polygon.exterior.coords)
            for i in range(len(wall_coords) - 1):
                start_point = wall_coords[i]
                end_point = wall_coords[i + 1]

                wall_edge = rendering.Line(start_point, end_point)
                wall_edge.set_color(0.8, 0.1, 0.1)

                self.viewer.add_geom(wall_edge)

        self.chassis_transform.set_translation(self.agent_pos[0], self.agent_pos[1])
        self.chassis_transform.set_rotation(-self.agent_pos[4])

        wheel_offset = self.wheel_base / 2
        self.wheel_left_transform.set_translation(
            self.agent_pos[0] + wheel_offset * np.cos(self.agent_pos[4]),
            self.agent_pos[1] - wheel_offset * np.sin(self.agent_pos[4])
        )
        self.wheel_left_transform.set_rotation(-self.agent_pos[4])

        self.wheel_right_transform.set_translation(
            self.agent_pos[0] - wheel_offset * np.cos(self.agent_pos[4]),
            self.agent_pos[1] + wheel_offset * np.sin(self.agent_pos[4])
        )
        self.wheel_right_transform.set_rotation(-self.agent_pos[4])

        self.goal_transform.set_translation(self.goal_state[0], self.goal_state[1])

        for geom in self.dynamic_geometries:
            self.viewer.geoms.remove(geom)
        self.dynamic_geometries.clear()

        points = [
            np.array([self.agent_pos[0], self.agent_pos[1]]),
            np.array([self.agent_pos[0] + self.wheel_base / 2, self.agent_pos[1]]), 
            np.array([self.agent_pos[0] - self.wheel_base / 2, self.agent_pos[1]]), 
            np.array([self.agent_pos[0] + self.wheel_base / 2, self.agent_pos[1] + self.wheel_radius]),
            np.array([self.agent_pos[0] - self.wheel_base / 2, self.agent_pos[1] + self.wheel_radius]), 
            np.array([self.agent_pos[0] + self.wheel_base / 2, self.agent_pos[1] - self.wheel_radius]), 
            np.array([self.agent_pos[0] - self.wheel_base / 2, self.agent_pos[1] - self.wheel_radius])  
        ]

        rotation_matrix = np.array([
            [np.cos(self.agent_pos[4]), np.sin(self.agent_pos[4])],
            [-np.sin(self.agent_pos[4]), np.cos(self.agent_pos[4])]
        ])
        rotated_points = [rotation_matrix.dot(point - np.array([self.agent_pos[0], self.agent_pos[1]])) + np.array([self.agent_pos[0], self.agent_pos[1]])
                          for point in points]

        for idx, point in enumerate(rotated_points):
            x, y = point[0], point[1]
            circle = rendering.make_circle(radius=0.02)
            transform = rendering.Transform(translation=(x, y))
            circle.add_attr(transform)
            circle.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(circle)
            self.dynamic_geometries.append(circle)

        shortest_ray_segment = None
        min_length = float('inf') 

        ray_start = Point(self.agent_pos[0], self.agent_pos[1])
        ray_end = Point(
            ray_start.x + self.max_range * np.sin(self.agent_pos[4]),
            ray_start.y + self.max_range * np.cos(self.agent_pos[4])
        )
        for wall_polygon in self.polygons:
            ray = LineString([ray_start, ray_end])
            intersection = ray.intersection(wall_polygon)

            if not intersection.is_empty:
                if isinstance(intersection, LineString):
                    intersection_point = intersection.coords[0]

                    ray_segment = rendering.Line(
                        (ray_start.x, ray_start.y),
                        (intersection_point[0], intersection_point[1])
                    )
                    ray_segment.set_color(0.0, 1.0, 0.0) 
                    ray_length = ray_start.distance(intersection)
                else:
                    ray_segment = rendering.Line(
                        (ray_start.x, ray_start.y),
                        (ray_end.x, ray_end.y)
                    )
                    ray_segment.set_color(1.0, 0.0, 0.0)
                    ray_length = 5.0
            else:
                ray_segment = rendering.Line(
                    (ray_start.x, ray_start.y),
                    (ray_end.x, ray_end.y)
                )
                ray_segment.set_color(1.0, 0.0, 0.0)
                ray_length = 5.0

            if ray_length < min_length:
                shortest_ray_segment = ray_segment
                min_length = ray_length

        self.viewer.add_geom(shortest_ray_segment)
        self.dynamic_geometries.append(shortest_ray_segment)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




