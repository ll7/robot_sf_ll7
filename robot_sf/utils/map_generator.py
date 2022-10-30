import random
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import sys
import os
import datetime
from typing import List

from shapely.geometry import Polygon

from .poly import PolygonCreationSettings, load_polygon, move_polygon, random_polygon


class MapGenerator:
    def __init__(self, x_width = 20, y_width = 20, n_obstacles = None):
        self.x_margin = [-x_width, x_width]
        self.y_margin = [-y_width, y_width]
        self.obstacles = dict()
        self.n_obstacles = None
        self.obstacles_list = List[Polygon]

    def generate(self, n_obstacles = None):
        self.reset()
        self.n_obstacles = n_obstacles

        if not n_obstacles:
            self.n_obstacles = random.randint(2, 10)

        for _ in range(self.n_obstacles):
            creation_config = PolygonCreationSettings(
                random.randint(3, 8), random.random(), random.random())
            polygon = random_polygon(creation_config)

            move_offset = [random.uniform(self.x_margin[0], self.x_margin[1]),
                           random.uniform(self.y_margin[0], self.y_margin[1])]
            polygon = move_polygon(polygon, move_offset)

            # TODO: think of putting those functions back in
            # polygon = scale_polygon(polygon, random.uniform(2, 5), random.choice([0, 1, None]))
            # polygon = constrain_vertex(polygon, self.x_margin[0], self.x_margin[1], self.y_margin[0], self.y_margin[1])

            self.obstacles_list.append(polygon)

        #Pupulate Obstacles dictionary
        self.prepare_for_serialization()

    # def show(self):
    #     for obstacle in self.obstacles_list:
    #         plt.plot(obstacle.vertex[:,0],obstacle.vertex[:,1])
    #     plt.xlim(self.x_margin[0]-1, self.x_margin[1]+1)
    #     plt.ylim(self.y_margin[0]-1, self.y_margin[1]+1)

    def reset(self):
        self.obstacles_list = []
        self.obstacles = dict()

    # def save(self,filename = None):
    #     if not filename:
    #         i = 0
    #         if not os.path.isdir('maps'):
    #             os.makedirs('maps')

    #         default_name = 'generated_map_'
    #         while True:
    #             if os.path.exists('maps'+'/' + default_name + str(i) + '.json'):
    #                 i += 1
    #             else:
    #                 with open('maps'+'/'+ default_name + str(i) + '.json','w') as f:
    #                     json.dump(self.obstacles,f)
    #                     break

    #     elif filename is not None:
    #         with open(filename,'w') as f:
    #             json.dump(self.obstacles, f)

    def prepare_for_serialization(self):
        self.obstacles['Obstacles'] = dict()
        for i in range(len(self.obstacles_list)):
            self.obstacles['Obstacles']['Obstacle_' + str(i)] = dict()
            self.obstacles['Obstacles']['Obstacle_' + str(i)]['Vertex'] = self.obstacles_list[i].vertex.tolist()
            self.obstacles['Obstacles']['Obstacle_' + str(i)]['Edges'] = self.obstacles_list[i].edges
            self.obstacles['Obstacles']['Obstacle_' + str(i)]['ID'] = i
        self.obstacles['Created'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.obstacles['NumberObstacles'] = len(self.obstacles_list)
        self.obstacles['x_margin'] = self.x_margin
        self.obstacles['y_margin'] = self.y_margin

    # def load(self, path_to_json):
    #     #Reset current object
    #     self.reset()

    #     if os.path.exists(path_to_json):
    #         with open(path_to_json) as f:
    #             tmp_data = json.load(f)

    #         self.n_obstacles = len(tmp_data['Obstacles'].keys())
    #         for key in tmp_data['Obstacles'].keys():
    #             self.obstacles_list.append(load_polygon(tmp_data['Obstacles'][key]['Vertex']))
    #         self.obstacles = tmp_data
    #     else:
    #         raise ValueError('File not found!')

    # def generate_and_show(self, n_obstacles = None):
    #     self.generate(n_obstacles)
    #     self.show()
