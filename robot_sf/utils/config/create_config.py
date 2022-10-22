import os
import toml
from datetime import datetime
import numpy as np
#Create Config File


#print(os.path.dirname(__file__))

if __name__ == '__main__':
    
    if not os.path.isdir('../maps'):
        raise ValueError('Cannot locate map files!')
    
    
    map_list = os.listdir('../maps')
    
    if not map_list:
        raise ValueError('maps folder is empty! Please generate and save some maps')
        
    
    tmp = dict()
    tmp['title'] = "Pedestrian Simulator Parameter config"
    tmp['map_files'] = dict()
    tmp['info'] = dict()
    tmp['info']['number_of_maps'] = len(map_list)
    tmp['info']['created'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp['info']['author'] = os.environ['COMPUTERNAME']
    
    
    tmp['simulator'] = dict()
    
    tmp['simulator']['generation'] = dict()
    tmp['simulator']['generation']['actions'] = ['standalone_individual','group','both','none']#
    tmp['simulator']['generation']['probabilities'] = [20,20,10,50]#
    tmp['simulator']['generation']['parameters'] = dict()#
    tmp['simulator']['generation']['parameters']['max_single_peds'] = 3#
    tmp['simulator']['generation']['parameters']['max_group_size'] = 7#
    tmp['simulator']['generation']['parameters']['group_width_max'] = 6#
    tmp['simulator']['generation']['parameters']['group_width_min']  = 2.5#
    tmp['simulator']['generation']['parameters']['average_speed'] = 0.5#
    
    
    tmp['simulator']['group_actions'] = dict()
    tmp['simulator']['group_actions']['actions'] = ['split','merge','group_standalone','none']#
    tmp['simulator']['group_actions']['probabilities'] = [10,10,30,50]#[10,10,30,50]
    tmp['simulator']['group_actions']['parameters'] = dict()
    tmp['simulator']['group_actions']['parameters']['max_standalone_grouping'] = 5#
    
    
    tmp['simulator']['stopping'] = dict()#
    tmp['simulator']['stopping']['actions'] = ['stop_pedestrian','stop_group','move_pedestrian','move_group','none']#
    tmp['simulator']['stopping']['probabilities'] = [10,10,10,10,60]#
    tmp['simulator']['stopping']['parameters'] = dict()#
    tmp['simulator']['stopping']['parameters']['max_nsteps_ped_stopped']  = 100#
    tmp['simulator']['stopping']['parameters']['max_nsteps_group_stopped']  = 100#
    tmp['simulator']['stopping']['parameters']['max_stopping_pedestrians'] = 5#
    tmp['simulator']['stopping']['parameters']['max_unfreezing_pedestrians'] = 5#
    
    
    
    
    tmp['simulator']['flags'] = dict()
    tmp['simulator']['flags']['allow_dynamic_grouping'] = True #
    tmp['simulator']['flags']['allow_max_steps_stop'] = True #
    tmp['simulator']['flags']['rendezvous_centroid'] = True #
    tmp['simulator']['flags']['topology_operations_on_stopped'] = True #
    tmp['simulator']['flags']['topology_operations_on_unfreezing'] = True #
    tmp['simulator']['flags']['random_map'] = True #
    tmp['simulator']['flags']['random_initial_population'] = True
    tmp['simulator']['flags']['random_initial_groups'] = False
    tmp['simulator']['flags']['default_map'] = False #
    tmp['simulator']['flags']['update_peds'] = True # 
    
    
    
    tmp['simulator']['pedestrian'] = dict()
    tmp['simulator']['pedestrian']['parameters'] = dict()
    tmp['simulator']['pedestrian']['parameters']['max_stopping_groups'] = 1
    tmp['simulator']['pedestrian']['parameters']['max_number_pedestrian_stop'] = 5
    tmp['simulator']['pedestrian']['parameters']['max_unfreezing_pedestrians'] = 5
    
    
    tmp['simulator']['default'] = dict()
    tmp['simulator']['default']['box_size'] = 20
    tmp['simulator']['default']['ped_sparsity'] = 15
    box_size = tmp['simulator']['default']['box_size']
    
    tmp['simulator']['default']['states'] = np.array([ [0.0, 10, -0.5, -0.5, 0.0, -(box_size+5)],
                                [0.5, 10, -0.5, -0.5, 0.5, -(box_size+5)],
                                [4.0, 0.0, 0.0, 0.5, 1.0, (box_size+5)],
                                 [5.0, 0.0, 0.0, 0.5, 2.0, (box_size+5)],
                                 [6.0, 0.0, 0.0, 0.5, (box_size+5), 6.0],
                                 [3.0, 0.0, 0.0, 0.5, (box_size+5), 7.0] ], dtype = float).tolist()
    
    tmp['simulator']['default']['groups'] = [[],[],[1, 0],[], [2,3],[],[]]
    tmp['simulator']['default']['obstacles'] = dict()
    tmp['simulator']['default']['obstacles']['obstacle1'] = [[-1.0,1.0,-1.0,-1.0], [1.0,1.0,-1.0,2.5], [1.0,-1.0,2.5,2.5], [-1.0,-1.0,2.5,-1.0]]
    tmp['simulator']['default']['obstacles']['obstacle2'] = [[-7,-5,-7,-7], [-7,-7,-7,-3], [-7,-5,-3,-7]]
    
    
    tmp['simulator']['custom'] = dict()
    tmp['simulator']['custom']['map_number'] = 6#
    tmp['simulator']['custom']['ped_sparsity'] = 15#
    tmp['simulator']['custom']['random_population'] = dict()
    tmp['simulator']['custom']['random_population']['max_initial_peds'] = [0, 2, 5, 10, 15, 20]
    tmp['simulator']['custom']['random_population']['max_initial_groups'] = 7
    tmp['simulator']['custom']['random_population']['max_peds_per_group'] = 3

    tmp['simulator']['robot'] = dict()
    tmp['simulator']['robot']['robot_radius'] = 1
    tmp['simulator']['robot']['activation_threshold'] = 0.5
    tmp['simulator']['flags']['activate_ped_robot_force'] = True
    tmp['simulator']['robot']['force_multiplier'] = 1.5
    
    
    
    #tmp['simulator']
    
    
    
    
    
    for i in range(len(map_list)):
        path = os.path.split(os.getcwd())[0] +'\\maps\\' + map_list[i]
        tmp['map_files']['path_to_map_' + str(i)] = path
    

    with open('map_config.toml','w') as f:
        toml.dump(tmp, f)
        
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        
