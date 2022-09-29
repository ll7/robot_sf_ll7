import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from PIL import Image

#%%

from .extenders_py_sf.extender_sim import ExtdSimulator
from .utils.utilities import linspace

#%%
class BinaryOccupancyGrid():
    """The class is responsible of creating an object representing a discrete map of the environment"""
    def __init__(self, map_height = 1, map_length = 1, map_resolution = 10, peds_sim_env = None):
        #If no arguments passed create default map. Use this if you plan to load an
        #existing map from file

        ''' map_height: height of the map in meters
            map_length: lenght of the map in meters
            map_resolution: number of cells per meters
            grid_size: number of cells along x and y axis
            cell_size: dimension of a single cell expressed in meters'''
        if peds_sim_env == None:
            self.peds_sim_env = ExtdSimulator(np.zeros((1,7)) )                  
        else:
            self.peds_sim_env = peds_sim_env        
        
        self.map_height = map_height
        self.map_length = map_length
        self.map_resolution = map_resolution
        self.grid_size = dict()
        self.grid_size['x'] = int(np.ceil(self.map_length*self.map_resolution))
        self.grid_size['y'] = int(np.ceil(self.map_height*self.map_resolution))

        self.cell_size = dict()
        self.cell_size['x'] = self.map_length/(self.grid_size['x'])
        self.cell_size['y'] = self.map_height/(self.grid_size['y'])

        x = linspace(0 + self.cell_size['x']/2, self.map_length - self.cell_size['x']/2, self.grid_size['x'])
        y = linspace(0 + self.cell_size['y']/2, self.map_height - self.cell_size['y']/2, self.grid_size['y'])
        y = np.flip(y)
        
        self.X,self.Y = np.meshgrid(x,y);
        
        self.min_val = np.array( [self.X[0,0] - self.cell_size['x']/2 ,  self.Y[-1,0] - self.cell_size['y']/2 ])
        self.max_val = np.array([  self.X[0,-1] + self.cell_size['x']/2 ,   self.Y[0,0] + self.cell_size['y']/2 ])
        
        self.Occupancy = np.zeros((self.grid_size['y'],self.grid_size['x']), dtype = bool)
        self.OccupancyRaw = self.Occupancy.copy()
        self.OccupancyFixed = self.Occupancy.copy()
        self.OccupancyRobot = self.Occupancy.copy()
        self.OccupancyFixedRaw = self.Occupancy.copy()
        self.OccupancyOverall = self.Occupancy.copy()

        self.grid_origin = [0,0]
        
        self.add_noise = True

    def saveMap(self,filename):
        #Check validity of input filename
        filename,extension = os.path.splitext(filename)
        dir,name = os.path.split(filename)

        if not dir:
            #relative path
            home = os.getcwd()
            platform = sys.platform
            if platform == 'win32':
                path = home + '\\data\\maps\\' + name +'.json'
            elif platform == 'linux':
                path = home + '/data/maps/' + name + '.json'
            else:
                raise NameError('Platform not supported!')
        else:
            #absolute path
            path = filename + '.json'
                

        map = dict()
        map['properties'] = dict()
        map['properties']['height'] = self.map_height
        map['properties']['lenght'] = self.map_length
        map['properties']['resolution'] = self.map_resolution
        map['properties']['grid_size'] = self.grid_size
        map['properties']['cell_size'] = self.cell_size
        map['data'] = dict()
        map['data']['x'] = self.X.tolist()
        map['data']['y'] = self.Y.tolist()
        map['data']['occupancy'] = self.Occupancy.tolist()
        map['data']['occupancyFixed'] = self.OccupancyFixed.tolist()

        with open(path,'w') as f:
            json.dump(map,f)
            print('Map saved at:' + path)



    def loadMap(self,filename):
        
        filename, extension = os.path.splitext(filename)
        if not extension == '.json':
            raise NameError('Invalid map format')

        fullpath = filename + extension
        with open(fullpath,'r') as f:
            data = json.load(f)

        #Check integrity and validity of input file
        if 'properties' in data and 'data' in data:
            #First layer ok
            if 'height' in data['properties'] and 'lenght' in data['properties'] and 'resolution' in data['properties'] and \
                'grid_size' in data['properties'] and 'cell_size' in data['properties'] and 'x' in data['data'] and \
                'y' in data['data'] and 'occupancy' in data['data']:
                self.map_height = data['properties']['height']
                self.map_length = data['properties']['lenght']
                self.map_resolution = data['properties']['resolution']
                self.grid_size = data['properties']['grid_size']
                self.cell_size = data['properties']['cell_size']
                self.X = np.array(data['data']['x'])
                self.Y = np.array(data['data']['y'])
                self.Occupancy = np.array(data['data']['occupancy'])
                #self.OccupancyFixed = np.array(data['data']['occupancyFixed'])
                self.OccupancyFixed = self.Occupancy
            else:
                raise NameError('Data Missing')
        else:
            raise NameError('Data Missing')


    
    def getPairOccupancy(self,pair,system_type = 'world', fixedObjectsMap = False):
        ''' This method returns the occupancy value of the map
            given a pair and a system_type'''

        #pair must be a (m,2) list or numpy array
        #system type must be either 'world' or 'grid'

        if system_type == 'world':
            pair = np.array(pair) #float is good
        elif system_type == 'grid':
            pair = np.array(pair,dtype=int) #pair must be casted to int
        else:
            raise NameError('Invalid System type')

        if system_type == 'world':
            #Need to search in world coordinates!
            #First check if input is valid
            pair =  self.check_if_valid_world_coordinates(pair)
            if not pair.any():
                raise ValueError('Invalid world coordinates with the current map!')

            #Now if pair is valid start searching the indexes of the corrisponding x and y values
            idx = self.convert_world_to_grid(pair)
        else:
            if not self.check_if_valid_grid_index(pair):
                raise ValueError('Invalid grid indices with the current map!')
            idx = pair

        values = list()
        if fixedObjectsMap:
            for i in range(idx.shape[0]):
                values.append(self.fixedObjectsMap[idx[i,0],idx[i,1]])
        else:
            for i in range(idx.shape[0]):
                values.append(self.Occupancy[idx[i,0],idx[i,1]])

        return values


    def setOccupancy(self,pair,val,system_type = 'world', fixedObjectsMap = False):
         #pair must be a (m,2) list or numpy array
        #system type must be either 'world' or 'grid'

        if system_type == 'world':
            pair = np.array(pair)
        elif system_type == 'grid':
            pair = np.array(pair,dtype=int)
        else:
            raise NameError('Invalid System type')

        val = np.array(val,dtype = bool)
        if not val.shape:
            val = val*np.ones((pair.shape[0],1))

        if system_type == 'world':
            #Need to search in world coordinates!
            #First check if input is valid
            pair = self.check_if_valid_world_coordinates(pair)
            
            if not pair.any():
                #print(f'Invalid world coordinates with the current map! {pair}')
                return None
               #raise ValueError('Invalid world coordinates with the current map!')

            #Now if pair is valid start searching the indexes of the corrisponding x and y values
            idx = self.convert_world_to_grid(pair)
        else:
            if not self.check_if_valid_grid_index(pair):
                raise ValueError('Invalid grid indices with the current map!')
            idx = pair

        if fixedObjectsMap:        
            for i in range(idx.shape[0]):
                self.OccupancyFixed[idx[i,0],idx[i,1]] = val[i]
                
            self.getObstaclesCoordinates()
            self.OccupancyFixed[:,0] = True
            self.OccupancyFixed[:,-1] = True
            self.OccupancyFixed[0,:] = True
            self.OccupancyFixed[-1,:] = True
            self.OccupancyOverall = self.OccupancyFixed
        else:
            for i in range(idx.shape[0]):
                self.Occupancy[idx[i,0],idx[i,1]] = val[i]
            self.updateOverallOccupancy()
            
    def getObstaclesCoordinates(self):
        self.obstaclesCoordinates = np.concatenate([self.X[self.OccupancyFixed][:,np.newaxis], self.Y[self.OccupancyFixed][:,np.newaxis] ], axis = 1)

        
    def updateOverallOccupancy(self):
        self.OccupancyOverall = np.logical_or(self.Occupancy, self.OccupancyFixed)

    """
    def getOccupancy(self, fixedObjectsMap = False):
        if fixedObjectsMap:
            occ = self.OccupancyFixed
        else:
            occ = self.Occupancy
        return occ
    """

    def show(self):
        plt.imshow(self.OccupancyOverall,\
            cmap = 'Greys', extent = [-self.grid_origin[0], -self.grid_origin[0] \
            + self.map_length, -self.grid_origin[1], -self.grid_origin[1] + self.map_height])
        plt.show()
        #Da sistemare


    def raycast(self, start_pt, end_pt):
        ''' Here it will be developed the easiest 
        implementation for a raycasting algorithm'''
        #Start point must belong to the map!
        idx_start = self.convert_world_to_grid(start_pt)
        #End point can also not belong to the map. We access to the last available
        idx_end = self.convert_world_to_grid_no_error(end_pt)

        d_row = abs(idx_end[0,1] - idx_start[0,1])
        d_col = abs(idx_end[0,0] - idx_start[0,0])
        
        if d_row == 0 and d_col == 0:
            #Start and end points are coincident
            return idx_start
        elif d_row == 0 and not d_col == 0:
            #Handle division by zero
            col_idx = linspace(idx_start[0,0],idx_end[0,0],d_col + 1)
            tmp = np.ones((col_idx.shape[0],2), dtype = int)
            tmp[:,1] = idx_end[0,1]*tmp[:,1]
            tmp[:,0] = col_idx

            return tmp

        else:
            #m = d_col/d_row
            m = (idx_end[0,0] - idx_start[0,0])/(idx_end[0,1] - idx_start[0,1])

            #Get indexes of intercepting ray
            
            if abs(m) <= 1:
                x = linspace(idx_start[0,1], idx_end[0,1], d_row + 1).astype(int) #columns index
                y = np.floor(m*(x - idx_start[0,1]) + idx_start[0,0]).astype(int)
                y[y>self.Occupancy.shape[0]-1] = self.Occupancy.shape[0]-1
            elif abs(m) > 1:
                y = linspace(idx_start[0,0], idx_end[0,0],d_col + 1).astype(int) #rows index
                x = np.floor((y - idx_start[0,0])/m + idx_start[0,1]).astype(int)
                x[x>self.Occupancy.shape[1]-1] = self.Occupancy.shape[1]-1

            indexes = np.zeros((x.shape[0],2), dtype = int)
            indexes[:,0] = y
            indexes[:,1] = x
            return indexes

    def does_ray_collide(self,ray_indexes):
        ''' This method checks if a given input ray
        intercept an obstacle present in the map'''

        #Input ray must be the output of the previous function!
        #so a (m,2) numpy array of ints

        intersections = self.OccupancyOverall[ray_indexes[:, 0], ray_indexes[:, 1]]
        if intersections.any():
            idx = np.where(intersections)[0][0]
            return  True, ray_indexes[idx,:], [self.X[0, ray_indexes[idx, 1]], self.Y[ray_indexes[idx, 0], 0]]
        return (False, None, None)
    
    

    def insertScan(self,scan, pose):
        ''' This method adds occupied value to the occupancy matrix
        given a ray input matrix

            pose: must be a (1,3) array containing (x,y,orientation)
            scan: must be a data structure with the following structure

                -properties: -range
                             -angle_opening
                             -angle_increment

                -data:       -ranges
                             -angles
                             -intensities'''
        if not self.check_if_valid_world_coordinates([pose[0], pose[1]]).any() :
            raise ValueError('Pose not in map!')

        return False



    def inflate(self,radius, fixedObjectsMap = False):
        ''' Grow in size the obstacles'''
        #create a copy of the occupancy matrix
        int_radius_step = round(self.map_resolution*radius)
        if fixedObjectsMap:  # problem here!!!!
            eval_points = np.asarray(np.where(self.OccupancyFixed)).T
            self.OccupancyFixed = fill_surrounding(self.OccupancyFixed,int_radius_step,eval_points)
            self.updateOverallOccupancy()
        else:
            eval_points = np.asarray(np.where(self.Occupancy)).T
            self.Occupancy = fill_surrounding(self.Occupancy,int_radius_step,eval_points, add_noise = self.add_noise)
            self.updateOverallOccupancy()
                        


    def check_if_valid_world_coordinates(self,pair, margin = 0):

        if isinstance(pair,list):
            pair = np.array(pair)
            
        if len(pair.shape)<2:
            pair = pair[np.newaxis,:]
    
        offset = margin*np.array([self.map_length, self.map_height])

        valid_pairs = np.bitwise_and( (pair >= (self.min_val + offset)).all(axis = 1) , (pair <= (self.max_val - offset)).all(axis = 1) )
        if valid_pairs.all():
            return pair
        elif valid_pairs.any():
            return pair[valid_pairs,:]
        else:
            return np.array(False)



    def check_if_valid_grid_index(self,pair):
        for i in range(pair.shape[0]):
            if pair[i,0] < 0 or pair[i,0] > self.grid_size['y'] or pair[i,1] < 0 or pair[i,1] > self.grid_size['x'] or not \
                issubclass(pair.dtype.type,np.integer):
                return False
        return True

    def convert_world_to_grid(self,pair):
        
        pair = self.check_if_valid_world_coordinates(pair)
        #if not pair.any():
        #    raise ValueError('Invalid world coordinates with the current map!')
        return self.convert_world_to_grid_no_error(pair)


    def convert_grid_to_world(self,pair):
        if not self.check_if_valid_grid_index(pair):
            raise ValueError('Invalid grid indices with the current map!')
        val = np.zeros(pair.shape)
        for i in range(pair.shape[0]):
            val[i,0] = self.X[0,pair[i,1]]
            val[i,1] = self.Y[pair[i,0],0]

        return val


    def convert_world_to_grid_no_error(self,pair):
        return np.concatenate((np.abs(self.Y[:,0][:,np.newaxis] - pair[:,1].T).argmin(axis = 0)[:,np.newaxis], \
                               np.abs(self.X[0,:][:,np.newaxis] - pair[:,0].T).argmin(axis = 0)[:,np.newaxis]), axis = 1)


    #update map from pedestrians simulation environment
    def update_from_peds_sim(self, map_margin = 1.5, fixedObjectsMap = False):
        # shift peds sim map to top-left quadrant if necessary
        if self.X[0,0] == self.cell_size['x']/2:
            maps_alignment_offset = map_margin*self.peds_sim_env.box_size
        else:
            maps_alignment_offset = 0
            
        if fixedObjectsMap:
            # assign fixed obstacles to map
            tmp = [item for item in self.peds_sim_env.env.obstacles if not item.size==0 ]
            obs_coordinates = np.concatenate(tmp)
            new_obs_coordinates = obs_coordinates + np.ones((obs_coordinates.shape[0],2))*maps_alignment_offset
            # reset occupancy map
            self.OccupancyFixed = np.zeros(self.OccupancyFixed.shape,dtype = bool)
            self.setOccupancy(new_obs_coordinates, np.ones((obs_coordinates.shape[0],1),dtype = bool), fixedObjectsMap = fixedObjectsMap)   
            zeros_mat = np.zeros(self.OccupancyFixed.shape, dtype = bool)
            zeros_mat[::2,::2]=True
            self.OccupancyFixedRaw = np.logical_and(self.OccupancyFixed.copy(), zeros_mat)
            self.inflate(.3, fixedObjectsMap=fixedObjectsMap)
        else:
            # get peds states
            peds_pos = self.peds_sim_env.current_positions
            # add offset to pedestrians maps (it can also have negative coordinates values)
            n_pedestrians = peds_pos.shape[0]
            peds_new_coordinates = peds_pos + np.ones((n_pedestrians,2))*maps_alignment_offset
            # reset occupancy map
            self.Occupancy = np.zeros(self.Occupancy.shape,dtype = bool)
            # assign pedestrians positions to robotMap
            self.setOccupancy(peds_new_coordinates, np.ones((n_pedestrians,1)))
            self.OccupancyRaw =self.Occupancy.copy()
            self.inflate(.4) # inflate pedestrians only


    def move_map_frame(self,new_position):
        ''' This method will change the position of the grid 
            origin. By default when constructing '''
        #if not type(range,list) or not type(range,np.ndarray):
        #    raise ValueError('Input argument for new position of the grid must be a list or a numpy array')

        self.X = self.X - new_position[0]
        self.Y = self.Y - new_position[1]
        self.grid_origin = [new_position[0], new_position[1]]
        
        self.min_val = np.array( [self.X[0,0] - self.cell_size['x']/2 ,  self.Y[-1,0] - self.cell_size['y']/2 ])
        self.max_val = np.array([  self.X[0,-1] + self.cell_size['x']/2 ,   self.Y[0,0] + self.cell_size['y']/2 ])
        
        
    def center_map_frame(self):
        self.move_map_frame([self.map_length/2,self.map_height/2])


    def load_from_image(self,filename, resolution):
        ''' This method loads a map from a figure'''

        if not isinstance(resolution,int) or resolution <= 0:
            raise ValueError('Resolution must be a positive integer!')

        in_image = Image.open(filename,'r').convert('L')
        lenght, height = in_image.size #Number of cells

        #Initialize attributes
        self.map_length = lenght/resolution
        self.map_height = height/resolution
        self.map_resolution = resolution
        self.grid_size['x'] = lenght
        self.grid_size['y'] = height

        self.cell_size['x'] = self.map_length/(self.grid_size['x'])
        self.cell_size['y'] = self.map_height/(self.grid_size['y'])

        x = linspace(0 + self.cell_size['x']/2, self.map_length - self.cell_size['x']/2, self.grid_size['x'])
        y = linspace(0 + self.cell_size['y']/2, self.map_height - self.cell_size['y']/2, self.grid_size['y'])
        y = np.flip(y)

        #Initializing matrices
        self.X = np.zeros((self.grid_size['y'], self.grid_size['x']))
        for i in range(self.X.shape[0]):
            self.X[i,:] = x

        self.Y = np.zeros((self.grid_size['y'], self.grid_size['x']))
        for i in range(self.Y.shape[1]):
            self.Y[:,i] = y

        self.Occupancy = np.zeros((height,lenght), dtype = bool)

        self.grid_origin = [0,0]

        #get pixel object
        pix_obj = in_image.load()

        #Apply mask to pixel object and assign values to 
        #Occupancy matrix

        for i in range(in_image.size[0]):
            for j in range(in_image.size[1]):
                if pix_obj[i,j] > 0 and pix_obj[i,j] < 255:
                    pix_obj[i,j] = 0

                if pix_obj[i,j] == 0:
                    #cell is occupied
                    self.Occupancy[j,i] = True
                elif pix_obj[i,j] == 255:
                    #cell is free
                    self.Occupancy[j,i] = False

        self.OccupancyFixed = self.Occupancy
        in_image.close()

def fill_surrounding(matrix,int_radius_step,coords, add_noise = False):

    def n_closest_fill(x: np.ndarray, n, d, new_fill):
        x_copy = x.copy()
        x_copy [n[0]-d:n[0]+d+1,n[1]-d:n[1]+d+1] = new_fill
        x = np.logical_or(x,x_copy)
        return x

    window = np.arange(-int_radius_step, int_radius_step + 1)
    msh_grid = np.stack(np.meshgrid(window, window), axis=2)
    bool_subm = np.sqrt(np.square(msh_grid).sum(axis=2)) < int_radius_step

    N = 2 * int_radius_step + 1
    p = np.round(np.exp(-(np.square(msh_grid) / 15).sum(axis=2)), 3)

    for i in np.arange(coords.shape[0]):
        if (coords[i,:]>int_radius_step).all() and (coords[i,:]<matrix.shape[0] - int_radius_step).all():
            bool_subm_i = np.random.random(size=(N, N)) < p if add_noise else bool_subm
            matrix = n_closest_fill(matrix, coords[i,:],int_radius_step, bool_subm_i) 
        else:
            matrix_filled = fill_surrounding_brute(matrix.copy(),int_radius_step,coords[i,0],coords[i,1])
            matrix = np.logical_or(matrix,matrix_filled)

    return matrix
        


def fill_surrounding_brute(matrix,int_radius_step,i,j):
        
    row_eval = np.unique(np.minimum(matrix.shape[0]-1,np.maximum(0,np.arange(i-int_radius_step , i+1+int_radius_step ))))
    col_eval = np.unique(np.minimum(matrix.shape[1]-1,np.maximum(0,np.arange(j-int_radius_step , j+1+int_radius_step ))))
    
    eval_indices = np.sqrt((row_eval-i)[:,np.newaxis]**2+(col_eval-j)[np.newaxis,:]**2)<int_radius_step
    
    matrix[row_eval[0]:row_eval[-1]+1,col_eval[0]:col_eval[-1]+1] = eval_indices
    
    return matrix
