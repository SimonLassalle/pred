import numpy as np
from scipy.ndimage import distance_transform_cdt
from referee.ProjectType import ProjectType

# This is a building project -> kind of building that can be put on the plan
class BuildingProject:
    def __init__(self, id, nb_rows, nb_columns):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.id=id
        self.type = ProjectType.NONE
        self.plan = np.zeros((nb_rows, nb_columns), dtype=np.int32)
        self.nb_cells = 0
        # this stores the coordinates of the mask (should now not be used with the fast algo)
        # self.manhattan_coords = []
        # this stores the mask itseflt for computing the manhattan distance
        self.manhattan_mask = None

    # generates manhattan_mask, that is used for finding the buildings at a specified manhattan distance
    def generateWalkingDistMask(self, max_walking_distance):
        temp=np.zeros((self.nb_rows+2*max_walking_distance, self.nb_columns+2*max_walking_distance), np.int32)
        temp[max_walking_distance:max_walking_distance+self.nb_rows,max_walking_distance:max_walking_distance+self.nb_columns] = self.plan
        temp=1-temp
        # the mask is generated from the distance tranform to any border cell of a building
        dist=distance_transform_cdt(temp, 'manhattan')
        self.manhattan_mask = np.multiply(np.where(dist <= max_walking_distance, 1, 0), temp)

    # Generates manhattan_coords (but should now not be used)
    def generateWalkingDistCoords(self, max_walking_distance):
        """temp=np.zeros((self.nb_rows+2*max_walking_distance, self.nb_columns+2*max_walking_distance), np.int32)
        temp[max_walking_distance:max_walking_distance+self.nb_rows,max_walking_distance:max_walking_distance+self.nb_columns] = self.plan
        temp=1-temp
        dist=distance_transform_cdt(temp, 'manhattan')
        manhattan_coords = np.multiply(np.where(dist <= max_walking_distance, 1, 0), temp)
        manhattan_coords =  (manhattan_coords[0]-max_walking_distance, manhattan_coords[1]-max_walking_distance)
        for i in range(len(manhattan_coords[0])):
            self.manhattan_coords.append((manhattan_coords[0][i],manhattan_coords[1][i]))"""

    def __str__(self):
        project = ""
        project += "Project #" + str(self.id) + "\n"
        project += "\tType: " + str(self.type) + "\n"
        project += "\tNb rows: " + str(self.nb_rows) + "\n"
        project += "\tNb columns: " + str(self.nb_columns) + "\n"
        project += "\tPlan: \n" + str(self.plan) + "\n"
        return project

# Derived class for residential projects
class ResidentialProject(BuildingProject):
    def __init__(self, id, nb_rows, nb_columns, capacity):
        BuildingProject.__init__(self, id, nb_rows, nb_columns)
        self.type = ProjectType.RESIDENTIAL
        self.capacity = capacity

# Derived class for utility projects
class UtilityProject(BuildingProject):
    def __init__(self, id, nb_rows, nb_columns, service_type):
        BuildingProject.__init__(self, id, nb_rows, nb_columns)
        self.type = ProjectType.UTILITY
        self.service_type = service_type

# Class for the buildings that are really instanciated on the plan
class Building:
    def __init__(self, id, project, row, column):
        self.id = id
        self.project = project
        self.row = row
        self.column = column
        self.score = 0
        # stores the list of different services in the manhattan_mask neighborhood
        self.services = set()
        # stores the list of all the building ids in the manhattan_mask neighborhood
        self.neighbors = None

    # returns the list of coordinates occupied by the building
    def getOccupiedCoords(self):
        coords = np.where( self.project.plan == 1 )
        coords = [(r+self.row, c+self.column) for r,c in zip(coords[0], coords[1]) ]
        return coords

    def __str__(self):
        return str(self.project.id) + " " + str(self.row) + " " + str(self.column)
