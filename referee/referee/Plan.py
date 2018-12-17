from referee.Building import BuildingProject, ResidentialProject, UtilityProject, Building
from referee.ProjectType import ProjectType
from imageio import imwrite
import numpy as np
import sys
from random import choices
import os.path

class Plan:

    def __init__(self, filename):
        self.filename = filename
        self.nb_rows = 0
        self.nb_columns = 0
        self.max_walking_distance = 0
        self.number_of_building_projects = 0
        self.number_of_utility_projects = 0
        self.number_of_residential_projects = 0
        self.biggest_residential = 0
        self.biggest_utility = 0
        self.smallest_residential = sys.maxsize
        self.smallest_utility = sys.maxsize
        self.building_projects = {}
        self.buildings = []
        self.cellsVal = None
        self.cellsId = None
        self.score = 0
        self.nb_services = 0
        self.max_residential_capacity = 0
        self.services = {}
        self.loadFile(filename)


    # Reset the plan for a new builing placement trial
    def reset(self):
        self.buildings = []
        self.cellsVal=np.zeros((self.nb_rows, self.nb_columns), dtype=np.int32)
        self.cellsId=np.full((self.nb_rows, self.nb_columns), -1, dtype=np.int32)

    # Loads an input file and initialize all internal structures
    def loadFile(self, filename):
        # this set is for detecting duplicate building projects
        unique_projects = set()
        with open(filename,'r') as file:
            # first line : map rows, columns max walking distance and number of buildings projects
            line=file.readline()
            fields=line.split()
            self.nb_rows = int(fields[0])
            self.nb_columns = int(fields[1])
            self.max_walking_distance = int(fields[2])
            self.number_of_building_projects = int(fields[3])
            self.cellsVal=np.zeros((self.nb_rows, self.nb_columns), dtype=np.int32)
            self.cellsId=np.full((self.nb_rows, self.nb_columns), -1, dtype=np.int32)
            for i in range(self.number_of_building_projects):
                # next we have number_of_buildings lines for each type of building
                line=file.readline()
                # build a project descriptor that will be used to detect if a building project is duplicated
                # in the input file
                prj_desc=line
                fields=line.split()
                if fields[0] == 'R':
                    self.number_of_residential_projects += 1
                    project = ResidentialProject(i, int(fields[1]), int(fields[2]), int(fields[3]))
                    if project.capacity > self.max_residential_capacity:
                        self.max_residential_capacity = project.capacity
                else:
                    self.number_of_utility_projects += 1
                    project = UtilityProject(i, int(fields[1]), int(fields[2]), int(fields[3]))
                    if project.service_type == 0:
                        print("warning: service type should not be zero !!!")
                    if not project.service_type in self.services.keys():
                        self.services[project.service_type] = []
                    self.services[project.service_type].append(project)
                # then we have the plan of the building
                for r in range(project.nb_rows):
                    line=file.readline()
                    prj_desc+=line
                    for c in range(project.nb_columns):
                        if line[c] == '#':
                            project.plan[r,c]=1
                # update the number of cells of the project
                project.nb_cells = np.sum(project.plan)
                if project.type == ProjectType.RESIDENTIAL:
                    self.biggest_residential = max(self.biggest_residential, project.nb_cells)
                    self.smallest_residential = min(self.smallest_residential, project.nb_cells)
                else:
                    self.biggest_utility = max(self.biggest_residential, project.nb_cells)
                    self.smallest_utility = min(self.smallest_utility, project.nb_cells)
                # add the new project
                if not prj_desc in unique_projects:
                    unique_projects.add(prj_desc)
                    self.building_projects[i] = project
                else:
                    self.number_of_building_projects -= 1
                    if project.type == ProjectType.RESIDENTIAL:
                        self.number_of_residential_projects -= 1
                    else:
                        self.number_of_utility_projects -= 1
                # generate the coords of the cells to be tested for score update
                #project.generateWalkingDistCoords(self.max_walking_distance)
                project.generateWalkingDistMask(self.max_walking_distance)
            self.nb_services = len(self.services)

    # Check if a builing can be placed at a specific position
    # for speed optimization, we suppose that row and column are correct (>=0 and < nb_row or nb_columns)
    def canPlaceBuilding(self, project_id, row, column):
        canPlace = False
        end_row = row + self.building_projects[project_id].nb_rows
        end_column = column + self.building_projects[project_id].nb_columns
        if end_row <= self.nb_rows and end_column <= self.nb_columns:
            canPlace = not np.any(np.logical_and(self.cellsVal[row:end_row, column:end_column], self.building_projects[project_id].plan))
        return canPlace

    # Creates the building but does not place it on the map
    # this also check what the neighboring buildings are
    def createBuilding(self, project_id, row, column):
        building_id=len(self.buildings)
        project = self.building_projects[project_id]
        new_building = Building(building_id, project, row, column)
        self.updateNeighbors(new_building)
        return new_building

    # Place an existing building on the map
    def placeBuilding(self, building):
        if building.project.type == ProjectType.RESIDENTIAL:
            value = building.project.capacity
        else:
            value = -(building.project.service_type)
            for residential_id in building.neighbors:
                self.buildings[residential_id-1].services.add(building.project.service_type)
                self.buildings[residential_id-1].score = self.buildings[residential_id-1].project.capacity * len(self.buildings[residential_id-1].services)
        row_end = building.row+building.project.nb_rows
        column_end = building.column+building.project.nb_columns
        self.cellsVal[building.row:row_end, building.column:column_end] += value*building.project.plan
        self.cellsId[building.row:row_end, building.column:column_end] += (building.id+1)*building.project.plan
        self.buildings.append(building)

    # Evaluates the potential score gain if placing it on the plan
    def calcPotential(self, building):
        potential = 0
        if building.project.type == ProjectType.RESIDENTIAL:
            potential = building.score
        else:
            for residential_id in building.neighbors:
                if not building.project.service_type in self.buildings[residential_id-1].services:
                    potential += self.buildings[residential_id-1].project.capacity
        return potential

    # Calculate the score of the whole plan
    def calcScore(self):
        # no need to recal everything (theoreticaly)
        """for building in self.buildings:
            if building.project.type == ProjectType.RESIDENTIAL:
                self.updateNeighbors(building)"""
        self.score = 0
        for building in self.buildings:
            if building.project.type == ProjectType.RESIDENTIAL:
                self.score += building.score

    # Old version of score estimation for a building
    """ def updateResidentialSlow(self, building_id):
        # get current building
        building = self.buildings[building_id]
        # get manhattan neighboors coords
        coords = building.project.manhattan_coords
        # Shift then at the building position
        coords = [ (building.row+coord[0], building.column+coord[1]) for coord in coords ]
        # keep only valid positions and only utility cells
        coords = filter( lambda coord : 0 <= coord[0] < (self.nb_rows-1) and  0 <= coord[1] < (self.nb_columns-1) and self.cellsVal[coord[0], coord[1]] < 0,coords)
        # list all valid cells
        services = [ -self.cellsVal[coord[0], coord[1]] for coord in coords]
        # update score
        building.services=set(services)
        building.score = building.project.capacity * len(building.services) """

    # clamp the mask  depending on the building position
    def clampMask( self, building, mask):
        map_row_start = building.row-self.max_walking_distance
        map_column_start = building.column-self.max_walking_distance
        map_row_end = map_row_start + mask.shape[0]
        map_column_end = map_column_start + mask.shape[1]
        if building.row-self.max_walking_distance < 0:
            row_start = self.max_walking_distance - building.row
            map_row_start += row_start
        else:
            row_start = 0
        if building.column-self.max_walking_distance < 0:
            column_start = self.max_walking_distance - building.column
            map_column_start += column_start
        else:
            column_start = 0
        if building.row+mask.shape[0]-self.max_walking_distance > self.nb_rows-1:
            row_end = mask.shape[0] - (building.row+mask.shape[0]-self.max_walking_distance - (self.nb_rows))
            map_row_end -= (building.row+mask.shape[0]-self.max_walking_distance - (self.nb_rows))
        else:
            row_end = mask.shape[0]
        if building.column+mask.shape[1]-self.max_walking_distance > self.nb_columns-1:
            column_end = mask.shape[1] - (building.column+mask.shape[1]-self.max_walking_distance - (self.nb_columns))
            map_column_end -= (building.column+mask.shape[1]-self.max_walking_distance - (self.nb_columns))
        else:
            column_end = mask.shape[1]

        return map_row_start, map_row_end, map_column_start, map_column_end, row_start, row_end, column_start, column_end

    # Update a residential building neighbors list (all) and score (only for residential)
    def updateNeighbors(self, building):
         # get manhattan neighboors mask
        mask = building.project.manhattan_mask
        # clamp the mask if necessary
        map_row_start, map_row_end, map_column_start, map_column_end, row_start, row_end, column_start, column_end = self.clampMask( building, mask)

        # get neighbor buildings
        zone = np.add(1,self.cellsId[map_row_start:map_row_end,map_column_start:map_column_end])
        vals = self.cellsVal[map_row_start:map_row_end,map_column_start:map_column_end]
        if building.project.type == ProjectType.RESIDENTIAL:
            zone = np.multiply(zone, np.where(vals < 0, 1, 0))
        else:
            zone = np.multiply(zone, np.where(vals > 0, 1, 0))
        zone = np.multiply(zone, mask[row_start:row_end,column_start:column_end]).flatten().tolist()
        unique = set(zone)
        if 0 in unique:
            unique.remove(0)
        building.neighbors = unique

        if building.project.type == ProjectType.RESIDENTIAL:
            for utility_id in building.neighbors:
                building.services.add(self.buildings[utility_id-1].project.service_type)
            building.score = building.project.capacity * len(building.services)

    # Display information about the plan
    def __str__(self):
        project=""
        project += "Project: " + self.filename + "\n"
        project += "Nb rows: " + str(self.nb_rows) + "\n"
        project += "Nb columns: " + str(self.nb_columns) + "\n"
        project += "Max walking distance: " + str(self.max_walking_distance) + "\n"
        project += "Number of building projects: " + str(self.number_of_building_projects) + "\n"
        project += "Number of services: " + str(self.biggest_residential) + "\n"
        project += "Bigest residential: " + str(self.biggest_utility) + "\n"
        project += "bigest utility: " + str(self.nb_services) + "\n"
        project += "Services: " + str(self.services) + "\n"

        for i in self.building_projects.keys():
            project += str(self.building_projects[i])
        return project

    # Saves the state of the building map (red = residential, green = utility)
    def saveDebugImage(self, filename):
        reds=np.array([[i, 0, 0] for i in range(32,256)])
        nb_reds=len(reds)-1
        temp = []
        for g in range(31,256,32):
            for b in range(31,256,32):
                temp.append([0,g,b])
        greens=np.array(temp)
        nb_greens = len(greens)
        np.random.shuffle(greens)
        img = np.zeros((self.cellsVal.shape[0], self.cellsVal.shape[1], 3), dtype =np.uint8)
        residential = np.where(self.cellsVal > 0)
        utility = np.where(self.cellsVal < 0)
        # red
        img[residential] = reds[nb_reds*self.cellsVal[residential]//self.max_residential_capacity]
        # green
        img[utility] = greens[-self.cellsVal[utility]%nb_greens]
        imwrite(filename, img )

    # Generates the output file (and building map)
    def savePlan(self):
        out_filename = self.filename[0:-3]+".out"
        with open(out_filename,'w') as file:
            file.write(str(len(self.buildings))+"\n")
            for i in range(len(self.buildings)):
                file.write(str(self.buildings[i])+"\n")
        score_filename = self.filename[0:-3]+".score"
        if os.path.isfile(score_filename):
            with open(score_filename,'r') as file:
                line = file.readline()
                self.max_score = int(line.split('/')[1])
                if self.score > self.max_score:
                    self.max_score = self.score
        else:
            self.max_score = self.score
        with open(score_filename,'w') as file:
            file.write(str(self.score)+"/"+str(self.max_score))
        img_filename = self.filename[0:-3]+".png"
        self.saveDebugImage(img_filename)
