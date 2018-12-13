from sys import argv
from random import randrange, seed, shuffle
import numpy as np
from Plan import Plan
from Building import Building
from time import time
from ProjectType import ProjectType

if __name__ == '__main__':
    my_plan = Plan(argv[1])
    #print(my_plan)

    # for debug
    #seed(0)

    percent = 0
    i=0
    cpt=0
    nb_cells = 0
    old_nb_cells = -1
    threshold = 100
    step = 10000

    free_cells = [(i//my_plan.nb_columns,i%my_plan.nb_columns) for i in range(my_plan.nb_rows*my_plan.nb_columns) ]
    shuffle(free_cells)
    retry = []

    while nb_cells != old_nb_cells:
        old_nb_cells = nb_cells
        while len(free_cells) > 0:
            # generate a random building at a random position
            r_or_u = randrange(0,2)
            if r_or_u == 0:
                building_type = randrange(0,my_plan.number_of_residential_projects)
            else:
                building_type = randrange(my_plan.number_of_residential_projects,my_plan.number_of_building_projects)
            # get the real id of the project we have selected
            building_type = list(my_plan.building_projects)[building_type]
            row, column = free_cells.pop()
            # and try to place it on the plan
            if my_plan.canPlaceBuilding(building_type, row, column):
                building = my_plan.createBuilding(building_type, row, column)
                potential = my_plan.calcPotential(building)
                if potential > 0 or randrange(0,threshold) == 0 or cpt == 0:
                    my_plan.placeBuilding(building)
                    nb_cells += building.project.nb_cells
                    cpt+=1
                else:
                    retry.append((row,column))
            else:
                retry.append((row,column))
            i+=1
            # display progress...
            if i%step == 0 and i >= step:
                percent = 100*nb_cells/(my_plan.nb_rows*my_plan.nb_columns)
                print(cpt, "(", percent, "%)" )
        free_cells = retry
        retry = []
        shuffle(free_cells)
        print("retry", len(free_cells), "cells")

    # save output file and more
    print("Processing score...")
    my_plan.calcScore()
    percent = 100*np.sum(np.where(my_plan.cellsVal != 0, 1, 0))/(my_plan.nb_rows*my_plan.nb_columns)
    print(cpt, "buildings (", percent, "%)")
    print("Score :" , my_plan.score)
    #print()
    #print(my_plan.cellsVal)
    #print()
    #print(my_plan.cellsId)
    my_plan.savePlan()