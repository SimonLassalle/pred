from sys import argv
from random import randrange, seed
import numpy as np
from gym_env.gym_polyhash.envs.env_source.Plan import Plan
from gym_env.gym_polyhash.envs.env_source.Building import Building
from time import time
from gym_env.gym_polyhash.envs.env_source.ProjectType import ProjectType
import math

if __name__ == '__main__':
    best_score = 0
    best_size = 0

    for prefered_size in range(0,110,10):
        print("prefered size=",prefered_size/100)

        my_plan = Plan(argv[1])
        #print(my_plan)

        # for debug
        #seed(0)

        percent = 0
        i=0
        cpt=0
        step = 1000
        old_score = 0

        for tries in range(2):
            print("pass", tries)
            row=0
            column=0
            should_exit = False
            while not should_exit:
                potentials = []
                buildings = []
                max_potential = 0
                placed=False
                # there is a hack here to have better score even if we don't fill space efficiently...
                if my_plan.cellsVal[row, column] == 0:
                    # calculates the potential score gain that we would have with each kind of building
                    for building_type in my_plan.building_projects.keys():
                        if my_plan.canPlaceBuilding(building_type, row, column):
                            building = my_plan.createBuilding(building_type, row, column)
                            buildings.append(building)
                            if tries == 0:
                                if building.project.type == ProjectType.RESIDENTIAL:
                                    # for map F
                                    # factor=2-abs((prefered_size/100*my_plan.biggest_residential)-building.project.nb_cells)/my_plan.biggest_residential
                                    # otherwise
                                    factor=1-abs((prefered_size/100*my_plan.biggest_residential)-building.project.nb_cells)/my_plan.biggest_residential
                                elif my_plan.smallest_utility < my_plan.biggest_utility:
                                    factor=1-(building.project.nb_cells-my_plan.smallest_utility)/(my_plan.biggest_utility-my_plan.smallest_utility)
                                else:
                                    factor=1-(building.project.nb_cells-my_plan.smallest_utility)/(1+my_plan.biggest_utility-my_plan.smallest_utility)
                                potentials.append(my_plan.calcPotential(building)*factor)
                            else:
                                potentials.append(my_plan.calcPotential(building))

                    # if the potential gain is greater than zero, lets get the building that we have to place
                    if len(potentials) > 0:
                        max_potential = max(potentials)
                    if max_potential > 0 or cpt == 0:
                        argmax = [i for i, j in enumerate(potentials) if j == max_potential]
                        if len(argmax) > 1:
                            # if different building have the same score : take the smallest one
                            nb_cells = [buildings[i].project.nb_cells for i in argmax]
                            min_of_cells = min(nb_cells)
                            index_of_min = [i for i,j in enumerate(nb_cells) if j == min_of_cells]
                            index = argmax[index_of_min[randrange(0,len(index_of_min))]]
                        else:
                            index = argmax[0]
                        #print(potentials,argmax, index)
                        building = buildings[index]
                        my_plan.placeBuilding(building)
                        cpt+=1
                        placed=True
                    # otherwise there might be cases where we should still place a building
                    # even if the expected score gain is zero
                    elif my_plan.number_of_utility_projects == 1 and max_potential == 0:
                        selected = False
                        for building in buildings:
                            if building.project.type == ProjectType.UTILITY:
                                selected = True
                                break
                        if selected:
                            my_plan.placeBuilding(building)
                            cpt+=1
                            placed=True

                # then, try the next position on the plan
                column += 1
                if column >= my_plan.nb_columns:
                    column = 0
                    row += 1
                    if row >= my_plan.nb_rows:
                        should_exit = True
                i+=1
                # every N steps, show the percentage of plan already processed and save a debug map
                if i%step == 0 and i >= step:
                    percent = 100*np.sum(np.where(my_plan.cellsVal != 0, 1, 0))/(my_plan.nb_rows*my_plan.nb_columns)
                    if i%(10*step) == 0:
                        my_plan.calcScore()
                        print(cpt, "(", percent, "%)", my_plan.score )
                        # Early exit...
                        if my_plan.score == old_score:
                            should_exit = True
                        old_score = my_plan.score
                    else:
                        print(cpt, "(", percent, "%)" )

        # At the end, generate the output file and other interesting information
        print("Processing score...")
        my_plan.calcScore()
        best_score = max(my_plan.score, best_score)
        if my_plan.score == best_score:
            best_size = prefered_size
            print("Best score !")
        my_plan.savePlan()
        percent = 100*np.sum(np.where(my_plan.cellsVal != 0, 1, 0))/(my_plan.nb_rows*my_plan.nb_columns)
        print(cpt, "buildings (", percent, "%)")
        print("Score :" , my_plan.score, "/", my_plan.max_score)
        print("Best score :",best_score, " (size=", best_size, ")")
        #print()
        #print(my_plan.cellsVal)
        #print()
        #print(my_plan.cellsId)
