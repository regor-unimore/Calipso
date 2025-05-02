import math
import numpy as np
from statistics import mean
import time
import gurobipy as gp
from gurobipy import GRB
from shapely.geometry import Point, Polygon, MultiPolygon

#test parameters
seeds = [5, 85, 582, 133, 3371, 8759, 17803, 24093, 69350, 81604]
limits = {'VHR': [160, 80, 40, 20], 'HR': [80, 40, 20, 10], 'MR': [40, 20, 10, 5], 'LR': [20, 10, 5, 2], 'VLR': [10, 5, 2, 1]}

# gra = 5000
nP = 113651
nPI = {1000: 1568561, 2000: 392163, 3000: 174283, 5000: 62736, 10000: 15681}
# a = {0.85: 1, 0.75: 2, 0.65: 6, 0.55: 12}
nCat = 4
nbPicturesPerDay = 1

# Create the structure for the info
coeffQ = [0, 0.25, 0.5, 1]
# coeffS = [0.5, 1]
round_angle = 2 * math.pi


def mycallback(model, where):
    # model._solution = model.cbGet(GRB.Callback.MIPSOL_OBJ)
    if where == GRB.Callback.MIPSOL:
        model._incumbenttime = model.cbGet(GRB.Callback.RUNTIME)


with open("2_RawDataFuture/day_2023_2025.dat", "r") as file:
    day = [line for line in file.readlines()]

with open("2_RawDataFuture/x_2023_2025.dat", "r") as file:
    x = [float(line) for line in file.readlines()]

with open("2_RawDataFuture/y_2023_2025.dat", "r") as file:
    y = [float(line) for line in file.readlines()]

angle = [0]*nP
for i in range(nP):
    angle[i] = math.atan2(y[i], x[i])
    if angle[i] < 0:
        angle[i] = round_angle + angle[i]

for gra in [1000]:  # , 3000, 2000, 1000]:

    with open(f"0_RawDataGeneral/CatPIAV_{gra}_{nCat}.txt", "r") as file:
        w_catPIAV = [coeffQ[int(line)] for line in file.readlines()]

    # with open(f"0_RawDataGeneral/PIScore_{gra}.txt", "r") as file:
        #  w_scorePI = [coeffS[int(line)] for line in file.readlines()]

    for L in [134]:  # ,134,268]:

        print(gra, L)

        with open(f"0_RawDataGeneral/PI_per_obs_{gra}_{L}_{nCat}.txt", "r") as file:
            PIpp = [[int(i) for i in line.strip().split("\t")]
                    if line != "" and line != "\n" else [] for line in file.readlines()]

        with open(f"0_RawDataGeneral/PI_per_obs_chrono_quality_class_{gra}_{L}_{nCat}.txt", "r") as file:
            w_quality_po = [[coeffQ[int(i)] for i in line.strip().split("\t")]
                            if line != "" and line != "\n" else [] for line in file.readlines()]

        with open("4_Solutions/H3/Heuristic3Solution_chrono_" + str(gra) + "_" + str(L) + "_" + str(nCat) + ".txt",
                  "r") as file:
            obj = float(file.readline().split("=")[1])
            file.readline()
            file.readline()
            selected_h3 = [int(file.readline()) for i in range(nP)]

        # ######################## MODEL ####################################################
        model = gp.Model()
        model.setParam("TimeLimit", 360)
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.Method, 2)

        nbTotPI = 0
        isPIActiveAtCatB = {(PIpp[i][j], w_quality_po[i][j]): 1 for i in range(nP) for j in
                            range(len(PIpp[i]))}  # se punto j può essere coperto con qualità q
        isActiveAtCat = {(i, j): 0 for (i, j) in
                         isPIActiveAtCatB.keys()}  # quante volte il punto j è coperto con qualità q
        nbPicPerDay = {day[i]: 0 for i in range(nP)}

        # ################# VARIABLES ######################
        isPositionActive = model.addVars(range(nP), lb=0, ub=1, vtype=GRB.INTEGER,
                                         name='Y')  # se scelgo l'osservazione i
        isPIActiveAtCat = model.addVars(isPIActiveAtCatB.keys(), lb=0, ub=1, vtype=GRB.INTEGER,
                                        name='X')  # se copro il punto j con qualità q

        model.update()

        for i in range(nP):
            nbPicPerDay[day[i]] += isPositionActive[i]
            for j in range(len(PIpp[i])):
                isActiveAtCat[(PIpp[i][j], w_quality_po[i][j])] += isPositionActive[i]

        # ######## OBJECTIVE FUNCTION ######################
        for (i, j) in isPIActiveAtCatB.keys():
            if j > w_catPIAV[i]:
                nbTotPI += isPIActiveAtCat[(i, j)] * (j - w_catPIAV[i])  # * w_scorePI[i]

        model.setObjective(nbTotPI, sense=gp.GRB.MAXIMIZE)

        # ########## CONSTRAINTS ###########################
        model.addConstrs(isPIActiveAtCat[(i, j)] <= isActiveAtCat[(i, j)] for (i, j) in
                         isPIActiveAtCatB.keys())  # collegamento tra scelta delle osservazioni e qualità dei punti
        model.addConstrs(isPIActiveAtCat.sum(i, '*') <= 1 for i in
                         range(nPI[gra]))  # ogni punto j può raccogliere profitto da una qualità soltanto
        model.addConstrs(nbPicPerDay[i] <= nbPicturesPerDay for i in nbPicPerDay.keys())  # capacità giornaliera

        # ######################## MODEL EXECUTION ###############################
        for ttype in ['LR', 'VLR']: #, 'HR', 'MR', 'LR', 'VLR']:

            n_iter = 10
            coeff_0 = 0.15
            step = 0.10

            obj_values = []
            coverture_values = []
            time_values = []
            iter_values = []

            for k in range(n_iter):

                t0 = time.time()
                iter_w_imp = 0
                index = 0
                ilim = 0
                coeff = coeff_0
                delta = coeff * round_angle
                tf = 0
                test = []
                selected = selected_h3
                np.random.seed(seeds[k])

                while tf < 3600:

                    index += 1
                    if iter_w_imp == limits[ttype][ilim]:
                        if ilim == 3:
                            break
                        ilim += 1
                        coeff += step
                        iter_w_imp = 0
                        delta = coeff * round_angle
                    print(f'Iteration: {index}, Coeff: {coeff}')

                    # ## Selection of random observations to unfix ###
                    alpha = np.random.random() * (round_angle - delta)
                    #print(f'range: [{alpha}, {alpha+delta}]')

                    counter = 0
                    for i in range(nP):
                        if alpha <= angle[i] < alpha + delta:
                            #print(f'angle free: {angle}')
                            isPositionActive[i].lb = 0
                            isPositionActive[i].ub = 1
                            counter += 1
                        else:
                            isPositionActive[i].lb = selected[i]  # 1
                            isPositionActive[i].ub = selected[i]  # 1
                        isPositionActive[i].setAttr("start", selected[i])

                    print(f"Number fixed: {(1 - counter/nP) * 100}%")
                    #n_fixed[coeff].append((1 - counter/nP) * 100)

                    #model._incumbenttime = -1
                    tm1 = time.time() - t0  # inizio modello
                    model.optimize()
                    tm2 = time.time() - t0  # fine modello

                    # retrieve solution
                    print(f'Is the solution optimal? {model.status == gp.GRB.OPTIMAL}')
                    print(f'Solution: {nbTotPI.getValue()}\n')

                    obj_prec = obj
                    obj = nbTotPI.getValue()
                    selected = [0] * nP
                    for i in range(nP):
                        selected[i] = isPositionActive[i].x

                    # coefficient update
                    if obj_prec == obj:
                        iter_w_imp += 1
                    else:
                        iter_w_imp = 0
                        ilim = 0  # VND
                        coeff = coeff_0  # VND
                        delta = coeff * round_angle  # VND
                        test.append([index, coeff, obj, tm1, tm2, time.time() - t0])
                    tf = time.time() - t0

                # end while
                visited = [0] * nPI[gra]
                for i in range(nP):
                    if selected[i] == 1:
                        for j in PIpp[i]:
                            visited[j] = 1
                coverture = sum(visited)

                tf = time.time() - t0
                time_values.append(tf)
                iter_values.append(index)
                obj_values.append(obj)
                coverture_values.append(coverture)

                g = open(f"4_Solutions/{ttype}_{coeff_0}_{step}_geo_sliceVND_{gra}_{L}_{nCat}.txt", "a")
                g.write(f'Objective Value = {obj} \n')
                g.write(f'Number of PI covered = {coverture} \n')
                g.write(f'Total execution time = {tf} \n')
                g.write(f'Number of iterations = {index} \n')
                g.write('Iter Coeff Obj StartM EndM Tend\n')
                for i in test:
                    g.write(f'{i[0]} {i[1]} {i[2]} {i[3]} {i[4]} {i[5]} \n')
                g.write('\n')
                g.close()

                f = open(f"4_Solutions/{ttype}_{coeff_0}_{step}_geo_sliceVND(sol)_{gra}_{L}_{nCat}.txt", "a")
                f.write(f'Objective Value = {obj} \n')
                for i in selected:
                    f.write("%d\n" % i)
                f.close()

            f = open(f"4_Solutions/{ttype}_{coeff_0}_{step}_geo_sliceVND_{gra}_{L}_{nCat}.txt", "a")
            f.write('Average objective function = ' + str(mean(obj_values)) + '\n')
            f.write('Average number of PI covered = ' + str(mean(coverture_values)) + '\n')
            f.write('Average number of iterations = ' + str(mean(iter_values)) + '\n')
            f.write('Average total time = ' + str(mean(time_values)) + '\n\n')
            f.close()


# old code
# c = [5, 3, 2, 1]  # 10 slices 90%-4:87,5%-3:83,3%-2:75%-1:50%
# s = np.random.randint(-coeff, coeff)
# angle = math.atan2(y[i], x[i])
# if s * math.pi / coeff <= angle < (s + 1) * math.pi / coeff: