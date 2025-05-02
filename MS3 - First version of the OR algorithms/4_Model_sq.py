import time
import gurobipy as gp
from gurobipy import GRB

#gra = 5000
nP = 113651
#ynP = 49
nPI = {1000: 1568561, 2000: 392163, 3000: 174283, 5000: 62736, 10000: 15681}
years = {10: 13068, 30: 36441, 49:56715, 50: 59964, 60: 67955, 80: 89823, 100: 113651}
nCat = 4
#L = 134
N = 1

# Create the structure for the info
coeffQ = [0, 0.25, 0.5, 1]
coeffS = [0.1, 1]

with open("2_RawDataFuture/day_2023_2025.dat", "r") as file:
    day = [line for line in file.readlines()]
    #day = day[:years[ynP]]


def mycallback(model, where):
  #model._solution = model.cbGet(GRB.Callback.MIPSOL_OBJ)
  if where == GRB.Callback.MIPSOL:
    model._incumbenttime = model.cbGet(GRB.Callback.RUNTIME)

for gra in [10000]:

    with open(f"0_RawDataGeneral/CatPIAV_{gra}_{nCat}.txt", "r") as file:
        w_catPIAV = [coeffQ[int(line)] for line in file.readlines()]

    with open(f"0_RawDataGeneral/PIScore_{gra}.txt", "r") as file:
        w_scorePI = [coeffS[int(line)] for line in file.readlines()]

    for L in [134]:

        print(gra, L)

        visited = [0]*nPI[gra]
        selected = [0]*nP #years[ynP]

        with open(f"0_RawDataGeneral/PI_per_obs_{gra}_{L}_4.txt", "r") as file:
            PIpp = [[int(i) for i in line.strip().split("\t")]
                    if line != "" and line != "\n" else [] for line in file.readlines()]
            #PIpp = PIpp[:years[ynP]]

        with open(f"0_RawDataGeneral/PI_per_obs_chrono_quality_class_{gra}_{L}_{nCat}.txt", "r") as file:
            w_quality_po = [[coeffQ[int(i)] for i in line.strip().split("\t")]
                            if line != "" and line != "\n" else [] for line in file.readlines()]
            #w_quality_po = w_quality_po[:years[ynP]]

        ############################# MODEL ################################################
        t1 = time.time() #start construction time
        model = gp.Model()
        model.setParam("TimeLimit", 3600)
        model.setParam(GRB.Param.Threads, 1)
        model.setParam(GRB.Param.Method, 2)#1
        model.setParam("MIPGap", 1e-6)
        #model.setParam(GRB.Param.MIPFocus, 1)

        nbTotPI = 0
        isPIActiveAtCatB = {(PIpp[i][j], w_quality_po[i][j]): 1 for i in range(nP) for j in range(len(PIpp[i]))} #se punto j può essere coperto con qualità q
        #isPIActiveAtCatB = {(p, q): 1 for p in range(nPI[gra]) for q in coeffQ}

        isActiveAtCat = {(i, j): 0 for (i,j) in isPIActiveAtCatB.keys()} #quante volte il punto j è coperto con qualità q
        nbPicPerDay = {day[i]: 0 for i in range(nP)}  #nP --> years[ynP]

        ################## VARIABLES ######################
        isPositionActive = model.addVars(range(nP), lb=0, ub=1, vtype=GRB.INTEGER, name='Y') #se scelgo l'osservazione i
        isPIActiveAtCat = model.addVars(isPIActiveAtCatB.keys(), lb=0, ub=1, vtype=GRB.INTEGER, name='X') #se copro il punto j con qualità q

        model.update()

        for i in range(nP):
            nbPicPerDay[day[i]] += isPositionActive[i]
            for j in range(len(PIpp[i])):
                isActiveAtCat[(PIpp[i][j], w_quality_po[i][j])] += isPositionActive[i]

        ######### OBJECTIVE FUNCTION ######################
        for (i, j) in isPIActiveAtCatB.keys():
            if j > w_catPIAV[i]:
                nbTotPI += isPIActiveAtCat[(i, j)] * (j - w_catPIAV[i]) * w_scorePI[i]

        model.setObjective(nbTotPI, sense=gp.GRB.MAXIMIZE)

        ########### CONSTRAINTS ###########################
        model.addConstrs(isPIActiveAtCat[(i, j)] <= isActiveAtCat[(i,j)] for (i, j) in isPIActiveAtCatB.keys()) #collegamento tra scelta delle osservazioni e qualità dei punti
        model.addConstrs(isPIActiveAtCat.sum(i, '*') <= 1 for i in range(nPI[gra])) #ogni punto j può raccogliere profitto da una qualità soltanto
        model.addConstrs(nbPicPerDay[i] <= N for i in nbPicPerDay.keys()) #capacità giornaliera

        t2 = time.time() #finish costruction time
        #model.optimize()

        # define private variable which can be set in callback
        model._incumbenttime = -1
        model.optimize(mycallback)
        t3 = time.time() #finish execution time

        for i in range(nP):
            selected[i] = isPositionActive[i].x #osservazioni selezionate
            if selected[i] == 1:
                for j in PIpp[i]:
                    visited[j] = 1
        coverture = sum(visited) #numero di punti coperti

        print(f'Is the solution optimal? {model.status == gp.GRB.OPTIMAL}\n')
        print('objective function = ', nbTotPI.getValue())
        print('number of PI visited = ', coverture)
        print('total time = ', t3-t1)
        print('construction time = ', t2-t1)
        print('execution time = ', t3 - t2)
        print('UB = ', model.ObjBound)
        print('Gap = ', model.MIPGap)
        print('Time incumbent', model._incumbenttime)


        f = open(f"4_Solutions/Model/ModelSolution_chrono_{gra}_{L}_{nCat}_10.txt", "w")
        f.write(f'Gra: {gra}, L: {L}') #, Year: {ynP} \n')
        f.write('objective function = ' + str(nbTotPI.getValue()) + '\n')
        f.write('number of PI visited = ' + str(coverture) + '\n')
        f.write('total time = ' + str(t3 - t1) + '\n')
        f.write('construction time = ' + str(t2 - t1) + '\n')
        f.write('execution time = ' + str(t3 - t2) + '\n')
        f.write('UB = ' + str(model.ObjBound) + '\n')
        f.write('Gap = ' + str(model.MIPGap) + '\n')
        f.write('explored nodes= ' + str(model.NodeCount) + '\n')
        f.write(f'Time incumbent= { model._incumbenttime } \n\n')

        
        for i in selected:
            f.write("%d\n" % i)
        f.close()




