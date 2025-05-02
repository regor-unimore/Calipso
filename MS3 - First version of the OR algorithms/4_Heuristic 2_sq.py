import time
import numpy as np
# Granularity
#gra = 5000

# Nb dots
nP = 113651
nPI = {1000:1568561,2000:392163, 3000: 174283, 5000:62736,10000:15681}

nCat = 4
#L = 134
N = 4

# Create the structure for the info
coeffQ = [0, 0.25, 0.5, 1]
#coeffS = [0.5, 1]

with open("2_RawDataFuture/day_2023_2025.dat", "r") as file:
    day = [line for line in file.readlines()]

ranges = {}
beg = 0
while beg < nP: #scorro tutti i giorni
    end = beg
    while end+1 < nP and day[end+1] == day[beg]:
        end = end+1
    ranges[day[beg]] = range(beg, end+1)
    beg = end+1

for gra in [10000, 5000, 3000, 2000, 1000]:

    with open(f"0_RawDataGeneral/CatPIAV_{gra}_{nCat}.txt", "r") as file:
        w_catPIAV = [coeffQ[int(line)] for line in file.readlines()]

    #with open(f"0_RawDataGeneral/PIScore_{gra}.txt", "r") as file:
     #   w_scorePI = [coeffS[int(line)] for line in file.readlines()]

    for L in [67]:

        print(gra, L)

        selected = [0] * nP
        visited = [0] * nPI[gra]
        w_quality = [w_catPIAV[i] for i in range(nPI[gra])]

        with open(f"0_RawDataGeneral/PI_per_obs_{gra}_{L}_4.txt", "r") as file:
            PIpp = [[int(i) for i in line.strip().split("\t")]
                    if line != "" and line != "\n" else [] for line in file.readlines()]

        with open(f"0_RawDataGeneral/PI_per_obs_chrono_quality_class_{gra}_{L}_{nCat}.txt", "r") as file:
            w_quality_po = [[coeffQ[int(i)] for i in line.strip().split("\t")]
                            if line != "" and line != "\n" else [] for line in file.readlines()]

        # solution construction
        t1 = time.time()
        for k in range(N): #assegno la Kesima obs a tutti
            beg = 0
            while beg < nP: #scorro tutti i giorni
                end = beg
                while end+1 < nP and day[end+1] == day[beg]:
                    end = end+1
                cur = PIpp[beg:(end+1)]
                curI = range(beg, end+1) #indice della riga che contiene lo scatto
                curS = sorted(zip(cur, curI), key=lambda i: sum(max(0, (w_quality_po[i[1]][j] - w_quality[i[0][j]])) for j in range(len(i[0]))))  # * w_scorePI[i[0][j]]

                if len(curS) >= 1:
                    maxobs = curS[-1]
                    selected[maxobs[1]] = 1 #prendo l'ultimo, qualità totale maggiore
                    for l, j in enumerate(maxobs[0]):  # per ogni PI nello scatto selezionato
                        w_quality[j] = max(w_quality[j], w_quality_po[maxobs[1]][l])
                        visited[j] = 1
                beg = end+1

        # compute objective function
        ob = 0
        for j in range(nPI[gra]):
            if w_quality[j] > w_catPIAV[j]:
                ob += (w_quality[j] - w_catPIAV[j])  # * w_scorePI[j]

        print("objective function = ", ob)
        print("visited PI = ", sum(visited))
        t2 = time.time()
        print("execution time = ", t2 - t1)

        # print solution
        f = open(f"4_Solutions/H2/Heuristic2Solution_chrono_{gra}_{L}_{nCat}_{N}.txt", "w+")
        f.write(f"objective function = {ob}\nvisited PI = {sum(visited)}\nexecution time = {t2 - t1}\n")
        for i in selected:
            f.write("%d\n" % i)
        f.close()
