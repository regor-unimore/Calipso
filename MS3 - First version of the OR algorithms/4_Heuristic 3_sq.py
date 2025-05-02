import time
import numpy as np
# Granularity
#gra = 5000

# Nb dots
nP = 113651
nPI = {1000:1568561,2000:392163, 3000: 174283, 5000:62736,10000:15681}

nCat = 4
#L = 134
N = 1

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

for gra in [10000, 5000, 3000, 2000, 1000]: #[1000, 2000, 5000, 10000]:

    with open(f"0_RawDataGeneral/CatPIAV_{gra}_{nCat}.txt", "r") as file:
        w_catPIAV = [coeffQ[int(line)] for line in file.readlines()]

    #with open(f"0_RawDataGeneral/PIScore_{gra}.txt", "r") as file:
        #w_scorePI = [coeffS[int(line)] for line in file.readlines()]

    for L in [268]:

        print(gra, L)

        selected = [0] * nP
        visited = [0] * nPI[gra]
        w_quality = [w_catPIAV[i] for i in range(nPI[gra])]
        days = {i: 0 for i in set(day)}

        with open(f"0_RawDataGeneral/PI_per_obs_{gra}_{L}_4.txt", "r") as file:
            PIpp = [[int(i) for i in line.strip().split("\t")]
                    if line != "" and line != "\n" else [] for line in file.readlines()]

        with open(f"0_RawDataGeneral/PI_per_obs_chrono_quality_class_{gra}_{L}_{nCat}.txt", "r") as file:
            w_quality_po = [[coeffQ[int(i)] for i in line.strip().split("\t")]
                            if line != "" and line != "\n" else [] for line in file.readlines()]

        t0 = time.time()
        #with open(f"Intersections_{gra}_134.txt", "r") as file:
         #   map = [[int(i) for i in line.strip().split("\t")]
          #                  if line != "" and line != "\n" else [] for line in file.readlines()]
        print("OK")
        t0 = time.time()-t0
        t1 = time.time()
        #ttime = 0
        #z_o = [sum(max(0, (w_quality_po[i][j] - w_quality[PIpp[i][j]]) * w_scorePI[PIpp[i][j]]) for j in range(len(PIpp[i]))) for i in range(nP)] #pesi iniziali
        z_o = [sum(
            max(0, w_quality_po[i][j] - w_quality[PIpp[i][j]]) for j in range(len(PIpp[i])))
               for i in range(nP)]  # pesi iniziali

        # solution construction
        while sum(selected) < len(days)*N:

            curS = sorted(range(nP), key=lambda i: z_o[i])
            maxobs = curS[-1]
            if z_o[maxobs] == -1:
                break
            selected[maxobs] = 1
            for l, j in enumerate(PIpp[maxobs]):  # per ogni PI nello scatto selezionato
                w_quality[j] = max(w_quality[j], w_quality_po[maxobs][l])
                visited[j] = 1
            days[day[maxobs]] += 1
            if days[day[maxobs]] == N or day[maxobs] == '2023NOV17':  # svuoto le osservazioni non selezionabili
                for i in ranges[day[maxobs]]:
                    z_o[i] = -1

            for i in range(nP): #map[maxobs]:  # aggiorni i pesi di chi ha intersezioni
                if z_o[i] != -1:
                    #z_o[i] = sum(max(0, (w_quality_po[i][j] - w_quality[PIpp[i][j]]) * w_scorePI[PIpp[i][j]]) for j in range(len(PIpp[i])))
                    z_o[i] = sum(max(0, w_quality_po[i][j] - w_quality[PIpp[i][j]]) for j in
                         range(len(PIpp[i])))

            #ttime = time.time()-t1

        # compute objective function
        ob = 0
        for j in range(nPI[gra]):
            if w_quality[j] > w_catPIAV[j]:
                ob += (w_quality[j] - w_catPIAV[j]) #* w_scorePI[j]

        print("objective function = ", ob)
        print("visited PI = ", sum(visited))
        t2 = time.time()
        print("import time = ", t0)
        print("execution time = ", t2-t1)

        #print solution
        f = open(f"4_Solutions/H3/Heuristic3Solution_chrono(nomap)_{gra}_{L}_{nCat}.txt", "w+")
        f.write(f"objective function = {ob}\nvisited PI = {sum(visited)}\nexecution time = {t2 - t1}\n")
        for i in selected:
            f.write("%d\n" % i)
        f.close()








''' OLD CODE

#curS = sorted(zip(PIpp, range(nP)), key=lambda i: sum(
 #   max(0, (w_quality_po[i[1]][j] - w_quality[i[0][j]])) for j in range(len(i[0]))))  # * w_scorePI[i[0][j]]
curS = sorted(zip(PIpp, range(nP)), key=lambda i: z_o[i[1]])

idx = -1
maxobs = curS[idx] 
while days[day[maxobs[1]]] == N or selected[maxobs[1]]==1:
    idx -= 1
    maxobs = curS[idx]
#if days[day[maxobs[1]]] < N:
selected[maxobs[1]] = 1 #prendo l'ultimo, qualità totale maggiore
for l, j in enumerate(maxobs[0]):  # per ogni PI nello scatto selezionato
    w_quality[j] = max(w_quality[j], w_quality_po[maxobs[1]][l])
    visited[j] = 1
days[day[maxobs[1]]] += 1

if days[day[maxobs[1]]] == N:  # svuoto le osservazioni non selezionabili
    for i in ranges[day[maxobs[1]]]:
        PIpp[i] = ""

for i in map[maxobs[1]]:  # aggiorni i pesi di chi ha intersezioni
    z_o[i] = sum(max(0, w_quality_po[i][j] - w_quality[PIpp[i][j]]) for j in range(len(PIpp[i])))
'''