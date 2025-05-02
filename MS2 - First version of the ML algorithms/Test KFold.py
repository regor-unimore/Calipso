import time
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump

#--- Lettura input ---#
orbit_to_remove = []
with open('data/orbit_to_remove') as file:
    for line in file:
        orbit_to_remove.append(float(line))

slice_idx = []
with open('data/slice_idx') as file:
    for line in file:
        slice_idx.append(int(line))

df = pd.read_csv("data/all_past.csv", sep=";")

#--- Elaborazione input ---#
frequency_to_keep = 4000000.0
df = df[df['FM_data_frequency'] == frequency_to_keep]
#df = df.replace([np.inf, -np.inf], np.nan).dropna() #ci sono nan solo nella colonna simulated
df = df[~df.FM_data_orbit_number.isin(orbit_to_remove)] #rimuovo l'8% dei dati
df['FM_data_solar_longitude_cos'] = np.cos(df['FM_data_solar_longitude']) #coverto l'angolo in sin e cos
df['FM_data_solar_longitude_sin'] = np.sin(df['FM_data_solar_longitude'])

X = df.drop(columns=['FM_data_dipole_tilt', 'FM_data_monopole_tilt','FM_data_ephemeris_time', 'FM_data_F10_7_index', 'FM_data_frequency', 'FM_data_median_corrected_echo_power', 'FM_data_orbit_number', 'FM_data_peak_corrected_echo_power', 'FM_data_peak_distorted_echo_power', 'FM_data_peak_simulated_echo_power', 'FM_data_solar_longitude'])

#X = df[['FM_data_x_coordinate', 'FM_data_y_coordinate']]
col_names = X.columns.tolist()
print(col_names)
X = X.to_numpy()
print(X.shape)

y = df['FM_data_peak_distorted_echo_power'].to_numpy()

#--- Algoritmi da testare ---#
algorithm = {
    'knn': KNeighborsRegressor(),
    'linear_regression': LinearRegression(),
    'decision_tree': DecisionTreeRegressor(random_state=0, criterion='squared_error', max_depth=20),
    'gr_boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=25, random_state=0, criterion='squared_error'),
    'random_forest': RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=0, max_depth=25, n_jobs=16),
    'neural_network': MLPRegressor(hidden_layer_sizes=(2, 1,), max_iter=50000, random_state=1, early_stopping=True),
    'support_v_r': SVR(C=10, gamma=0.0001, kernel='rbf')
}

#kf = GroupKFold(n_splits=10)
#kf.get_n_splits(X, slice_idx)

kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(X)

def prediction(alg, small):
    f = open("Results/New Data/"+alg+"_distorted/MAE_"+small+"_noantennas.txt", "w")
    fold = -1
    mae = []
    coeff = {column: [] for column in col_names}
    y_all_pred = np.zeros(y.shape)

    for train_index, test_index in kf.split(X, y):#, slice_idx):
        fold = fold + 1
        print(fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # standardizzo le features
        if small in ["knn", "lr", "nn", "svr"]:
            scaler = StandardScaler() #solo per lr e knn
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        f.write('TRAIN: ' + str(train_index)+'\n')
        f.write('TEST: ' + str(test_index)+'\n')

        if small == "rand":
            t1 = time.time()
            y_pred = np.random.choice(y_test, len(y_test))  # per random
            t2 = time.time()

        else:
            t1 = time.time()
            model = algorithm[alg]
            model.fit(X_train, y_train)
            #dump(model, 'Results/New Data/'+small+'_model_' + str(fold) + '.joblib')
            if small == 'lr': # salvo i coefficienti
                for i in range(X.shape[1]):
                    coeff[col_names[i]].append(model.coef_[i])
            if small in ['rf', 'dt', 'gb']: #salvo le feature importances
                for i in range(X.shape[1]):
                    coeff[col_names[i]].append(model.feature_importances_[i])

            y_pred = model.predict(X_test)
            t2 = time.time()

        # salvo i risultati del fold corrente
        f.write('  MAE = ' + str(mean_absolute_error(y_test, y_pred))+'\n')
        f.write('  MAPE = ' + str(mean_absolute_percentage_error(y_test, y_pred)) + '\n')
        f.write('  MSE = ' + str(mean_squared_error(y_test, y_pred)) + '\n')
        f.write('  Execution time = ' + str(t2 - t1) + '\n')

        mae.append(mean_absolute_error(y_test, y_pred))
        #np.savetxt("Results/New Data Shuffle/"+alg+"/"+small+"_y_test_" + str(fold)+".txt", y_test)
        #np.savetxt("Results/New Data Shuffle/"+alg+"/"+small+"_y_pred_" + str(fold)+".txt", y_pred)
        y_all_pred[test_index] = y_pred

    #salvo i risultato complessivi
    f.write('\nGlobal MAE = ' + str(mean_absolute_error(y, y_all_pred)))
    f.write('\nGlobal MAPE = ' + str(mean_absolute_percentage_error(y, y_all_pred)))
    f.write('\nGlobal MSE = ' + str(mean_squared_error(y, y_all_pred)))
    if small == 'lr':
        f.write('\n\nCoefficients = ' + '\n')
        for i in coeff.keys():  # per ogni feature
            f.write(str(i) + " " + str(mean(coeff[i])) + '\n')
    if small in ['rf', 'dt', 'gb']:
        f.write('\n\nFeature Importance = ' + '\n')
        for i in coeff.keys():  # per ogni feature
            f.write(str(i) + " " + str(mean(coeff[i])) + '\n')
    f.close()
    np.savetxt("Results/New Data/"+alg+"_distorted/"+small+"_y_all_pred_noantennas.txt", y_all_pred)


#prediction("random_pred","rand")
#prediction("knn", "knn")
#prediction("linear_regression", "lr")
#prediction("decision_tree", "dt")
prediction("random_forest", "rf")
#prediction("gr_boosting", "gb")
#prediction("neural_network", "nn")
#prediction("support_v_r", "svr")
