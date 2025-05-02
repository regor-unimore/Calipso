import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model

n = 113651

#df = pd.read_csv(sys.argv[1],sep=";")
df = pd.read_csv('all_2023_2025.csv', sep=";")

df['solar_longitude_cos'] = np.cos(df['solar_longitude_2023_2025']) #coverto l'angolo in sin e cos
df['solar_longitude_sin'] = np.sin(df['solar_longitude_2023_2025'])

X = df.drop(columns=['ephemeris_time_2023_2025', 'orbit_number_2023_2025', 'frequency_2023_2025', 'solar_longitude_2023_2025'])#.to_numpy()
#print(X.head())
y = {i:[] for i in range(10)}
y_avg = [0]*n

for i in range(10):
    print(i)
    #model = keras.models.load_model("C:/Users/user/Desktop/Ferrari Benedetta/Marsis/Machine Learning/Results/New Data/nn_geo_model_0.keras")
    model = load_model(f"nn_chrono_model_{i}.keras")
    model.summary()
    y[i] = model.predict(X)
    print(y[i])
    np.savetxt(f'nn_chrono_prediction_2023_2025_{i}.txt', y[i])

for i in range(n):
    y_avg[i] = sum(y[k][i] for k in range(10))/10

np.savetxt(f'nn_chrono_prediction_2023_2025_avg.txt', y_avg)
