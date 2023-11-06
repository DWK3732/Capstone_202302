import pandas as pd
import netCDF4 as nc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. 데이터 불러오기
drifter_2005 = nc.Dataset('./learn_data/NOAA_drifter_2005.nc','r')
sea_data = nc.Dataset('interpolated_sea.nc', 'r')
wind_data = nc.Dataset('interpolated_wind.nc', 'r')

print("file load complete")

times_2005=drifter_2005.variables['times'][:]
lons_2005=drifter_2005.variables['lons'][:]
lats_2005=drifter_2005.variables['lats'][:]
drifter_times = pd.DataFrame(times_2005)
drifter_lons = pd.DataFrame(lons_2005)
drifter_lats=pd.DataFrame(lats_2005)
drifter_times=drifter_times*24

for i in range(31):
    var=f"drifter{i+1}"
    globals()[var]=pd.DataFrame({'time':drifter_times.loc[i] , 'lons':drifter_lons.loc[i], 'lats' : drifter_lats.loc[i]}).dropna()

drifter_2005.close()

print("drifter complete")

############################################
uo_sea = sea_data.variables['uo'][:]
vo_sea = sea_data.variables['vo'][:]
uo_wind = wind_data.variables['u10'][:]
vo_wind = wind_data.variables['v10'][:]

print("data load complete")
############################################

latitude_vals = sea_data.variables['latitude'][:]
longitude_vals = sea_data.variables['longitude'][:]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# 시간의 최대 범위를 정의합니다.
max_time_range = sea_data.dimensions['time'].size

def get_sea_wind_data(time, lon, lat):
    if time >= max_time_range:
        return [np.nan, np.nan, np.nan, np.nan]  # 시간 범위를 벗어날 경우 NaN을 반환합니다.

    lat_idx = find_nearest_idx(latitude_vals, lat)
    lon_idx = find_nearest_idx(longitude_vals, lon)
    
    uo_s = uo_sea[time, 0, lat_idx, lon_idx]
    vo_s = vo_sea[time, 0, lat_idx, lon_idx]
    
    uo_w = uo_wind[time, lon_idx, lat_idx]
    vo_w = vo_wind[time, lon_idx, lat_idx]

    return [uo_s, vo_s, uo_w, vo_w]

combined_data = []



for i in range(31):
    drifter = globals()[f"drifter{i+1}"]
    for index, row in drifter.iterrows():
        time, lon, lat = int(row['time']), int(row['lons']), int(row['lats'])
        uo_s, vo_s, uo_w, vo_w = get_sea_wind_data(time, lon, lat)
        combined_data.append([time, lon, lat, uo_s, vo_s, uo_w, vo_w])

df = pd.DataFrame(combined_data, columns=['time', 'lons', 'lats', 'uo_sea', 'vo_sea', 'uo_wind', 'vo_wind'])
df.dropna(inplace=True)  # NaN 값을 가진 행을 제거합니다.

print(df.isnull().sum())

print("dataframe complete")

############################################


# 2. 데이터 스케일링 및 전처리
columns_to_scale = df.columns.difference(['uo_sea', 'vo_sea'])
scaler = MinMaxScaler()
scaled_data = df.copy()
scaled_data[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print(scaled_data.isnull().sum())

X = scaled_data.iloc[:, :-2].values  # 변경된 부분
y = scaled_data.iloc[:, -2:].values  # 변경된 부분

print("scaling complete")

# 시퀀스 데이터 형태로 변환
X = X.reshape((X.shape[0], 1, X.shape[1]))

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# NaN 값을 가진 행을 제거합니다.
# 이 때, 3차원의 X_train과 X_test 배열을 고려해야 합니다.
mask_train = ~np.isnan(X_train).any(axis=2).squeeze()
mask_test = ~np.isnan(X_test).any(axis=2).squeeze()

X_train = X_train[mask_train]
y_train = y_train[mask_train].astype('float32')
X_test = X_test[mask_test]
y_test = y_test[mask_test].astype('float32')

print("split complete")

# 데이터 내 NaN과 inf의 유무를 체크합니다.
for data, name in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    print(f"NaN in {name}: ", np.any(np.isnan(data)))
    print(f"Inf in {name}: ", np.any(np.isinf(data)))

############################################

# 3. LSTM 모델 구성 및 학습
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(2))
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

print("model complete")
############################################

# 4. 예측
y_pred = model.predict(X_test)

print("predict complete")
############################################

# # (Optional) 예측 결과 스케일링 되돌리기
# y_pred_original_scale = scaler.inverse_transform(np.hstack((X_test[:, 0, :-2], y_pred)))
# predicted_lons_lats = y_pred_original_scale[:, -2:]
