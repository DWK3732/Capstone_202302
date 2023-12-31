import pandas as pd

data = {
    'Year': [2016, 2016, 2016, 2016, 2016, 2016, 2016],
    'Month': [3, 3, 3, 3, 7, 7, 7],
    'Day': [13, 15, 19, 19, 15, 18, 20],
    'Hour': [11, 20, 12, 12, 15, 12, 1],
    'Minute': [49, 15, 1, 38, 40, 48, 49],
    'Longitude': [127 + 4.310/60, 126 + 36.949/60, 129 + 16.916/60, 129 + 12.744/60, 127 + 4.468/60, 129 + 17.834/60, 129 + 23.897/60],
    'Latitude': [32 + 30.442/60, 33 + 45.559/60, 34 + 56.978/60, 34 + 59.200/60, 32 + 30.976/60, 34 + 55.305/60, 37 + 33.155/60]
}

df = pd.DataFrame(data)

# DataFrame을 CSV 파일로 저장
df.to_csv('drifter_2016.csv', index=False)