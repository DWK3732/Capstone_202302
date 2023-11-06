import pandas as pd
import folium


for index in range(7):

    # 1. csv 파일 읽기
    data = pd.read_csv(f'predictions_{index}.csv')

    # 2. 필요한 데이터 추출
    lons = data['lons'].values
    lats = data['lats'].values
    times = data['time'].values

    # 3. 지도에 데이터 표시하기
    # 초기 지도 위치와 확대 정도 설정 (예시: 첫 번째 데이터를 중심으로)
    m = folium.Map(location=[lats[0], lons[0]], zoom_start=10)

    # 각 위치에 마커와 선 추가하기
    for lat, lon, time in zip(lats, lons, times):
        folium.Marker([lat, lon], tooltip=f"Time: {int(time)} hours since 2016-01-01 00:00").add_to(m)
    folium.PolyLine(list(zip(lats, lons)), color="blue", weight=2.5).add_to(m)

    # 4. html 파일로 저장하기(interactive map)
    m.save(f'map_{index}.html')
