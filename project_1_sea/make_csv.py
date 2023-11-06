#################################################################################
########################
# 0. 라이브러리 불러오기 #
########################
import pandas as pd
import netCDF4 as nc
import numpy as np
import xarray as xr

# 단계별 확인을 위한 함수
def print_boxed_text(text):
    lines = [line for line in text.splitlines() if line]  # 공백 문자열 제거
    max_len = len(max(lines, key=len))
    border = "#" * (max_len + 4)
    print(border)
    for line in lines:
        print(f"# {line.ljust(max_len)} #")
    print(border)





#################################################################################
########################
# 1. 2005년 뜰개 데이터 #
########################

# 1-1. 데이터 불러오기
drifter_2005=nc.Dataset('learn_data/NOAA_drifter_2005.nc','r')

# 1-2. 데이터 변수 지정
times_2005=drifter_2005.variables['times'][:]
lons_2005=drifter_2005.variables['lons'][:]
lats_2005=drifter_2005.variables['lats'][:]

# 1-3. 데이터 프레임으로 변환
drifter_times = pd.DataFrame(times_2005)
drifter_lons = pd.DataFrame(lons_2005)
drifter_lats=pd.DataFrame(lats_2005)

# 1-4. 시간 단위를 일단위 에서 시단위로 변환(기존 times에서는 1 = 1일단위)
drifter_times=drifter_times*24

# 1-5. 뜰개 각각을 이름으로하는 31개의 데이터 프레임 생성
for i in range(31):
    var=f"drifter{i+1}"
    globals()[var]=pd.DataFrame({'time':drifter_times.loc[i] , 'lons':drifter_lons.loc[i], 'lats' : drifter_lats.loc[i]}).dropna()



##################################################################
# 특정 drifter의 데이터 프레임 형식(1단계)                          #
# time	lons	lats                                             #
##################################################################
boxed_text = """
특정 drifter의 데이터 프레임 형식(1단계)
time    lons    lats
"""
print_boxed_text(boxed_text)

#################################################################################
#################################
# 2. 2005+2006년 표층 해류 데이터 #
#################################

# 2-1. 6시간단위로 처리된 데이터 불러오기
dataset = xr.open_dataset('interpolated_sea.nc', engine='netcdf4')

# 2-2. 뜰개 데이터에 일치하는 데이터 추출
for i in range(31):
    # 2-2-1. 뜰개 이름으로 뜰개의 정보 불러오기
    var=f"drifter{i+1}"
    if var in globals():
        drifter = globals()[var]
        uo_list=[]
        vo_list=[]
        
        # 2-2-2. 불러온 뜰개의 데이터를 하나씩 불러와서 uo, vo값을 추출
        for j in range(len(drifter)):
            ## 뜰개의 정보를 input_ 에 저장하기
            input_time_hours=drifter.loc[j]['time'] # 시간을 datetime 형식으로 변환
            base_time = np.datetime64('2005-01-01T00:00:00') # 2005년 1월 1일 0시를 기준으로 설정
            input_time = base_time + np.timedelta64(int(input_time_hours), 'h') # 기준 시간에 시간을 더해줌
            input_longitude = drifter.loc[j]['lons']  # 경도 설정
            input_latitude = drifter.loc[j]['lats']   # 위도 설정

            ## 불러온 뜰개의 정보를 바탕으로 uo, vo값을 추출
            # 가장 가까운 uo 찾기
            nearest_point = dataset['uo'].sel(
            time=input_time,
            longitude=input_longitude,
            latitude=input_latitude,
            method='nearest'
            )
            uo_value = nearest_point.item()
            uo_list.append(uo_value)

            # 가장 가까운 vo 찾기
            nearest_point = dataset['vo'].sel(
            time=input_time,
            longitude=input_longitude,
            latitude=input_latitude,
            method='nearest'
            )
            vo_value = nearest_point.item()
            vo_list.append(vo_value)
            
    # 2-3. 뜰개 데이터에 uo, vo값 추가
    drifter['uo']=uo_list
    drifter['vo']=vo_list

##################################################################
# 특정 drifter의 데이터 프레임 형식(2단계)                         #
# time	lons	lats	uo	vo                                   #
##################################################################
boxed_text = """
특정 drifter의 데이터 프레임 형식(2단계)
time	lons	lats	uo	vo
"""
print_boxed_text(boxed_text)

#################################################################################
#################################
# 3. 2005+2006년 표층 바람 데이터 #
#################################

# 3-1. 6시간단위로 처리된 데이터 불러오기
dataset = xr.open_dataset('interpolated_wind.nc', engine='netcdf4')

# 3-2. 뜰개 데이터에 일치하는 데이터 추출
for i in range(31):
    # 3-2-1. 뜰개 이름으로 뜰개의 정보 불러오기
    var=f"drifter{i+1}"
    if var in globals():
        drifter = globals()[var]
        u10_list=[]
        v10_list=[]

        # 3-2-2. 불러온 뜰개의 데이터를 하나씩 불러와서 u10, v10값을 추출
        for j in range(len(drifter)):
            input_time_hours=drifter.loc[j]['time']# 시간을 datetime 형식으로 변환
            base_time = np.datetime64('2005-01-01T00:00:00')# 2005년 1월 1일 0시를 기준으로 설정
            input_time = base_time + np.timedelta64(int(input_time_hours), 'h')# 기준 시간에 시간을 더해줌
            input_longitude = drifter.loc[j]['lons'] # 경도 입력
            input_latitude = drifter.loc[j]['lats']   # 위도 입력

            ## 불러온 뜰개의 정보를 바탕으로 u10, v10값을 추출
            # 가장 가까운 u10 찾기
            nearest_point = dataset['u10'].sel(
            time=input_time,
            longitude=input_longitude,
            latitude=input_latitude,
            method='nearest'
            )
            u10_value = nearest_point.item()
            u10_list.append(u10_value)

            # 가장 가까운 v10 찾기
            nearest_point = dataset['v10'].sel(
            time=input_time,
            longitude=input_longitude,
            latitude=input_latitude,
            method='nearest'
            )
            v10_value = nearest_point.item()
            v10_list.append(v10_value)

    # 3-3. 뜰개 데이터에 u10, v10값 추가
    drifter['u10']=u10_list
    drifter['v10']=v10_list

##################################################################
# 특정 drifter의 데이터 프레임 형식(3단계)                         #
# time	lons	lats	uo	vo	u10	v10                          #
##################################################################
boxed_text = """
특정 drifter의 데이터 프레임 형식(3단계)
time	lons	lats	uo	vo	u10	v10
"""
print_boxed_text(boxed_text)

#################################################################################
###################################################
# 4. drifter에 6시간 뒤의 정보 추가하기(정답 레이블) #
###################################################

for i in range(31):
    var = f"drifter{i+1}"
    next_time = globals()[var]['time'][1:]
    next_lons = globals()[var]['lons'][1:]
    next_lats = globals()[var]['lats'][1:]
    next_data = pd.DataFrame({'next_time': next_time, 'next_lons': next_lons, 'next_lats': next_lats})
    next_data = next_data.reset_index()
    globals()[var] = globals()[var].drop(globals()[var].index[-1])
    globals()[var] = pd.concat([globals()[var], next_data],axis=1)
    globals()[var] = globals()[var].drop('index', axis=1).dropna()

###########################################################################
# 특정 drifter의 데이터 프레임 형식(4단계)                                  #
# time	lons	lats	uo	vo	u10	v10	next_time	next_lons	next_lats #
###########################################################################
boxed_text = """
특정 drifter의 데이터 프레임 형식(4단계)
time	lons	lats	uo	vo	u10	v10	next_time	next_lons	next_lats
"""
print_boxed_text(boxed_text)

#################################################################################
######################################################
# 5. drifter들을 합쳐서 하나의 데이터 프레임으로 만들기 #
######################################################

# 5-1. 공백 리스트 생성
dataframes = []

# 5-2. 공백 리스트에 각각의 drifter 데이터 프레임을 추가
for i in range(1, 32):
    var = f"drifter{i}"
    dataframes.append(globals()[var])

# 5-3. 공백 리스트에 추가된 drifter 데이터 프레임을 하나의 데이터 프레임으로 합치기
data_drifter = pd.concat(dataframes, ignore_index=True)

###########################################################################
# data_drifter의 데이터 프레임 형식(5단계)                                  #
# time	lons	lats	uo	vo	u10	v10	next_time	next_lons	next_lats #
###########################################################################
boxed_text = """
data_drifter의 데이터 프레임 형식(5단계)
time	lons	lats	uo	vo	u10	v10	next_time	next_lons	next_lats
"""
print_boxed_text(boxed_text)

#################################################################################
###############################################
# 6. data_drifter 데이터 프레임을 csv로 저장하기#
###############################################

data_drifter.to_csv('data_drifter.csv', index=False)

print("data_drifter.csv 저장 완료")