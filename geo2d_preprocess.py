#데이터 전처리 하기
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# .nc 파일 열기
dataset = xr.open_dataset('learn_data/2005_01_sea.nc', engine='netcdf4')

# 필요한 변수 'vo'와 'uo' 추출
vo_data = dataset['vo']
uo_data = dataset['uo']


# 아래와 같이 수정하여 numpy datetime64 객체로 변환합니다.
start_time = vo_data['time'].min().values.astype('datetime64[ns]')
end_time = vo_data['time'].max().values.astype('datetime64[ns]')

# 이제 np.arange()를 사용하여 새로운 시간 배열을 생성합니다.
step = np.timedelta64(6, 'h')
new_time_np = np.arange(start_time, end_time, step)


new_time = da.from_array(new_time_np, chunks=1)  # Dask 배열로 변환

# Dask 배열을 사용하여 보간 수행
vo_interp = vo_data.interp(time=new_time_np, method='linear')
uo_interp = uo_data.interp(time=new_time_np, method='linear')


# 결과를 계산 및 저장
vo_interp = vo_interp.compute()
uo_interp = uo_interp.compute()

# 새로운 데이터셋 생성
new_dataset = xr.Dataset({'vo': vo_interp, 'uo': uo_interp})

# 저장할 파일 이름 설정
output_filename = 'interpolated_data.nc'

# 데이터셋을 .nc 파일로 저장
new_dataset.to_netcdf(output_filename)

# 파일을 닫습니다.
dataset.close()
