import netCDF4 as nc

# .nc 파일 열기
nc_file = nc.Dataset('learn_data/{파일이름}', 'r')

# 변수와 속성 정보를 저장할 파일 이름
output_file = '{저장파일}.txt'

# 파일을 쓰기 모드('w')로 열기
with open(output_file, 'w', encoding='utf-8') as file:
    # 변수 목록 저장
    variables = nc_file.variables.keys()
    file.write("변수 목록:\n")
    for var_name in variables:
        file.write(f"{var_name}\n")

    # 파일 내의 차원 목록 확인
    dimensions = nc_file.dimensions.keys()
    file.write("차원 목록:\n")
    for dim_name in dimensions:
        file.write(f"{dim_name}\n")

    # 변수의 자세한 정보 저장
    for var_name in variables:
        var = nc_file.variables[var_name]
        file.write(f"\n변수 '{var_name}':\n")
        file.write(f"자료형: {var.dtype}\n")
        file.write(f"차원: {var.dimensions}\n")
        #file.write(f"단위: {var.units}\n")
        file.write("... 기타 정보 ...\n")

# .nc 파일 닫기
nc_file.close()

print(f"변수 및 속성 정보가 '{output_file}' 파일에 저장되었습니다.")