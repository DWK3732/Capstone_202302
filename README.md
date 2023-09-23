# Capstone_202302

## python 가상환경 설정하기
'''python
python -m venv myvenv
source myvenv/Scripts/activate
'''

## windows의 경우 powershell권한을 관리자로 실행하여 다음 명령어 실행
'''bash
Set-ExecutionPolicy RemoteSigned
'''
Y로 대답하면 됨

## python 필요한 패키지 설치하기
'''python
pip install -r requirements.txt
'''

## python 가상환경에 사용된 패키지 목록 저장하기
'''python
pip freeze > requirements.txt
'''

## python 가상환경 종료하기
'''python
deactivate
'''