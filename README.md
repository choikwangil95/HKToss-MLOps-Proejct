[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/1b159e9516ca802fafc0cf1d1e26aea4)

# HKToss-MLOps-Proejct
HK Toss MLOps 중간 프로젝트
- 이유진
- 이주안
- 정혜진
- 한예은
- 최광일 ([https://github.com/choikwangil95](https://github.com/choikwangil95))

## 목차
- [1 프로젝트 개요](#1-프로젝트-개요)
- [2 프로젝트 구조](#2-프로젝트-구조)
- [3 개발환경 구성](#3-개발환경-구성)
- [4 개발 프로세스](#4-개발-프로세스)

## 1 프로젝트 개요
주택청약 당첨가점 예측 모델 개발

![image](https://github.com/user-attachments/assets/6ad59110-0be8-4077-9479-3cbd20262a55)

### 문제정의
- 문제
  - 청약 가입자 약 2700만명
  - 청약 당첨가점은 청약 기간 종료 후, 당첨자 발표일에 확인 가능
  - 청약 매물별, 공급면적별 당첨 가점이 다름 (20~80점)
  - 청약 1순위 인원 약 1000만명임
  - 청약 당첨 후 포기시 불이익 존재 (그냥 막 신청하면 되는게 아님)
- 해결
  - 청약 당첨자 발표 전 당첨가점을 예측

### 주요기능
- T.B.D

### 데이터셋
- [청약홈) 주택청약 지역별 분양정보](https://www.data.go.kr/data/15101046/fileData.do#/tab-layer-file)
- [청약홈) 주택청약 가점정보](https://www.data.go.kr/data/15126242/fileData.do)
- [청약홈) 주택청약 경쟁률](https://www.data.go.kr/data/15101048/fileData.do)
- [청약홈) 주택청약 지역별, 연령별 신청자 정보](https://www.data.go.kr/data/15110975/fileData.do)
- [청약홈) 주택청약 지역별, 연령별 당첨자 정보](https://www.data.go.kr/data/15110976/fileData.do)
- [청약홈) 주택청약 지역별, 연령별, 통장별 가입현황](https://www.data.go.kr/data/15088657/fileData.do)
- [청약홈) 주택청약 통장 가입기간별 가입현황](https://www.data.go.kr/data/15088656/fileData.do)

## 2 프로젝트 구조
### 폴더구조
```markdown
📁 src # 소스코드
 ㄴ 📁 datasets           # 전체 데이터셋
 ㄴ 📁 features           # 학습할 데이터셋
 ㄴ 📁 models             # 학습된 모델
 ㄴ 📄 baseline.ipynb
📄 requirements.txt
```

### 아키텍쳐

[참고: 구글 MLOps 아키텍쳐](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=ko#mlops_level_0_manual_process)

![image](https://github.com/user-attachments/assets/e8b38089-6776-4a21-8bb4-51cc1eaa441a)


## 3 개발환경 구성
- 1 사전 요구사항
  - Language: Python 3.11
  - Pacakge manager: Anaconda / Miniconda

- 2 git 설정

```bash
# git 저장소 초기화
git init

# git 원격 저장소 origin 추가
git remote add origin https://github.com/choikwangil95/HKToss-MLOps-Proejct.git

# git 로컬 저장소 동기화
git pull origin main
```

- 3 가상환경 설정
```bash
# 가상환경 생성
conda create -n <가상환경 이름> python=3.11 pip

# 가상환경 활성화
conda activate <가상환경 이름>
```

- 4 커널 설정
```bash
# jupyter notebook 커널 라이브러리 설치
conda install -n <가상환경 이름> ipykernel --update-deps --force-reinstall

# jupyter notebook 커널 설정
# - 1 우측 상단 커널 선택
# - 2 Python Environments 선택
# - 3 <가상환경 이름> 선택
```

- 5 패키지 설치
```bash
# 파이썬 패키지 설치
pip install -r requirements.txt
```

## 4 개발 프로세스

<img src="https://github.com/user-attachments/assets/ce06d476-6f07-4209-bf8e-3739d2801e9b" width="600px"/>

- 0 브랜치 관리
  - `main`: 상용환경
  - `develop`: 개발환경

- 1 원격 브랜치 동기화
```bash
# develop 브랜치 원격 저장소 동기화
git pull origin develop
```
- 2 로컬 브랜치 생성
```bash
# develop 브랜치에서 신규 브랜치 생성
git checkout -b <브랜치 이름>
```
- 3 로컬 작업내용 저장
```bash
# 작업내용 staging에 추가
git add .

# 작업내용 local repogitory에 추가
git commit -m "<커밋 메세지>"
```
- 4 로컬 작업내용 병합
```bash
# develop 브랜치로 이동
git checkout develop

# 작업기간동안 다른 사람에 의해 변경되었을 develop 원격 브랜치 동기화
git pull origin develop

# develop 브랜치에 생성한 브랜치 로컬 작업내용 병합
git merge <브랜치 이름>
```
- 5 로컬 작업내용 원격 병합
```bash
git push origin develop
```

