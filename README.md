[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/1b159e9516ca802fafc0cf1d1e26aea4)

# HK Toss MLOps 중간 프로젝트

🏠 주택청약 당첨가점 예측 ML 모델 개발

### 팀원
- [🔗](https://github.com/eugeneee0126) 이유진
- [🔗](https://github.com/Joo-an) 이주안
- [🔗](https://github.com/) 정혜진
- [🔗](https://github.com/) 한예은
- [🔗](https://github.com/choikwangil95) 최광일

### 목차
- [1 프로젝트 개요](#1-프로젝트-개요)
- [2 프로젝트 구조](#2-프로젝트-구조)
- [3 개발환경 구성](#3-개발환경-구성)
- [4 개발 프로세스](#4-개발-프로세스)

## 1 프로젝트 개요

### 문제정의
- 청약 가입자 약 2,700만 명, 이 중 1순위 가입자 약 1,000만 명
- 당첨 가점은 청약 종료 후에 공개되며, 청약 매물과 공급면적에 따라 20~80점까지 다양함
- 무분별한 청약 신청은 당첨 후 포기 시 불이익 발생
- 당첨 가점을 사전에 예측하여 합리적인 청약 신청 유도

### 주요기능
- T.B.D

### 데이터셋
- [🔗](https://www.data.go.kr/data/15101046/fileData.do) 주택청약 지역별 분양정보
- [🔗](https://www.data.go.kr/data/15126242/fileData.do) 주택청약 가점정보
- [🔗](https://www.data.go.kr/data/15101048/fileData.do) 주택청약 경쟁률
- [🔗](https://www.data.go.kr/data/15110975/fileData.do) 주택청약 지역별, 연령별 신청자 정보
- [🔗](https://www.data.go.kr/data/15110976/fileData.do) 주택청약 지역별, 연령별 당첨자 정보
- [🔗](https://www.data.go.kr/data/15088657/fileData.do) 주택청약 지역별, 연령별, 통장별 가입현황
- [🔗](https://www.data.go.kr/data/15088656/fileData.do) 주택청약 통장 가입기간별 가입현황
  
## 2 프로젝트 구조
### 폴더구조
```markdown
📁 src
 ㄴ 📁 datasets             # 원본 데이터셋
 ㄴ 📁 features             # 학습할 데이터셋
 ㄴ 📁 models               # 학습된 모델
 ㄴ 📄 baseline.ipynb       # 모델 학습 코드
📄 requirements.txt
```

### 아키텍쳐

- [🔗](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=ko#mlops_level_0_manual_process) 구글 MLOps 아키텍쳐

![image](https://github.com/user-attachments/assets/e8b38089-6776-4a21-8bb4-51cc1eaa441a)


## 3 개발환경 구성
### 기술 스택
- **언어**: Python 3.11
- **패키지 관리**: Anaconda / Miniconda

### 프로젝트 설정

```bash
# 1 프로젝트 폴더 생성 및 저장소 초기화
mkdir <folder_name>
cd <folder_name>
git init

# 2 저장소 복제 및 동기화
git remote add origin https://github.com/choikwangil95/HKToss-MLOps-Proejct.git
git pull origin main

# 3 가상환경 설정
conda create -n <env_name> python=3.11 pip
conda activate <env_name>

# 4 Jupyter Notebook 커널 설정
conda install ipykernel --update-deps --force-reinstall

# 5 requirements 설치
pip install -r requirements.txt
```

## 4 개발 프로세스

<img src="https://github.com/user-attachments/assets/ce06d476-6f07-4209-bf8e-3739d2801e9b" width="600px"/>

### 브랜치 관리
- `main` : 운영 환경
- `develop` : 개발 환경
- `feature` : 기능 개발

### 작업 흐름

```bash
# 1 최신 develop 브랜치 동기화
git checkout develop
git pull origin develop

# 2 새로운 기능 브랜치 생성
git checkout -b <feature_branch>

# 3 작업 후 변경 사항 저장
git add .
git commit -m "커밋 메시지"

# 4 develop 브랜치 병합 (충돌 확인 필수)
git checkout develop
git pull origin develop
git merge <feature_branch>

# 5 원격 저장소 반영
git push origin develop
```
