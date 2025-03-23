# HK Toss MLOps 중간 프로젝트

🏠 주택청약 당첨가점, 시세차익 예측 ML 모델 개발

[서비스 링크](https://hktoss-blue-jeans.streamlit.app/)

<img alt="스크린샷 2025-03-23 오후 8 07 13" src="https://github.com/user-attachments/assets/8dc2d344-9b84-4a33-a350-9547e3d4afb4" width="60%" />

### 팀원
- [🔗](https://github.com/eugeneee0126) 이유진
- [🔗](https://github.com/Joo-an) 이주안
- [🔗](https://github.com/yoo754) 정혜진
- [🔗](https://github.com/choikwangil95) 최광일
- [🔗](https://github.com/yeaaaun) 한예은

### 목차
- [1 프로젝트 개요](#1-프로젝트-개요)
- [2 프로젝트 구조](#2-프로젝트-구조)
- [3 개발환경 구성](#3-개발환경-구성)
- [4 개발 프로세스](#4-개발-프로세스)

## 1 프로젝트 개요

### 문제정의
- 문제
  - 청약 가입자 약 2,700만 명, 이 중 1순위 가입자 약 1,000만 명
  - 청약 당첨가점은 청약 종료 후에 공개되며, 청약 매물과 공급면적에 따라 20~80점까지 다양함
  - 무분별한 청약 신청은 당첨 후 포기 시 불이익 발생
- 해결
  - 당첨 가점을 사전에 예측하여 합리적인 청약 신청 유도
  - 청약 매물별 시세차익을 예측하여 추가 정보를 제공

### 주요기능
- 1 공고중인 청약 매물 목록 확인
<img width="600" alt="스크린샷 2025-03-23 오후 8 15 31" src="https://github.com/user-attachments/assets/08100519-2150-4185-8b1b-055167cf7acd" />

- 2 주택청약 당첨가점, 시세차익 예측
<img width="600" alt="스크린샷 2025-03-23 오후 8 15 43" src="https://github.com/user-attachments/assets/98391028-387c-45e0-854d-418f2a84b06c" />

- 3 예측값에 대한 설명을 제공
<img width="600" alt="스크린샷 2025-03-23 오후 8 15 54" src="https://github.com/user-attachments/assets/5c9d5d6e-fa5e-4940-a890-069ab9d59576" />
<img width="600" alt="스크린샷 2025-03-23 오후 8 16 00" src="https://github.com/user-attachments/assets/9fc5b5ff-fd3b-41d7-aba0-d1f6dad3078a" />

### 데이터셋
- [한국부동산원] 주택청약 지역별 분양정보, 가점정보, 경쟁률 (2020년~2025년)
- [청약홈] 주택청약 공급금액 (2020년~2025년)
- [국토교통부] 아파트 실거래가 (2020년~2025년)
- [한국은행] 기준금리
- [뉴스기사] 주택청약 매물별 뉴스기사 (2000개)
  
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
