# HKToss-MLOps-Proejct
HK Toss MLOps 중간 프로젝트
- 이유진
- 이주안
- 정혜진
- 한예은
- [최광일](https://github.com/choikwangil95)

## 목차
- [1 프로젝트 개요](#1-프로젝트-개요)
- [2 프로젝트 구조](#2-프로젝트-구조)
- [3 개발환경 구성](#3-개발환경-구성)

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

## 2 프로젝트 구조
### 폴더구조
```markdown
- 📁 datasets
- 📁 features
- 📁 src
  - 📄 baseline.ipynb
- 📄 requirements.txt
```

### 아키텍쳐
- T.B.D

## 3 개발환경 구성
### 사전 요구사항
- Language: Python 3.11
- Pacakge manager: Anaconda / Miniconda

### git 설정

```bash
# git 저장소 초기화
git init

# git 원격 저장소 origin 추가
git remote add origin https://github.com/choikwangil95/HKToss-MLOps-Proejct.git

# git 로컬 저장소 동기화
git pull origin main
```

### 가상환경 설정
```bash
# 가상환경 생성
conda create -n <가상환경 이름> python=3.11 pip

# 가상환경 활성화
conda activate <가상환경 이름>
```

### 커널 설정
```bash
# jupyter notebook 커널 라이브러리 설치
conda install -n <가상환경 이름> ipykernel --update-deps --force-reinstall

# jupyter notebook 커널 설정
# 우측 상단 커널 선택 - Python Environments 선택 -  <가상환경 이름> 선택
```

### 패키지 설치
```bash
# 파이썬 패키지 설치
pip install -r requirements.txt
```
