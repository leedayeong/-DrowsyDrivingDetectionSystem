# Drowsy Driving Detection System - 졸음 운전 방지 시스템
## Drowsy Driving Detection System with OpenCV &amp; KNN

<br>

## 🚀 목차
- [🚀 목차](#-목차)
- [📝 프로젝트 개요](#-프로젝트-개요)
- [💡 주요 기능](#-주요-기능)
- [💡 상세 동작](#-상세-동작)
- [💡 시연](#-시연)
- [⚙️ 기술 스택](#️-기술-스택)


<br>

## 📝 프로젝트 개요
![개요](/uploads/107bae7b2ec7a56a995fce0eef969a1d/개요.png)
* <strong>진행 기간</strong>: 2020.08.05 ~ 2020.09.31
* <strong>목표</strong>
  * 졸음 운전을 판단합니다.
  * 졸음 운전의 정도에 따라 그에 맞는 알람을 울리도록 설계하였습니다. 
<br>


## 💡 주요 기능

![시스템 구성도](/image/system.png)

> 전처리 기능
- Gray Scaling : 얼굴 및 눈 검출에서 빠른 영상 처리를 위해 이미지를 그레이 스케일링 시킴(Luma 기법)
- Light Processing : 영상에 있어서 조명의 영향은 영상처리에 상당히 많은 영향을 끼침. 따라서 영상에서 조명 영향을 받을 때 그 영향을 최소화하는 작업을 진행함
  - 1. LAB 모델로 L채널 분리
  - 2. Median Filtering : 앞의 과정에서 얻은 명도 값은 실제 조명 상태와 차이가 있기 때문에 실제 조명 상태의 명도 얻음
  - 3. 결과 역상으로 실제 조명상태의 반대 명도 값을 얻음
- Gray Scaling 한 이미지와 Light Processing을 통해 조명에 의해 얻은 반대 명도 값을 합쳐줌
- Clear Image : 조명상태를 배재한 흑백 이미지를 얻음


> Face & EYE Detection
 - HOG(Histogram of Oriented Gradient) Face Pattern : 얼굴 검출
 - Landmark Estimation : 눈 검출

> Drosy Detection
- dlib : EAR(Eyes aspect Ratio) : 눈의 비율을 이용해 눈 감김을 확인
- TTS (Text to Speech) : 졸음 상태인 경우 알람을 울림


> Drosy Level Selection
- KNN (K-Nearest Neighbor) : 졸음 상태를 3단계로 구분

<br>

## 💡 상세 동작

![상세 동작](/image/detail.png)

<br>
## 💡 시연

![시연](/image/test.png)


## ⚙️ 기술 스택
- Library : OpenCV, KNN, dlib
  
- Language : Python3

- OS : Raspbian

- Hardware : Raspberry Pi, Pi Camera

<br>

