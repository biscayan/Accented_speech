# Accented speech recognition

## 연구의 의의

## 연구의 목적

<br/>

## 연구 기간

- 석사학위 논문 주제로 연구를 진행
- 연구 기간 : 2020.04 ~ 2021.04

<br/>

## 기술스택
- Python 3.7.7
- Pytorch 1.6.0
- Cuda 10.1.243
- Ubuntu 18.04.5 LTS

<br/>

## 모델

1. Baseline model [코드 확인](https://github.com/biscayan/Accented_speech/blob/master/speech_recognition/code/experiment/asr_model.py#L70)

<img src="https://user-images.githubusercontent.com/56914074/115953413-459a1c00-a526-11eb-97da-d8ff8b7be2e4.PNG" width="50%">

</br>
</br>

2. DANN model [코드 확인](https://github.com/biscayan/Accented_speech/blob/master/speech_recognition/code/experiment/asr_model.py#L103)

<img src="https://user-images.githubusercontent.com/56914074/115942643-34c9b600-a4e6-11eb-81eb-84a0f28e96f0.PNG" width="50%">

</br>
</br>

## 실험세팅
### 코퍼스
- Mozilla의 [Common voice](https://commonvoice.mozilla.org/ko) 6.1버전을 사용

#### 데이터셋

1. Training set

|Dataset|Region|Files|Hours|
|:---:|:---:|:---:|:---:|
|US-160k|US|160,000|196|
|AU-20k|Australia|20,000|26|
|AU-37k|Australia|37,0000|50|
|CA-20k|Canada|20,000|25|
|CA-32k|Canada|32,000|43|
|EN-20k|England|20,000|24|
|EN-40k|England|40,000|53|
|EN-60k|England|60,000|77|
|EN-80k|England|80,000|101|
|EN-100k|England|100,000|126|
|IN-20k|India|20,000|26|
|IN-40k|India|40,000|57|

</br>

2. Development set

|Dataset|Region|Files|Hours|
|:---:|:---:|:---:|:---:|
|AU-dev|Australia|2,000|3|
|CA-dev|Canada|2,000|3|
|EN-dev|England|2,000|3|
|IN-dev|India|2,000|3|

</br>

3. Test set

|Dataset|Region|Files|Hours|
|:---:|:---:|:---:|:---:|
|AU-test|Australia|2,000|3|
|CA-test|Canada|2,000|3|
|EN-test|England|2,000|3|
|IN-test|India|2,000|3|

</br>

---

### 전처리
코퍼스를 다운받았을 때, 모든 음성파일들이 mp3포맷을 가지고 있었다.  
실험을 유연하게 진행하기 위하여 Goldwave를 사용하여 모든 음성파일들을 wav포맷으로 바꾸었다.  
포맷을 바꾸면서 모든 파일들은 16,000HZ에 모노타입으로 인코딩되었다.

---

### 특징추출


## 실험결과


### 실험결과 분석

## 문제 해결
1. Out Of Memory (OOM)