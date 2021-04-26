# Accented speech recognition

## 연구의 목적
지난 수십년 간 컴퓨터 기술이 많이 발전하였으며 특히 딥러닝의 부흥은 음성인식기술의 비약적인 발전을 불러일으켰다.  
이러한 발전에도 불구하고 감정, 억양, 소음 등이 섞인 특정 발화에 대해서는 좋은 성능을 보이지 못하는 것이 음성인식기술의 현 주소이다.  
따라서 본 연구에서는 그 중에서도 억양이 섞인 발화에 대해 성능을 높이고자 하였다.  

<br/>

## 연구 기간

- 석사학위 논문 주제로 연구를 진행하였다.
- 연구 기간 : 2020.04 ~ 2021.04

<br/>

## 기술스택
- Python 3.7.7
- Pytorch 1.6.0
- Cuda 10.1.243
- Ubuntu 18.04.5 LTS

<br/>

## 적용한 기법

1. Domain Adversarial Neural Network (DANN)
- DANN은 domain adaptation 기법 중에 하나로 source domain과 target domain이 적대적으로 학습이 되면서 성능향상이 일어나는 모델이다.  
- DANN은 input data의 도메인을 잘 구별하지 못하게 하면서 source domain과 target domain의 특성의 차이를 줄이는 것을 목표로 한다.  
- DANN은 다음과 같이 3개의 sub-network들로 이루어져있다.  

    (1) Feature extractor : feature extractor는 input data에서 모델을 학습시키기 위해 유용한 추상적인 특징들을 추출한다.  
    추출된 특징들은 다음 단계의 sub-network들인 domain classifier와 label predictor에 동시에 전달이 된다.  

    (2) Domain classifier : domain classifier는 input data의 domain을 구별한다.  
        DANN이 source domain과 target domain의 특성의 차이를 줄이는 것을 목표로 하기 때문에 domain classifier는 input data의 domain을 올바르게 구별하지 않아야 한다.  
        이는 domain classifier의 최하단에 위치한 Gradient Reversal Layer (GRL)를 통해서 실현이 가능하다. GRL은 DANN에서 핵심적인 역할을 수행한다.   
        GRL은 back-propagation과정에서 domain classifier의 손실값을 반전시켜서 전달을한다. 따라서 다음학습 시에 feature extractor가 domain-invariant한 특징을 만들어 낼 수 있도록 도와준다.  

    (3) Label predictor : label predictor는 input data의 label을 예측한다. 본 연구에서는 영어 알파벳 A ~ Z, space, apostrophe까지 총 28개의 label 중에서 예측을 한다.  

<br/>

2. Connectionist Temporal Classification (CTC)
- CTC는 end-to-end speech recognition 분야에서 LAS와 쌍벽을 이루는 중요한 모델이다.  
- CTC는 

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
- Mozilla의 [Common voice](https://commonvoice.mozilla.org/ko) 6.1버전을 사용하였다.
- Common Voice는 60개 언어의 음성데이터를 오픈소스로 제공한다.  
- Common Voice의 참여자들은 음성파일을 제공할 때 그들의 억양을 같이 알려주기 때문에 억양음성인식에서 유용하게 사용될 수 있다.  

#### 억양
- 실험을 위하여 5개의 Engliah accent가 사용되었다.  
- Source domain : US accent (US)  
- Target domain : Australia accent (AU), Canada accent (CA), England accent(EN), India accent (IN)

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
실험을 유연하게 진행하기 위하여 Goldwave를 사용하여 모든 음성파일들을 wav포맷으로 바꾸었으며 포맷을 바꿀 때 모든 파일들은 16,000HZ에 모노타입으로 인코딩되었다.

---

### 특징추출
pyaudio를 사용하여 128 size의 mel-spectrogram들을 추출하여 input feature로 사용하였다.

## 실험결과


### 실험결과 분석

## 문제 해결
1. Out Of Memory (OOM)