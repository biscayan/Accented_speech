# Accented speech recognition
CTC와 DANN을 이용하여 accented speech recognition의 성능 개선

<br/>

## 연구의 목적
지난 수십 년 간 컴퓨터 기술이 많이 발전하였으며 특히 딥러닝의 부흥은 음성인식기술의 비약적인 발전을 불러일으켰습니다.  
이러한 발전에도 불구하고 감정, 억양, 소음 등이 섞인 특정 발화에 대해서는 좋은 성능을 보이지 못하고 있습니다.  
그중에서도 억양이 섞인 발화는 표준 발화와 비교했을 때 언어학적인 차이가 존재하여 제대로 인식을 하기가 어렵습니다.  
따라서 본 연구에서는 CTC와 DANN을 적용하여 억양음성인식의 성능을 높였습니다.  

<br/>

## 연구 기간

- 석사학위 논문 주제로 연구를 진행하였습니다.  
- 연구 기간 : 2020.04 ~ 2021.04

<br/>

## 기술스택
- Python 3.7.7
- Pytorch 1.6.0
- Cuda 10.1.243
- Ubuntu 18.04.5 LTS

<br/>

## 적용한 기법

**1. Domain Adversarial Neural Network (DANN)**
- DANN은 domain adaptation 기법의 하나로 source domain data와 target domain data가 적대적으로 학습이 됩니다. 
- 즉, DANN은 input data의 도메인을 잘 구별하지 못하게 하면서 source domain data와 target domain data의 특성의 차이를 줄이는 것을 목표로 합니다.  
- DANN은 다음과 같이 3개의 sub-network들로 이루어져 있습니다.  

    (1) **Feature extractor** 
    - Feature extractor는 input data에서 모델 학습에 유용한 추상적인 특징들을 추출합니다.  
    - 추출된 특징은 다음 단계의 sub-network들인 domain classifier와 label predictor에 동시에 전달이 됩니다.  

    (2) **Domain classifier** 
    - Domain classifier는 input data의 domain을 구별합니다.  
    - DANN이 source domain data와 target domain data의 특성의 차이를 줄이는 것을 목표로 하기 때문에 domain classifier는 input data의 domain을 올바르게 구별하지 않도록 해야 합니다.  
    - Domain classifier의 최하단에 위치한 Gradient Reversal Layer(GRL)는 DANN에서 핵심적인 역할을 수행합니다.   
    - GRL은 back-propagation 과정에서 domain loss를 반전시켜서 feature extractor로 전달합니다. 따라서 다음 학습 시에 feature extractor가 domain-invariant한 특징을 만들어 낼 수 있도록 도와줍니다.  
    - GRL은 back-propagation 과정에서만 활성화되며 forward-propagation 중에는 파라미터값을 변경하지 않습니다.   

    (3) **Label predictor** 
    - Label predictor는 input data의 label을 예측합니다.  
    - 본 연구에서는 영어 알파벳 A ~ Z, space, apostrophe까지 총 28가지 중에서 label을 예측합니다.  

<br/>

**2. Connectionist Temporal Classification (CTC)**
- CTC는 end-to-end speech recognition 분야에서 LAS와 쌍벽을 이루는 중요한 모델입니다.  
- CTC는 input data와 label을 자동으로 매치하기 때문에 데이터를 pre-segment 할 필요가 없습니다.  
- CTC는 결과 label을 바로 얻을 수 있기 때문에 후처리 작업이 필요 없고 음성인식 과정을 간편화시켜 매우 유용합니다.  

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
- 실험을 위해서 Mozilla의 [Common voice](https://commonvoice.mozilla.org/ko) 6.1 버전을 사용하였습니다.  
- Common Voice는 60개 언어에 대해서 막대한 양의 검증된 음성데이터를 오픈소스로 제공합니다.  
- Common Voice의 참여자들은 음성파일을 제공할 때 그들의 억양을 같이 알려주기 때문에 억양음성인식에서 유용하게 사용될 수 있습니다.  

#### 억양
- 본 연구에서는 accented speech recognition 실험을 위해서 5개의 English accent가 사용되었습니다.  
- 5개의 English accent는 source domain과 target domain으로 나누어졌습니다.
    - Source domain : US accent (US)  
    - Target domain : Australia accent (AU), Canada accent (CA), England accent(EN), India accent (IN)

#### 데이터셋

- 실험에 사용된 데이터셋은 아래와 같습니다.  

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

### 전처리

- 코퍼스를 다운받았을 때, 모든 음성파일이 mp3포맷을 가지고 있었습니다. Goldwave를 사용하여 모든 음성파일을 wav포맷으로 바꾸었고 16,000HZ에 모노 타입으로 인코딩하였습니다.
- 코퍼스의 transcript파일에서 실험에 필요한 정보들만을 추출하여 새로운 csv파일로 만들어 실험에 사용하였습니다.

</br>

### 특징추출
- pytorch의 torchaudio 라이브러리를 이용해서 128 size의 mel-spectrogram들을 추출하였고 input feature로 사용하였습니다.  

</br>

## 실험결과
- 실험결과는 Character Error Rate(CER)와 Word Error Rate(WER) 두 개의 metric을 이용하여 성능을 평가하였습니다.  
- 각각의 표에서 실험결과는 CER/WER의 쌍으로 기재되어있습니다.

1. Baseline 모델과 DANN 모델의 실험결과를 비교  
    - Baseline-src : 16만 개의 source domain data로만 학습시킨 baseline 모델  
    - Baseline-src-tgt : 16만 개의 source domain data와 2만 개의 target domain data로 학습시킨 baseline 모델  
    - DANN-base : 16만 개의 source domain data와 2만 개의 target domain data로 학습시킨 DANN 모델  

        |Accent|Baseline-src|Baseline-src-tgt|DANN-base|
        |:---:|:---:|:---:|:---:|
        |AU|28.85% / 65.95%|25.09% / 61.06%|**24.22% / 59.80%**|
        |CA|14.76% / 43.16%|13.77% / 40.68%|**13.55% / 40.15%**|
        |EN|25.49% / 61.53%|24.43% / 59.78%|**21.60% / 54.67%**|
        |IN|36.41% / 76.03%|30.52% / 69.41%|**28.83% / 66.49%**|

</br>

2. 기존의 DANN 모델과 target domain data의 양을 증가시킨 DANN 모델과의 실험결과를 비교  
    - DANN-base : 16만 개의 source domain data와 2만 개의 target domain data로 학습시킨 DANN 모델  
    - DANN-inc : 16만 개의 source domain data와 target domain data를 추가하여 학습시킨 DANN 모델  
    총 target domain data의 수 : AU->3만 7천 개, CA->3만 2천 개, EN->4만 개, IN->4만 개  

        |Accent|DANN-base|DANN-inc|
        |:---:|:---:|:---:|
        |AU|24.22% / 59.80%|**22.72% / 57.38%**|
        |CA|13.55% / 40.15%|**13.63% / 40.02%**|
        |EN|21.60% / 54.67%|**19.96% / 52.53%**|
        |IN|28.83% / 66.49%|**26.24% / 63.56%**|

</br>

3. EN data의 양을 증가시키면서 학습시킨 DANN 모델들의 실험결과를 비교  
    - DANN-EN1 : 16만 개의 source domain data와 2만 개의 EN data로 학습시킨 DANN 모델  
    - DANN-EN2 : 16만 개의 source domain data와 4만 개의 EN data로 학습시킨 DANN 모델   
    - DANN-EN3 : 16만 개의 source domain data와 6만 개의 EN data로 학습시킨 DANN 모델   
    - DANN-EN4 : 16만 개의 source domain data와 8만 개의 EN data로 학습시킨 DANN 모델   
    - DANN-EN5 : 16만 개의 source domain data와 10만 개의 EN data로 학습시킨 DANN 모델   

        |Accent|DANN-EN1|DANN-EN2|DANN-EN3|DANN-EN4|DANN-EN5|
        |:---:|:---:|:---:|:---:|:---:|:---:|
        |EN|21.60% / 54.67%|19.96% / 52.53%|18.65% / 49.91%|17.15% / 47.32%|**16.35% / 45.68%**|

</br>

### 실험결과 분석
- Baseline 모델과 DANN 모델을 비교했을 때, 모든 accent의 DANN 모델에서 성능향상이 일어났으므로 Domain Adversarial Training(DAT)이 source domain data와 target domain data의 차이를 효과적으로 줄였음을 확인하였습니다.  
- 하지만 accent마다 성능향상 정도의 차이는 존재했는데 CA accent에서는 성능향상이 미미했고 EN, IN accent에서는 성능향상이 크게 일어났습니다.  
- EN, IN accent가 source domain data로 사용된 US accent와는 언어학적으로 큰 차이가 있는 반면에 CA accent는 US accent와 언어학적으로 비슷하기 때문에 성능향상 정도의 차이가 발생하였습니다.  
- target domain data의 양에 따른 성능향상을 확인하기 위하여 추가실험들도 진행하였는데 target domain data가 추가될수록 성능향상이 꾸준히 일어났음을 확인하였습니다.  

</br>

## 문제 해결
- Out Of Memory (OOM)  
실험에 사용되는 데이터의 양이 많아짐에 따라 OOM 문제가 발생하였고 더 이상 실험이 진행되지 않았습니다.  

    (1) 우선, GPU 메모리 부족 문제는 병렬처리를 통하여 해결하였습니다. 하지만, 연구소에서 다른 인원들도 GPU를 사용해야 하고 최대로 사용할 수 있는 GPU 메모리의 양은 한정되어 있어서 연구소 인원들과 상의 후 사용할 GPU의 번호를 지정하여 원활히 실험을 진행할 수 있었습니다.  

    (2) 그다음으로 CPU ram에서도 메모리 부족 문제가 발생하였습니다. 연구소에서 사용할 수 있는 ram의 총 용량은 130G 정도로 많은 양임에도 불구하고 메모리 부족 문제가 발생하여 의아했습니다. garbage collect, delete 등 여러 시도를 해보았지만, 여전히 문제가 해결되지 않았습니다.  
    몇 일 동안 고심을 하던 와중에 dataset을 작성한 코드를 천천히 살펴보았고 데이터프레임을 읽어올 때 메모리 누수가 발생하는 것을 알게 되었습니다. 따라서 dataset 코드를 [코드 확인](https://github.com/biscayan/Accented_speech/blob/master/speech_recognition/code/experiment/cv_dataset.py) 수정하고 정상적으로 실험을 진행할 수 있었습니다.  