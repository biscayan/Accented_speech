# Accent classification
CNN, LSTM, GRU를 이용하여 English accent 분류 및 성능 비교

<br/>

## 연구의 목적
억양이 섞인 발화는 표준발화에 비해 인식을 하기가 많이 어렵다.  
input data의 억양이 무엇인지 분류를 해낼 수 있다면, 억양정보를 활용하여 음성인식의 성능을 높일 수 있다.  
따라서 본 연구에서는 CNN, LSTM, GRU 모델을 사용하여 input data의 accent들을 분류하였고 각각의 모델의 성능을 비교하였다.  
 
<br/>

## 연구 기간

- 대학원 3학기 음성언어기술주제연구 과목 수강 중에 연구를 진행하였다.
- 연구 기간 : 2020.04 ~ 2020.06 

<br/>

## 기술스택
(구글 코랩 환경에서 실험 진행)
- Python
- Pytorch
- Cuda

## 사용한 모델
1. CNN [코드 확인](https://github.com/biscayan/Accented_speech/blob/master/accent_classification/exp_termpaper/code/CNN/model.py#L5)
2. LSTM [코드 확인](https://github.com/biscayan/Accented_speech/blob/master/accent_classification/exp_termpaper/code/LSTM/model.py#L5)
3. GRU [코드 확인](https://github.com/biscayan/Accented_speech/blob/master/accent_classification/exp_termpaper/code/GRU/model.py#L5)

<br/>

## 실험세팅
### 코퍼스
- Mozilla의 [Common voice](https://commonvoice.mozilla.org/ko) 4.0 버전을 사용하였다.
    - Common Voice는 60개 언어의 음성데이터를 오픈소스로 제공한다.  
    - Common Voice의 참여자들은 음성파일을 제공할 때 그들의 억양을 같이 알려주기 때문에 억양음성인식에서 유용하게 사용될 수 있다.  

#### 억양
- 실험을 위하여 5개의 Engliah accent가 사용되었다.  
    - Australia accent (AU), Canada accent (CA), England accent(EN), India accent (IN), US accent (US)

#### 데이터셋

1. Training set

    |Dataset|Region|Files|
    |:---:|:---:|:---:|
    |AU-train|Australia|3,000|
    |CA-train|Canada|3,000|
    |EN-train|England|3,000|
    |IN-train|India|3,000|
    |US-traint|US|3,000|

</br>

2. Development set

    |Dataset|Region|Files|
    |:---:|:---:|:---:|
    |AU-dev|Australia|1,000|
    |CA-dev|Canada|1,000|
    |EN-dev|England|1,000|
    |IN-dev|India|1,000|
    |US-dev|US|1,000|

</br>

3. Test set

    |Dataset|Region|Files|
    |:---:|:---:|:---:|
    |AU-test|Australia|1,000|
    |CA-test|Canada|1,000|
    |EN-test|England|1,000|
    |IN-test|India|1,000|
    |US-test|US|1,000|

</br>

---

### 전처리
코퍼스를 다운받았을 때, 모든 음성파일들이 mp3포맷을 가지고 있었다.  
실험을 유연하게 진행하기 위하여 Goldwave를 사용하여 모든 음성파일들을 wav포맷으로 바꾸었으며 포맷을 바꿀 때 모든 파일들은 16,000HZ에 모노타입으로 인코딩되었다.

---

### 특징 추출
- librosa모듈을 이용하여 13차원의 MFCC feature들을 추출하여 input feature로 사용하였다. 
- 모든 MFCC feature들은 32ms의 window size를 가지며 16ms로 frame shift가 일어났다. 

<br/>

## 실험결과
accent 분류의 정확도를 기준으로 하여 모델의 성능을 평가하였다.  

1. CNN model  

    |Model|CNN layer|FC layer|Epoch|Batch size|Learning rate|Accuracy|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |CNN01|1|1|150|256|0.00001|87%|
    |**CNN02**|**1**|**2**|**150**|**256**|**0.00001**|**88%**|
    |CNN03|2|1|150|256|0.00001|81%|
    |CNN04|2|2|150|256|0.00001|83%|
    |CNN05|3|1|150|256|0.00001|76%|
    |CNN06|3|2|150|256|0.00001|77%|

<br/>

2. LSTM model

    |Model|LSTM layer|Hidden unit|Epoch|Batch size|Learning rate|Accuracy|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |LSTM01|1|50|150|256|0.0001|34%|
    |LSTM02|1|100|150|256|0.0001|44%|
    |LSTM03|1|150|150|256|0.0001|64%|
    |LSTM04|2|50|150|256|0.0001|42%|
    |LSTM05|2|100|150|256|0.0001|47%|
    |**LSTM06**|**2**|**150**|**150**|**256**|**0.00001**|**66%**|

<br/>

3. GRU model

    |Model|GRU layer|Hidden unit|Epoch|Batch size|Learning rate|Accuracy|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |GRU01|1|50|150|256|0.0001|64%|
    |GRU02|1|100|150|256|0.0001|70%|
    |GRU03|1|150|150|256|0.0001|69%|
    |GRU04|2|50|150|256|0.0001|68%|
    |GRU05|2|100|150|256|0.0001|72%|
    |**GRU06**|**2**|**150**|**150**|**256**|**0.00001**|**75%**|

<br/>

---

### 실험결과 분석
CNN, LSTM, GRU모델들의 accent분류 성능을 비교했을 때, CNN02 모델에서 88%의 정확도로 가장 좋은 성능을 보였다.  
LSTM모델의 최고성능은 66%, GRU모델의 최고성능은 75%로 CNN모델에 비해서 현저하게 낮았다.  
따라서 classfication 문제에서는 CNN 모델이 좋은 성능을 가지는 것으로 보인다.  
하지만 LSTM모델과 GRU모델은 모델이 복잡해질수록 높은 성능을 보이고 있지만, CNN모델은 모델이 복잡해질수록 오히려 성능이 낮아지고 있다.  
이러한 현상은 어떻게 설명할 수 있을까?  
본 연구에서 사용된 데이터의 양이 많지 않은데 이러한 데이터들이 LSTM과 GRU모델의 수용력을 충분히 만족시키지 못 할 수 있다.  
따라서 후속 연구에서 데이터의 양을 더 늘려서 실험을 진행해보도록 하겠다.   

<br/>

## 문제 해결