# Accent classification

## 연구의 목적

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
2. LSTM
3. GRU

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

|Dataset|Accent|Number of files|
|:---:|:---:|:---:|
|Training set|AU|3000|
|Training set|CA|3000|
|Training set|EN|3000|
|Training set|IN|3000|
|Training set|US|3000|
|Dev set|AU|1000|
|Dev set|CA|1000|
|Dev set|EN|1000|
|Dev set|IN|1000|
|Dev set|US|1000|
|Test set|AU|1000|
|Test set|CA|1000|
|Test set|EN|1000|
|Test set|IN|1000|
|Test set|US|1000|

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
    |LSTM01|1|50|150|256|0.00001|34%|
    |LSTM02|1|100|150|256|0.00001|44%|
    |LSTM03|1|150|150|256|0.00001|64%|
    |LSTM04|2|50|150|256|0.00001|42%|
    |LSTM05|2|100|150|256|0.00001|47%|
    |**LSTM06**|**2**|**150**|**150**|**256**|**0.00001**|**66%**|

<br/>

3. GRU model

    |Model|GRU layer|Hidden unit|Epoch|Batch size|Learning rate|Accuracy|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |GRU01|1|50|150|256|0.00001|64%|
    |GRU02|1|100|150|256|0.00001|70%|
    |GRU03|1|150|150|256|0.00001|69%|
    |GRU04|2|50|150|256|0.00001|68%|
    |GRU05|2|100|150|256|0.00001|72%|
    |**GRU06**|**2**|**150**|**150**|**256**|**0.00001**|**75%**|

<br/>

---

### 실험결과 분석

## 문제 해결