# Accented_speech
Several experiments for accent classification and speech recognition
## Accent classification
### Data
Common Voice dataset is used for the experiment.  
dataset is provided by Mozilla and it is registered as open source. [Common voice](https://commonvoice.mozilla.org/ko)  
The purpose of the dataset is providing useful data for ASR studies. The dataset is comprised of dozens of languages from a widely used language such as English, to a little bit unfamiliar language such as Tamil.  
Among them, English corpus is used in this study.  
The dataset is collected from the volunteers. They recorded their speeches and these are forwarded to Mozilla.  
Furthermore, they can self-report their age, gender and accent. Thus, experiments about accent can be done by using the dataset.  
In these experiments, 5 accents are chosen: Australian English (AU), Canadian English (CA), England English (EN), Indian English (IN) and United States English (US).  
### Experimental setups
All experiments are done with pytorch which is one of the famous deep learning framework.  
Before extracting input features, all speech files are converted into wav files from mp3 files using Goldwave.  
Goldwave can convert the format of files easily and it can also determine sampling rate.  
The files are sampled at 16000HZ which is widely used for processing speeches.  
As the input features, 13-dimensional Mel Frequency Cepstral Coefficient (MFCC)s are extracted using librosa module provided by python.  
In detail, MFCCs are extracted with window size of 32ms and frame shift of 16ms.  