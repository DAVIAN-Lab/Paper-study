# Paper-study

Davian Lab 논문 스터디를 위한 저장소입니다.



## 2017-07-27
* Taehoon Kim

  [Word Embedding Based Correlation Model for Question Answer Matching - AAAI 2017](https://www.dropbox.com/s/pf9yrewgejjbu2t/Word%20Embedding%20Based%20Correlation%20Model%20for%20Question%20Answer%20Matching_%EA%B9%80%ED%83%9C%ED%9B%88_20170727.pdf?dl=0)  
  - Q&A task에 CNN을 적용한 논문이었고 CNN input으로 question word와 answer workd의
    cosine simularity matrix를 사용함. input 의 simularity값을 좀 더 의미있게?
    계산하기 위해 answer word들을 question word들로 translation시키는 matrix M 을 쓴다는 점도 특이함


## 2017-08-03
* Kyeongpil Kang

	[Quasi-Recurrent Neural Netowrks](https://drive.google.com/open?id=0B5pbHg6gugiOM1BoR1liN1lwcEE)
	
	- RNN 방법에 CNN 방법을 접목 시킴
	- 먼저 전체 time step에 대해 convolution을 하고 time step에 따라 liear multiplication을 없애 속도를 빨리 함
	- 해당 모델에 따른 Attention과 sequence to sequence 방법도 제시
	- 기존 LSTM에 비해 속도도 빠르면서 성능도 개선, 더 긴 sequence에 대해서 학습 가능성

	[Recurrent Models of Visual Attention](https://drive.google.com/open?id=0B5pbHg6gugiORWVDYlFjRGtITkE)
	
	- 전체 사진을 보지 말고 매 스텝마다 일부 패치를 보고 다음 스텝에서 해당 패치에서 어느 위치의 패치 뽑아서 인식할지를 학습
	- controller는 RNN, Reinforcement learning으로 학습