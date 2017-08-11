# Paper-study

Davian Lab 논문 스터디를 위한 저장소입니다.
## 2017-08-09
* Keetae Park

  [A Persona-Based Neural Conversation Model](https://drive.google.com/open?id=0B9xAbjSTQ9G1SmZIWWU4VmRWdFE)  
	- Standford  박사인 Jiwei Li와 Microsoft가 합작하여 연구한 논문으로, 2016년에 ACL에서 발표되었습니다. 
	- 이 연구분야는 NLP에서 Neural Conversation Models 혹은 Neural Dialogue Generation으로 불리며, Open-domain Chatbot과 personal assistant와 같은 Conversation AI가 궁극적인 목표입니다
	- 본 논문에서는 기존에 사용하던 Seq2seq + MLE의 알고리즘의 Decoder에서 각 단어를 예측할 때 사용자 정보를 의미하는 Vi 벡터를 삽입함으로써, 기존의 generic하며 dull한 응답을 개선하고 chatbot에도 character를 입힐 수 있다는 점을 보였습니다.
	- Vi 벡터를 삽입하는 방법으로 Speaker Model과 Speaker-Addressee Model 등 두 모형을 제시하는데, Speaker Model은 응답자의 성격만 고려하는데 반하여 Speak-addressee Model은 질문자와 응답자의 관계 또한 고려하여 서로 다른 대화상대자에 따라 달라지는 자세를 반영할 수 있게 하였습니다.
	- 저자는 단지 성격을 입히는 것 이상으로 나아가서 generic하고 vague한 응답을 수정하는 방안을 제시하는데, 그것이 N-best list를 특정 score를 기준으로 shuffle하는 것입니다. Score는 응답되는 sequence의 길이와 input message의 likelihood에 Penalty를 거는 방식으로 설정합니다.
	- 본 논문은 기존의 딱딱하던 발화자들에게 어느 정도 character 또는 persona를 씌워줬다는 점에서 의의를 가진다고 생각됩니다

* Jisu Lim

	[End-to-end Learning of Image based Lane-Change Decision](https://www.dropbox.com/s/fr45shissiydjcg/naver_labs.pdf?dl=0)
	- NAVER_LABS에서 안전하게 차선 변경을 하기 위한 알고리즘 논문입니다.
	  http://www.naverlabs.com/showroom/autonomous_driving.html
	  
	  위 사이트는 네이버 랩스 홈페이지 입니다.
	  차량에 부착 된 2개의 카메라를 사용하여 찍힌 사진의 모습을 보고 차선변경이 가능할 경우 'FREE', 불가능할 경우 'BLOCKED', 모호할 경우 'UNDEFINED'의 레이블을 추가합니다.
	  VGG 16 에 수집된 이미지들을 학습시켜 차량이 안전하게 차선 변경이 가능한지의 여부를 학습니다.


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

* Seongjae Choi
	
	[Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203)

	- 기존 Model-Free RL에 Model-Based 개념을 적용함으로 써 Agent가 상상 및 추론을 하게 함
	- 상상으로 만들어진 Predicted Observation, Predicted Reward에서 RNN(LSTM)을 이용해 정보를 추출하고 이를 optimal policy를 찾는 곳에 활용 함
	- Sokoban 게임에서, 기존 standard model(A3C)가 60% 정도의 성능을 낸 것에 반해, 87% 성능을 확보 함
	- Model-Based planning Method인 MCTS와 결합해 사용할 경우 computation cost는 18배 정도 줄어들었음


