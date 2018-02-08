# Paper-study

Davian Lab 논문 스터디를 위한 저장소입니다.

## 2018-02-07
* 최성재

    Building Earth Mover’s Distance on Bilingual Word Embeddings for Machine Translation

* 최진호

    YOLO9000: Better, Faster, Stronger

## 2018-01-31
* 강경필

  Latent Relational Metric Learning via Memory-based Attention for Collaborative Ranking
  
* 최민석

  MobileNets: Efficient Convolutional Neural Networks for Mobile Vision

## 2018-01-24
* 정성효

  Input Convex Neural Networks(ICML 2017)
  
* 김태훈

  Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models

## 2018-01-17
* 박기태

  Adversarial attack
  
* 방효진
  
  Multi-scale Dense Networks for Resource Efficient Image Classification(MSDNet, ICLR 2018)

## 2018-01-10
* 최윤제
  
  Neural Speed Reading via Skim-RNN (ICLR 2018)
  
* 최민제

  Neural Hawkes Process


## 2018-01-03
* 최민석
  
  DeepEyes: Progressive Visual Analyticsfor Designing Deep Neural Networks
  
* 김민정

  Toward Controlled Generation of text

## 2017-12-27
* 조원웅

  Toward Multimodal Image-to-Image Translation

## 2017-12-20
* 최성재
  
  Decoding with Value Networks for Neural Machine Translation 

* 정성효

  Deep Image Prior

## 2017-12-13
* 최진호
  
  Neural scene derendering (2017CVPR)

* 강경필

  A Simple but Tough-to-Beat Baseline for Sentence Embeddings(ICLR 2017)
 
## 2017-12-06
* 김태훈

  Word Translation Without Parallel Data

* 최태균

  An Online Sequence-to-Sequence Model Using Partial Conditioning

## 2017-11-29

* 임지수

  Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning

* 최윤제

  StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

## 2017-11-22

* 박기태

  TextureGAN: Controlling Deep Image Synthesis with Texture Patches

* 효진

  An End-to-End Spatio-Temporal Attention Model for Human Action Recognition from Skeleton Data

## 2017-11-15
* 최민제

  Dynamic Routing Between Capsules

## 2017-11-08
* 조원웅

  Robust Image Sentiment Analysis Using Progressively Trained and Domain Transferred Deep Networks

* 김민정

  Gated Word-Character Recurrent Language Model

## 2017-11-01
* 최민석
  
  Synthesized Classifiers for Zero-Shot

* 최성재

  Deep  Compositional Question Answering with Neural Module Networks 
  
## 2017-10-25
* 최진호
  
  Get To the Point: Summarization with Pointer-Generator Networks
  
* 강경필
  
  supervised word mover's distance
  
## 2017-10-18
* 김태훈
    
    [Learning to remember rare events](https://www.dropbox.com/s/opfp8zb0f2wtoy8/Learning%20to%20remember%20rare%20events_%EA%B9%80%ED%83%9C%ED%9B%88_20171018.pdf?dl=0)
    - Memory matrix를 통한 one shot learning 학습
    - cnn의 fully connected layer의 output을 key로 memory update
    - omniglot data을 통한 실험
    - Extended Neural GPU(Synthetic task), GNMT(translation)등을 사용한 실험
   
* 정성효

  An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

## 2017-10-11
* 최태균

  DiSAN: Directional Self-Attention Network for RNN/CNN-free Language Understanding 입니다.

* 임지수

  The Conditional Analogy GAN: Swapping Fashion Articles on People Images

## 2017-09-20
* 박기태

  Learning Transferable Architectures for Scalable Image Recognition
  
* 최민제

  Patient Subtyping via Time-Aware LSTM Networks

## 2017-09-13
* 김민정

  Constituent-Centric Neural Architecture for Reading Comprehension
  
## 2017-09-06
* Kyeongpil Kang
    [Factorized Variational Autoencoders for Modeling Audience Reactions to Movies](https://drive.google.com/open?id=0B5pbHg6gugiOalNrbUpyQm81UlU)
    - Tensor factorization과 Variational Autoencoder를 결합함
    - 각 사람에 대해, 각 시간에 대한 latent vector z를 생성하고 이를 decoder에서 image(face landmark) 복원


## 2017-08-30

## 2017-08-24
* JinHo Choi

    [Learning Cooperative Visual Dialog Agents With Deep Reinforcement](https://www.dropbox.com/s/27my4nn6v0yigjw/Learning%20Cooperative%20Visual%20Dialog%20Agents%20with%20Deep%20Reinforcement.pptx?dl=0)
    - RL을 사용해 두 에이전트 봇끼리 자연어로 대화하면서 guessing game을 풀도록 함
    - 두 봇에게 각각 한정된 단어만 사용하도록 했을때 사람의 개입 없이 나름대로 comunication protocol 을 생성해서 대화를 함

* Sunghyo Chung

    [Explaining and Harnessing Adversarial Examples](https://www.dropbox.com/s/4vn6yy3ak7i84td/Explaining%20and%20Harnessing%20Adversarial%20Examples.pdf?dl=0)
    - 딥러닝 모델이 adverarial example에 취약한 이유를 모델의 nonlinear한 속성 때문이 아닌 linear한 속성(piecewise linear)때문에 나타남을 예시를 통해 보임
    - ReLU등의 activation unit은 sigmoid에 비해 의도적으로 linear하게 디자인 되어 optimize가 쉽지만 작은 perturbation에도 민감하게 반응하게 됨
    - Adverarial example 생성하기 위해 "fast gradient sign method"를 제시함


## 2017-08-17
* Choi Taekyoon

    [End-to-end Neural Coreference Resolution](https://www.dropbox.com/s/ftfd9uwlbbkakgf/End2end_neural_coreference_resolution.pdf?dl=0)
    - Coreference Task에 대해서 End-to-end로 접근하여 해결하고자 하는 방법입니다.
    - Span Representation을 활용하여 mention을 detect하고 각 mention간의 antecedent score를 통해 coreference 관계를 찾습니다.
    - 모든 span의 경우에 대한 coreference를 보지않고 mention score를 통해 일부 span에 대해서만 관계를 보게하여 computation 성능을 최적화하고자 합니다.

* Kim Teahun

    [Sparse Composite Document Vectors using soft clustering over distributional representations](https://www.dropbox.com/s/r2l12nue0we4zrj/SCDV%20-%20Sparse%20Composite%20Document%20Vectors%20using%20soft%20clustering%20over%20distributional%20representations_%EA%B9%80%ED%83%9C%ED%9B%88_20170816.pdf?dl=0)
    - Doc2Vec을 생성하는 새로운 모델(SCDV)에 대한 논문입니다.
    - skip-gram, Gaussian Mixture Model, Sparsity를 활용하여 skip gram의 Word2Vec에
    Topic별 발생 확율을 곱해 doc2vec을 생성한 모델입니다.

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


## 2017-07-27
* Taehun Kim

    [Word Embedding Based Correlation Model for Question Answer Matching - AAAI 2017](https://www.dropbox.com/s/pf9yrewgejjbu2t/Word%20Embedding%20Based%20Correlation%20Model%20for%20Question%20Answer%20Matching_%EA%B9%80%ED%83%9C%ED%9B%88_20170727.pdf?dl=0)
    - Q&A task에 CNN을 적용한 논문이었고 CNN input으로 question word와 answer workd의
    cosine simularity matrix를 사용함. input 의 simularity값을 좀 더 의미있게?
    계산하기 위해 answer word들을 question word들로 translation시키는 matrix M 을 쓴다는 점도 특이함
