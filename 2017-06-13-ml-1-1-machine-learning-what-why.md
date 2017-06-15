---
layout: post
title: "ML 1.1 Machine learning: what and why?"
description: "Chapter 1. Introduction"
date: 2017-06-13
tags: [machine learning, probabilistic, statistics, data science]
comments: true
---

머신러닝을 본격적으로 배운지 벌써 1년이 되어 간다. 짧다면 짧은 3개월이란 시간 동안 미친듯이 배우고, 소화하려고 노력했었다. 그리고 배움을 증명하고자 나에게 주는 훈장처럼 산 책이 **Kevin P. Murphy** 의 **[Machine Learning - A Probabilistic Perspective](https://www.amazon.com/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020)** 이다. 하루에 한 챕터씩 읽자고 배짱있게 주문했던 책이었으나, 구매한지 약 10달이 지난 지금 아직 Ch2도 못 끝냈다.

<center><img src="{{ baseurl }}/images/2017-03-06-ml-1-1-introduction-1.jpg"></center>

그래서 오늘부터 딱딱해진 머리를 다시 말랑말랑하게 하는 작업을 하려한다. 책은 총 28챕터로 각 대주제 챕터에는 최소 4개에서 최대 8개의 작은 소주제 챕터로 구성되어 있다. 소주제가 평균 6개라 가정한다면 총 168개의 챕터가 있다는 말이다. 일주일에 최소 1챕터라도 쓰는게 목표인데 이 책을 모두 정리하여 글쓰기까지 약 168주 ( = 168 / 52 = 3.23년 )가 걸리겠다. 그래도 시작이 반이다 하지 않는가.

---

## 1.1 Machine learning: what and why?

<center><mark>"We are drowning in information and starving for knowledge. - John Naisbitt."</mark></center>

우리는 빅데이터 시대에 들어섰다. 그 예로 현재 약 1 trillion (1조) web pages 가 존재하고 있고, (2008년 기준) 유투브에는 매 초 한 시간의 동영상이 업로드 되며, 사람의 1000개의 유전체는 $$3.8 \times 10^9$$ 짝의 길이를 가지는 데이터로 구성되어 있다.

**Machine learning**은 이러한 데이터의 홍수에서 automated methods of data analysis를 의미한다. 특히 우리는 machine learning을 데이터 속에서 자동으로 패턴을 감지하고, 미래의 데이터를 예측하기 위해 밝혀지지 않은 패턴을 사용하고, 또는 불확실성 아래 어떠한 의사결정을 수행하기 위한 일련의 방법론 (a set of methods) 이라 정의한다.

본 책에서 주요 관점은 **Probability theory (확률론)**이 문제를 풀기 위한 가장 좋은 방법이라는 것이다. Probability theory 는 불확실성을 포함한 문제에서 언제든지 적용가능하다. 머신러닝에서 불확실성은 다양한 형태로 나타난다: 주어진 과거 데이터로부터 무엇이 가장 미래를 잘 예측하는가? 무엇이 데이터를 가장 잘 설명하는 최적의 모델인가? 어떠한 측정법을 수행해야 하는가? 등. 확률론적 머신러닝 접근법은 통계학 분야와 아주 밀접하게 관계되어 있다. 단지 emphasis and terminology (강조점과 용어들)만 미세하게 다를 뿐이다.

앞으로 우리는 다양한 데이터와 업무에 맞는 다양한 확률론적 모형들을 배울 것이며, 또한 이러한 모형들을 사용하고 학습시키는 다양한 알고리즘들도 배울 것이다. 최종적인 목표는 확률론적 모델링과 추론에 대한 통합된 시각을 갖는 것이라 말할 수 있다. 비록 대부분의 사람들은 어마어마한 데이터셋을 다루는 방법론과 computational efficiency 에 주의를 기울일 테지만, 그것들은 다른 책에서 더 잘 다뤄줄 것이므로 이 책에선 생략한다.

또한 우리가 중요하게 봐야할 것은 그렇게 어마어마한 데이터셋을 다룰지라도, 우리가 관심을 가지는 the effective number of data points 는 꽤나 작다는 것이다. 사실 다양한 도메인을 거친 데이터는 **long tail**이라 불리우는 영역을 내보인다. words와 같이 작은 것들은 아주 공통적이지만, 대부분의 모든 것들은 아주 생소한 것들을 의믜한다(Pareto 법칙). 이것은 이 책에서 다루는 small samples sizes로부터 일반화 되어진 핵심 통계학적 이슈는 빅데이터 분야와 아주 밀접하게 관계되있다는 것을 의미한다.

* * *

### 1.1.1 Types of machine learning

머신러닝은 대게 두 종류의 주요 타입으로 나뉜다.

첫번째, **predictive** or **supervised learning** approach. 목표는 입력 데이터 $$X$$ 와 출력 데이터 $$y$$ 의 mapping을 학습하는 것이다. 그리고 입/출력 데이터는 라벨링이 되어져 있으며 $$D = \{(X_i, y_i)\}^N$$ 로 표현할 수 있다. 여기서 $$D$$ 는 **training set**이라 불리며, $$N$$ 은 number of training examples 이다.

training input $$X_i$$ 는 $$D$$-dimensional ( $$D$$-차원 )로 이뤄진 벡터이다. 가장 간단한 예로 사람의 키와 몸무게를 생각하면 된다. 그리고 이를 **features, attributes** or **covariates** 라 부른다. 그러나 일반적으로 $$X_i$$ 는 더 복잡한 구조의 object가 되곤 하는데, 예를 들어 이미지, 문장, 이메일 주소, 시계열 자료, 분자 모양, 그래프 등으로도 나타낼 수 있다.

output or **response variable** 도 input 데이터와 비슷하게 원칙적으로는 아무 값이나 될 수 있으나, 거의 대부분의 방법론들은 $$y_i$$ 가 유한한 범위의 **categorical** or **nominal** variable 이라고 가정한다. $$y_i \in \{1, \dots, C\}$$ (such as male or female), or that $$y_i$$ is a real-valued scalar (such as income level) $$y_i$$가 categorical 변수라면, **classification** or **pattern recognition** 문제가 되고, $$y_i$$ 가 real-valued 변수라면, **regression** 문제가 된다. 또 다른 변수에 대해 **ordinal regression** 문제가 있는데, 이는 $$y$$의 라벨 공간이 A-F 학점과 같이 순서에 의한 값일 경우이다 (ordinal-valued).

머신러닝의 두번째 주요 타입은 **descriptive** or **unsupervised learning** approach이다. 오로지 input 데이터만 가진다, $$D = \{X_i\}^N$$, 그리고 목표는 데이터 속에서 “interesting patterns”를 찾는 것이다. 이것은 때때로 **knowledge discovery**라고도 불린다. <del>이것은 상대적으로 보다 덜 정의된 문제이며</del>, 우리가 아직 이러한 종류의 패턴에 대해서 밝혀진게 많이 없어 명확한 error metric 도 없는 상황이다. (우리가 예측값을 주어진 관측값( $$y$$ )과 비교하는 방법을 쓰는 supervised learning 과 다르게)

마지막으로 세번째 타입은 <del>다소 덜 사용되는</del> **reinforcement learning(강화학습)** 이다. 이것은 적절한 reward와 punishment가 주어진 상황에서 어떻게 결정하고 행동할지에 대한 학습에 유용하다. 그러나 불행하게도, RL은 이 책에선 다루지 않는다.
