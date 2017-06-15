---
layout: post
title: "ML 1.2 Supervised learning"
description: "Chapter 1. Introduction"
date: 2017-06-13
tags: [machine learning, probabilistic, statistics, data science]
comments: true
---

### 1.2 Supervised learning

ML 에서 가장 광범위하게 사용되는 Supervised Learning에 대해서 알아보자.

#### 1.2.1 Classification

Classification 의 목표는 inputs $$X$$ 에서 outputs $$y$$ 로의 mapping 을 학습시키는 것이다. $$y \in \{ 1, \dots , C\}$$ 이고, $$C$$ 는 classes 의 개수이다. 만약 $$C = 2$$ 라면, **binary classification** (이 경우 $$y \in \{ 0, 1 \}$$ ), 만약 $$C > 2$$ 이면 **multi classification**이라 부른다. 또 만약 클래스의 라벨이 not mutually exclusive 하다면 (예를들어 한 사람을 “키가크고 힘이세다” 라고 분류했다면), **multi-label classification** 이라고 부른다. (**multiple output model** 이라고도 함). 보통 우리가 “classification” 이란 용어를 쓸 때는 multiclass classification with a single output 임을 가정한다.

우리는 흔히 문제를 formalize (형상화) 하는 방법으로 **function approximation** 을 한다. 알지 못하는 function $$f$$ 에 대해 $$y = f(X)$$ 로 식을 만들고, 라벨링 되어있는 training set 으로 function $$f$$ 를 추정하기 위해 학습하는 것이 목표라 할 수 있다. 그리고 나서 $$\hat{y} = \hat{f(X)}$$ 을 사용하여 예측을 하는 것이다. 우리의 주 목표는 novel inputs (미지의 데이터) 로부터 예측을 하는 것인데, 이는 학습데이터 외에 우리가 한번도 보지 못했던 데이터를 의미한다. (이것을 **generalization**이라고 부른다). 만약 training set 에 대응되는 $$y$$ 값을 예측하는건 그냥 학습데이터에서 답을 찾으면 되는 것이므로 의미없다.

##### 1.2.1.1 Example

<center><img src="{{ baseurl }}/images/2017-03-08-ml-1-2-supervised-learning-1.png"></center>

> **Figure 1.1** Left: 라벨링 된 컬러 모양들의 training examples, 그리고 3개의 unlabeled test cases.  
> Right: $$N \times D$$ 형태의 training data. row $$i$$ 는 feature vector $$X_i$$ 를 표현한다. 마지막 colum Label 은 $$y_i \in \{0, 1\}$$ 이다.

위의 그림 Figure 1.1(a) 를 보면, 0과 1로 라벨링 된 two classes of object 를 볼 수 있다. 여기서 inputs 은 colored shapes 이며, $$D$$ 개의 features들로 구성된 $$N \times D$$ matrix $$X$$ 이다. (Figure 1.1(b)) input feature $$X$$ 는 discrete (이산형) 일 수도, continuous (연속형) 일 수도 있으며, 둘 다일 수도 있다.

Figure 1.1 에서 test set 에는 파란 초승달, 노란 원 그리고 파란 화살표가 있다. 이들 모두 이전에는 보지 못했던 조합이다. (학습데이터에 똑같은 것이 없다). 그러므로 우리는 학습데이터를 뛰어넘어 **generalize** 를 할 필요가 있다. 합리적인 추측을 한다면 파란 초승달은 $$y=1$$ 이 되어야 한다. 왜냐하면 모든 파란색 모양들은 학습데이터에서 라벨링이 1이기 때문이다. 노란 원은 분류하기 좀 까다로운데, 어떤 노란색 모양들은 $$y=1$$ 이고, 또 어떤 것들은 $$y=0$$ 이기 때문이다. 게다가 어떤 원 모양들은 $$y=1$$ 이며 $$y=0$$ 이기도 한다. 결론적으로 노란색 원의 경우 올바르게 라벨링하기 어렵고 그 기준이 명확하지 않다. 파란 화살표도 비슷한 문제를 가진다.

##### 1.2.1.2 The need for probabilistic predictions

위의 노란 원 처럼 모호한 케이스를 다루기 위해 우리는 확률이론으로 돌아가야 한다.

우리는 가능한 라벨에 대한 확률 분포를 input vector $$X$$ 와 training set $$D$$ 를 사용해 $$p(y \vert X, D)$$ 로 표현할 수 있다. 일반적으로 카테고리값 $$C$$ 에 의해 벡터의 길이가 정해진다. (만약 클래스가 2개 뿐이라면 $$p(y=1 \vert X, D)$$ 하나의 값으로 충분히 표현할 수 있다. 왜냐하면 $$p(y=1 \vert X, D) + p(y=0 \vert X, D) = 1$$ 이기 때문이다.) 이 표기법에서 주의해야 할 부분은 test input $$X$$ 과 training set $$D$$ 의 조건부 확률이란 점이다. ( conditioning bar $$\vert$$ 로 표시한다. ) 우리는 또한 예측을 하기 위해 모델 ( $$M$$ )도 $$p(y \vert X, D, M)$$ 처럼 조건부 term 에 추가하여야 하는데, 보통 문맥에서 모델이 명확하게 보이는 경우는 생략한다.

우리는 주어진 probabilistic output 을 바탕으로, 항상 “true label” 에 대한 “best guess” 를 계산할 수 있다.

<center>$$\hat{y} = \hat{f(X)} = \overset{C}{\underset{c=1}{\text{argmax}}}~p(y=c \vert X, D)$$</center>

이것은 distribution $$p(y \vert X, D)$$ 의 **mode** 이며, 클래스 라벨 중 가장 확률이 높은 것을 선택한다는 의미이다. **MAP estimate** (MAP; **maximum a posteriori**) 로 잘 알려져 있다.

