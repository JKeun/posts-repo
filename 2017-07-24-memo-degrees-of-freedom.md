---
layout: post
title: "자유도 ( Degrees of Freedom )"
description: "Memo ( 1 ) - Degrees of Freedom"
date: 2017-07-24
tags: [memo, degrees of freedom, 자유도, SST, SSE, SSR, regression, statistics]
comments: true
---

<center><mark>"In statistics, the number of values in the final calculation of a statistic that are free to vary."</mark></center>
<br>

통계학의 여러 문제들을 풀기 위해선 [자유도 ( Degrees of Freedom )](https://en.wikipedia.org/wiki/Degrees_of_freedom_(statistics))를 결정해야 할 때가 있다. 자유도는 "계산에 사용되는 자유로운 데이터의 수" 라고 정의 한다. 그렇다면 도대체 자유로운 데이터는 무엇이며, 자유롭지 않은 데이터는 무엇이란 말인가? 자유도에 대한 개념을 설명하기 위해 항상 나오는 [평균 ( Mean )](https://en.wikipedia.org/wiki/Mean)을 구하는 예를 살펴보자.

##### AN ILLUSTRATION WITH A SAMPLE MEAN
우리 모두가 알듯이 평균은 모든 데이터를 더한 후, 데이터의 총 갯수만큼 나눠서 구할 수 있다. 그렇다면 예를 들어 우리가 갖고 있는 데이터의 평균이 $$25$$ 이고, 데이터의 값들이 $$20, 10, 50,$$ 그리고 하나의 *unknown value* 가 있다고 하자. 여기서 *unknown value* 를 구하는 방정식은 $$(20 + 10 + 50 + x) / 4 = 25$$ 으로 세울 수 있고, 이를 풀면 $$x = 20$$ 으로 결정*( determined )*된다.

그럼 더 나아가서 평균이 $$25$$ 이고, 데이터의 값들이 $$20, 10,$$ 그리고 두 개의 *unknown values* 가 있다고 하자. 이 두 *unknown values* 는 다를 수도 있기 때문에, 두 *different variables* 은 $$x, y$$ 로 정의한다. 그리고 방정식을 세우면 $$(20 + 10 + x + y) = 25$$ 이 되고, 이를 풀면 $$y = 70 - x$$ 가 된다. 이 식은 일단 우리가 $$x$$ 의 값을 선택하면, $$y$$ 의 값은 완전히 결정됨을 의미한다. 즉, 현재 우리는 ***한 번의 선택 ( one choice )*** 을 할 수 있고, 그것은 ***하나의 자유도 ( one degree of freedom )*** 가 생겼음을 의미한다.

그럼 더더더 나아가서 $$100$$개의 샘플을 갖고 있다고 가정하자. 만약 우리가 평균이 $$20$$ 이라는 것만 알고 나머지 데이터 값들은 모른다면, 그 때의 자유도는 $$99$$ 가 되고, 모든 값들을 다 더한다면 $$20 \times 100 = 2000$$ 이 될 것이다. 또한 우리가 $$99$$ 개의 데이터 값들을 안다면, 마지막 한 개의 값은 자동으로 결정될 것이다.


##### STANDARD DEVIATION
통계학을 배우면서 자유도의 개념을 가장 먼저 접하는 순간이 [표준편차 ( Stnadard deviation )](https://en.wikipedia.org/wiki/Standard_deviation)를 배울 때다. 그럼 먼저 ( 샘플 ) 표준편차의 식을 보면서 살펴보기로 하자.

$$s = \sqrt{\frac{1}{N-1} \sum (x_i - \bar{x})^2}, \quad df: n - 1$$

흔히 ( 샘플 ) 표준편차를 구할 때, 평균과 ( mean : $$\bar{x}$$ ) 각 관찰값들의 차이의 합을 $$N-1$$ 으로 나누어야 우리가 원하는 ( unbiased ) 표준편차가 나온다. ( 불편추정량에 대해서는 [Bias of an estimator](https://en.wikipedia.org/wiki/Bias_of_an_estimator) 를 참고하자. )

여기서 자유도가 $$n$$ 이 아니라 $$n-1$$ 인 이유는, $$n$$ 개의 관찰값 ( *unknown values* ) 과 평균 ( sample mean : $$\bar{x}$$) 이 위 공식에 쓰여졌기 때문이다. 즉, 표준편차를 구하기 위해 사용된 평균을 우리는 이미 알고, 이로 인해 $$1$$ 개의 관찰값은 자동으로 결정 ( *determined* ) 되기 때문에 $$n-1$$ 개의 자유도를 갖게 된 것이다.

**"다시말해 어떤 *수식값 ( 통계량 )* 을 찾기 위해 *사용된 parameter 의 갯수 ( $$k$$ )* 를 *샘플의 크기 ( $$N$$ )* 에서 빼준 것이 *자유도의 수 ( $$N-k$$ )* 가 된다."**

##### SUM OF SQUARES
회귀분석에서 $$R^2$$(결정계수) 를 구하거나 회귀모형의 $$F$$-검정을 할 때, 나오는 개념이 [제곱합(SUM OF SQUARES)](https://en.wikipedia.org/wiki/Partition_of_sums_of_squares) 이다.
<center><img src="https://image.slidesharecdn.com/linearregression-140903114216-phpapp01/95/linear-regression-22-638.jpg?cb=1409744639" width="400"></center>

- 결정계수 ( $$R^2$$; Coefficient of Determination ) : $$\frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$
- 총제곱합 ( SST; Total Sum of Squres ) : $$\sum{(Y_i - \bar{Y})^2}$$
- 오차제곱합 ( SSE; Sum of Squres Error ) : $$\sum{(Y_i - \hat{Y_i})^2}$$
- 회귀제곱합 ( SSR; Sum of Squres Regression ) : $$\sum{(\hat{Y_i} - \bar{Y})^2}$$

우리가 추정한 회귀식이 표본을 잘 설명하고 있다면 설명된 제곱합 SSR 은 설명 안 된 제곱합 SSE 에 비해 상대적으로 클 것이다. 따라서 결정계수 ( $$R^2$$ ) 는 $$1$$ 에 가까워지게 된다. 반대로 회귀식이 표본을 잘 설명하지 않는다면 설명 안 된 제곱합 SSE 이 설명된 제곱합 SSR 에 비해 상대적으로 크게 되어 결정계수는 $$0$$ 에 가까워지게 된다.

$$
\begin{align}
제곱합 \quad &SST = SSE + SSR \\
자유도 \quad &n-1 = (n-k-1) + (k) \\
\end{align}
$$

그렇다면 각 제곱합들의 자유도는 어떻게 될까? 먼저 SST 의 자유도는 수식에서 평균 ( $$\bar{Y}$$ ) 이라는 parameter 가 사용되었으므로 자유도는 $$n -1$$ 이 된다. SSE 의 자유도는 예측값 ( $$\hat{Y_i} = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$$ ) 을 추정하기 위해 회귀식의 parameter 인 $$\beta_0, \beta_1, \dots, \beta_k$$ 가 사용되었으므로 자유도는 $$n-k-1$$ 이 된다. 마지막으로 SSR 의 자유도는 SST - SSE 이므로 자유도는 $$1$$ 이 된다. 이는 곧 회귀식에 사용한 독립변수 ( $$X$$, independent variables ) 의 갯수와 같다는 것을 알 수 있다.
