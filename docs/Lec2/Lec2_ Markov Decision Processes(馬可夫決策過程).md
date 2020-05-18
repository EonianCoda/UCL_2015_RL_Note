---
tags: 機器學習, 強化學習, UCL Course on RL
---
# Lec2: Markov Decision Processes(馬可夫決策過程)

Markov Process(MP, 馬可夫過程)
---
### Markov Property

1. 未來的狀態(state)只與<font color='EA0000'><b>現在的狀態</b></font>有關，跟過去以往的狀態無關
2. 一個State $S_t$擁有馬可夫性質(Markov) if only if
![](http://latex.codecogs.com/gif.latex?\\P[S_{t+1}| S_t]=P[S_{t+1}| S_1,...,S_t])
$$P[S_{t+1}| S_t]=P[S_{t+1}| S_1,...,S_t]$$
```
未來只會取決於現在!
```

### Definition

馬可夫過程(Markov Process)是由下列兩個東西所定義的：
1. $\cal S$：具有<font color="EA0000">Markov property</font>的有限狀態集合
2. $\cal P$：狀態轉移的機率矩陣，$\cal P_{ss'}$是從狀態s轉移至狀態s'的機率，下圖是P矩陣的示意圖:
![](https://raw.githubusercontent.com/EonianCoda/UCL_2015_RL_Note/master/docs/Lec2/pictures/3.PNG =420x200)

* 每行<font color='EA000'><b>row</b></font>總和為**1** (從一個狀態轉移到其他狀態的機率和為**1**)
* 下圖是一個**馬可夫過程**的範例：
圓形代表狀態，箭頭代表狀態之間轉移的機率
![](https://raw.githubusercontent.com/EonianCoda/UCL_2015_RL_Note/master/docs/Lec2/pictures/4.PNG =370x310) 


### Episode

1.Episode是指「一連串的事件，直到結束」:$$Episode=(S_1, S_2, S_3,..., S_T)$$
2.假設起始狀態為$S_1$="*Class1*"，當遇到狀態$S_T=$"*Sleep*"時結束：

&#160; 由於每次<font color='EA000'>**狀態的轉移**</font>都是**隨機的**，所以每次從$S_1$="*Class1*"開始，到$S_T=$"*Sleep*"結束，中間的過程都是不固定的，舉例來說，以$S_1$="*Class1*"開始的Episode可以是：
(C1表class1, C2表class2, C3表class3)
1. C1 -> C2 -> C3 -> Pass -> Sleep
2. C1 -> FB -> FB -> C1 -> C2 -> Sleep
3. C1 -> C2 -> C3 -> Pub -> C2 -> C3 -> Pass -> Sleep


Markov Reward Process(MRP)
---
### Definition

一個Markov Reward Process(MRP)是由以下四個元素組成：
1. $\cal S$：具有Markov property的有限狀態集合
2. $\cal P$：狀態轉移的機率矩陣
3. $\cal R$：一個收益函數(reward function)，用於定義某個狀態的良好程度，$\cal R_s$表$S_t=s$時的收益期望值：
$${\cal R_s}= E[R_{t+1}|S_t=s]$$
(值得注意的是，$R_{t+1}$是指時間t+1時的收益，並非收益函數$\cal R$)
4. $\gamma$：折扣因子(discount factor)，$\gamma\in[0,1]$

* 以下是一個**MRP**的範例：
其中<font color='EA0000'><b>紅字</b></font>表示的是各狀態的良好程度，即收益函數$\cal R$，其他部分皆與MP相同。
![](https://raw.githubusercontent.com/EonianCoda/UCL_2015_RL_Note/master/docs/Lec2/pictures/6.PNG =370x310) 
* 對於MRP的**Episode**定義如下：$$Episode^{MRP}=(S_1,R_{2},S_2,R_{3},...,S_T,R_{T+1})$$

### Return
定義符號<b>$G_t$</b>為「從時間$t$到結束，經過折扣(discounted)的未來收益總和」：$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...={\sum_{k=0}^{\infty}}\gamma^kR_{t+k+1}$$
* 當折扣因子$\gamma\rightarrow0$時，表示只注重<font color='EA0000'>當前收益</font>
* 當折扣因子$\gamma\rightarrow1$時，表示當前收益與未來收益**同樣重要**

### State-Value Function

定義State-Value function $v(s)$為「當遇到狀態$s$時，從現在到結束時的預期總收益」：$$v(s)=E[G_t|\;S_t=s]$$

### Bellman Equation For value function

一個Value function可以被拆分為兩部分：
1. 當前收益$R_{t+1}$
2. 未來預期收益$\gamma\,v(S_{t+1})$
$$
\begin{align}
v(s) &= E[\,G_t|\;S_t=s]\\
&= E[\,R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...\;|\;S_t=s]\\
&= E[\,R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+...)\;|\;S_t=s]\\
&= E[\,R_{t+1}+\gamma\;G_{t+1}\;|\;S_t=s]\\
&= E[\,R_{t+1}+\gamma\;v(S_{t+1})\;|\;S_t=s]\\
&= E[\,R_{t+1}\;|\;S_t=s] + E[\,\gamma\;v(S_{t+1})\;|\;S_t=s]
\end{align}
$$
根據上式，Value function又可以寫成：
$$
E[R_{t+1}\;|\;S_t=s]=\;{\cal R_s} \tag{1}
$$
$$
\begin{align}
E[\,\gamma\;v(S_{t+1})\;|\;S_t=s] 
&= \gamma E[\;v(S_{t+1})\;|\;S_t=s] \tag{2} \\
&= \gamma {\sum_{s'\in S}^{\infty}}P_{ss'}\,v(s')
\end{align}
$$
$$
根據(1)與(2)：\;
v(s)={\cal R_s}+\gamma{\sum_{s'\in S}^{\infty}}P_{ss'}\,v(s')$$
* ${\cal R_s}$：狀態S的收益期望值
* ${\sum_{s'\in S}^{\infty}}P_{ss'}\,v(s')$：窮舉所有狀態s可能的轉移，並將其轉移機率乘上以新狀態$S'$開始至結束的預期總收益

### Bellman Equation in Matrix Form
一個value function的Bellman Equation可以以**矩陣**的方式呈現：
$$
v= {\cal R + \gamma P}v
$$
![](https://raw.githubusercontent.com/EonianCoda/UCL_2015_RL_Note/master/docs/Lec2/pictures/7.PNG)
解上述方程式的方法：
$$
\begin{align}
v=& {\cal R + \gamma P}v \\
Iv =& {\cal R + \gamma P}v\; \; \; \; (I為單位矩陣)\\
(I-{\cal \gamma P})v =& {\cal R} \\
v =& (I-{\cal \gamma P})^{-1}\,{\cal R}
\end{align}
$$
* 假設有n個狀態，則計算的複雜度為<font color='EA0000'><b>O($n^3$)</b></font>
(因為計算**反矩陣**的複雜度為O($n^3$)，而計算反矩陣是會花費最長時間的地方)
* 直接去解反矩陣這種方式只適用於很小的MRP&nbsp; 
(因為複雜度過高)
* 下面列出三種方法，常用於解很大的MRP：
    1. Dynamic programming　(動態規劃)
    2. Monte-Carlo evaluation　(蒙地卡羅估計)
    3. Temporal-Difference learning　(時間差分)

Markov Decision Process(MDP, 馬可夫決策過程)
---
### Definition

一個Markov Decision Process(MDP)是由以下五個元素組成：$$MDP=\cal< S,\,A,P,R,\gamma>$$
1. $\cal S$：有限的狀態集合
2. $\cal A$：有限的動作集合
3. $\cal P$：狀態轉移的機率矩陣，<font color='EA0000'>${\cal P}_{ss'}^a$</font>表在「在狀態s時，執行動作a」轉移至狀態$s'$的機率 $${{\cal P}_{ss'}^a}=[S_{t+1}=s'\;|\;S_t=s,\,A_t=a]$$
5. $\cal R$：一個收益函數(reward function)，$R_{s}^a$表「在狀態s時，執行動作a」的預期收益 $$R_{s}^a=E[\,R_{t+1}\;|\;S_t=s,\,A_t=a]$$
6. $\gamma$：折扣因子(discount factor)，$\gamma \in [0,1]$

* 下圖是MDP的範例：
<font color='EA0000'>紅字</font>表示動作集合$\cal A$，在紅字下的收益R表$R_{s}^{a}$
![](https://raw.githubusercontent.com/EonianCoda/UCL_2015_RL_Note/master/docs/Lec2/pictures/8.PNG =370x310)
* * 對於MDP的**Episode**定義如下：$$Episode^{MDP}=(S_1,A_1,R_{2},S_2,A_2,R_{3},...,S_T,A_T,R_{T+1})$$

### Policy
定義Policy <font color='EA0000'>$\pi$</font>，是一個給定某個狀態下的動作**機率分布**，即「在某狀態下，會執行某動作的機率」組成的分布：$$\pi(a|s)=P(\,A_t=a\;|\;S_t=s)$$
* Policy是<font color='EA0000'>**與時間無關的**</font>，即在任何時間下，Policy對於某狀態的動作機率分布是相同的

定義某個**MDP**遵循Policy $\pi$：$$MDP^{\pi}=\cal< S,\,A,P^\pi,R^\pi,\gamma>$$
1. ${\cal P^\pi}$：由Policy $\pi$控制的狀態轉移矩陣，以下定義<font color='EA0000'>${\cal P_{s,s'}^{\pi}}$</font>：$${{\cal P}_{ss'}^a}=[S_{t+1}=s'\;|\;S_t=s,\,A_t=a]$$$${\cal P_{s,s'}^{\pi}}=\sum_{a\in {\cal A}}\pi(a|s){\cal P_{ss'}^{a}}$$
2. ${\cal R^\pi}$：與Policy $\pi$相關的預期收益函數，以下定義<font color='EA0000'>${\cal R^{\pi}_{s}}$</font>：$$R_{s}^a=E[\,R_{t+1}\;|\;S_t=s,\,A_t=a]$$$${\cal R^{\pi}_{s}}=\sum_{a\in {\cal A}}\pi(a|s){\cal R_{s}^{a}}$$

### State-Value function on Policy $\pi$
在Policy $\pi$下，從狀態s開始的預期總收益：$$v_\pi(s)=E_\pi[G_t\;|\;S_t=s]$$
### Action-Value function on Policy $\pi$
在Policy $\pi$下，「從狀態s開始且採取動作a」的預期總收益：$$q_\pi(s,a)=E_\pi[G_t\;|\;S_t=s,\,A_t=a]$$

### Bellman Expectation Equation
$v_\pi(s)=E_\pi[G_t\;|\;S_t=s]$
$q_\pi(s,a)=E_\pi[G_t\;|\;S_t=s,\,A_t=a]$
1. 將$G_t$分解為**當前收益**與**未來預期收益**兩項
$$
v_\pi(s)=E_\pi[R_{t+1}+\gamma\;v_\pi(S_{t+1})\;|\;S_t=s]
$$
$$q_\pi(s)=E_\pi[R_{t+1}+\gamma\;q_\pi(S_{t+1})\;|\;S_t=s,A_t=a]
$$
2. 將$v_\pi(s)$以$q_\pi(s,a)$表示、$q_\pi(s,a)$以$v_\pi(s)$表示:
$$
v_\pi(s)=\sum_{a\in \cal A}\pi(a\;|\;s)q_\pi(s,a)
$$
$$
q_\pi(s,a)={\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')}
$$
3. 再將彼此的結果互相代入:
$$
\begin{align}
v_\pi(s)
&=\sum_{a \in {\cal A}}\pi(a\;|\;s)({\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s'))}\\
&=\sum_{a \in {\cal A}}\pi(a\;|\;s){\cal R_{s}^a}+\gamma \sum_{a \in {\cal A}}\sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')}\\
&={\cal R_{s}^\pi}+\gamma \sum_{a \in {\cal A}}\sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')}\; \; (因為\sum_{a \in {\cal A}}\pi(a\;|\;s){\cal R_{s}^a}={\cal R_{s}^\pi}，這是{\cal R_{s}^\pi}的定義)\\
\end{align}
$$
$$
\begin{align}
q_\pi(s,a)
&={\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}}(\sum_{a'\in \cal A}\pi(a'\;|\;s')q_\pi(s',a'))\\
&={\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}\sum_{a'\in \cal A}{\cal P_{ss'}^{a}}\,\pi(a'\;|\;s')q_\pi(s',a')
\end{align}
$$
#### In matrx form
$$
\begin{align}
v_\pi&=R^\pi+\gamma\,{\cal P^\pi}\,v_\pi\\
I\,v_\pi&=R^\pi+\gamma\,{\cal P^\pi}\,v_\pi\; \; (I為單位矩陣)\\
(I\,-\gamma{\cal P})v_\pi &= R^\pi \\
v_\pi&=(I\,-\gamma{\cal P^\pi})^{-1} R ^ {\pi}
\end{align}
$$
可以發現這個結果與我們上方推導(Bellman Equation in Matrix Form)的結果幾乎一模一樣，差別只是這裡的${\cal R}$與${\cal P}$都是遵循Policy $\pi$

### Optimal MDP
#### Optimal Value Function
定義$v_*(s)$為Optimal **state-value** Function：
$$
v_*(s) = \max_{\pi} v_{\pi}(s)
$$
定義$q_*(s, a)$為Optimal **action-value** Function：
$$
q_*(s,a) = \max_{\pi} q_{\pi}(s,a)
$$
#### Optimal Policy
定義Policy的大小關係為：
$$
\pi \geq \pi'\; \; \; if \; v_\pi(s) \geq v_{\pi'}(s) \; ,\forall s 
$$
對於任何MDP而言，存在$\pi_*$為Optimal Policy，滿足：
1. $\pi_* \geq \pi \; ,\forall \pi$
2. $v_{\pi_*}(s)=v_*(s)$ (遵循Policy $\pi$的v(s)將會是Optimal的)
3. $q_{\pi_*}(s,a)=q_*(s,a)$ (遵循Policy $\pi$的q(s,a)將會是Optimal的)

#### Finding an Optimal Policy
要找到最佳的Policy $\pi_*$可透過最大化$q_*(s,a)$：
$$
\DeclareMathOperator*{\argmax}{argmax}
\pi_*(a\;|\;s) = 
\begin{cases}
1 \; \; if \; a = \argmax_{a \in {\cal A}}\, q_*(s,a) \\
0 \; \; otherwise \\
\end{cases}
$$
* 最佳的Policy便是在每個狀態中都選擇最佳的Action，即$q_*(s,a)$
* Policy $\pi$是，是一個給定某個狀態下的動作**機率分布**，而現在只有會讓q(s,a)最大的Action會被採用，所以對於Policy $\pi$而言，無論遇到任何狀態s，機率分布都會長的像(1,0,0),(0,1,0),(0,0,1)，即只有一個動作的機率是1，其他都是0

#### Bellman Optimality Equation
由於最佳的Policy便是在每個狀態中都選擇最佳的Action，即$q_*(s,a)$，因此我們可以把在Bellman Expectation Equation中討論的結果化簡，以找到最佳的Equation：
$$
\begin{align}
v_\pi(s)
&=\sum_{a \in {\cal A}}\pi(a\;|\;s)({\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s'))}\\
&=\sum_{a \in {\cal A}}\pi(a\;|\;s){\cal R_{s}^a}+\gamma \sum_{a \in {\cal A}}\sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')}\\
&={\cal R_{s}^\pi}+\gamma \sum_{a \in {\cal A}}\sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')}\; \; (因為\sum_{a \in {\cal A}}\pi(a\;|\;s){\cal R_{s}^a}={\cal R_{s}^\pi}，這是{\cal R_{s}^\pi}的定義)\\
\end{align}
$$
$$
\begin{align}
q_\pi(s,a)
&={\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}}(\sum_{a'\in \cal A}\pi(a'\;|\;s')q_\pi(s',a'))\\
&={\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}\sum_{a'\in \cal A}{\cal P_{ss'}^{a}}\,\pi(a'\;|\;s')q_\pi(s',a')
\end{align}
$$
上面兩個式子是我們推出的最後結果，如果要遵循最佳的Policy的話，則意思在不同狀態中找到最好的Action：
$$
\begin{align}
v_*(s)
&=\max_a({\cal R_{s}^\pi}+\gamma \sum_{a \in {\cal A}}\sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')})\\
&=\max_a({\cal R_{s}^a}+\gamma \sum_{a \in {\cal A}}\sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')})\\
&(由於現在採取最佳Policy，因此當遇到狀態s時{\cal R_{s}^\pi}\\
&只有一種可能，那便是會有最大收益的\max_a{\cal R_{s}^a})\\
&=\max_a({\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}v_\pi(s')})\\
&(同理，由於採取最佳Policy,所以只有最佳動作會被採用，機率=1，\\
&其他動作被採取的機率是0，因此不需要\sum_{a \in {\cal A}}，只要\max_a{\cal P^{a}_{ss'}})
\end{align}
$$
$$
\begin{align}
q_*(s,a)
&={\cal R_{s}^a}+\max_{a'}(\gamma \sum_{s'\in \cal S}\sum_{a'\in \cal A}{\cal P_{ss'}^{a}}\,\pi(a'\;|\;s')q_\pi(s',a'))\\
&={\cal R_{s}^a}+(\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}}\max_{a'}\,\sum_{a'\in \cal A}\pi(a'\;|\;s')q_\pi(s',a'))\\
&({\cal P^a_{ss'}跟a'無關，a'是下一個狀態中要採用動作，現在狀態要採用的動作已經被決定了})\\
&={\cal R_{s}^a}+\gamma \sum_{s'\in \cal S}{\cal P_{ss'}^{a}}\max_{a'}\,q_\pi(s',a')\\
&(由於採取最佳Policy,所以在下個狀態s'中，只有最佳動作會被採用\\
&，因此\pi(a'\;|\;s')只有一個動作機率會是1，其他為0，所以可以把\sum_{a'\in \cal A}\pi(a'\;|\;s')刪除)
\end{align}
$$






資料來源:
https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf
