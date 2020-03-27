# 时间序列分析

## 时间序列基本概念
 - 时间序列的概率分布
 - 时间序列的预处理

### 时间序列的概率分布

对于时间序列 $\\{X_t,\ t\in T\\}$, 任取正整数 $m$，任取$t_1,\ t_2,\ ...,\ t_m\in T$，
则$m$维随机向量$(X_{t_1},\ X_{t_2},\ ...,\ X_{t_m})$的联合概率分布定义为:
$$
F_{t_1, t_2, ..., t_m}(x_1, x_2, ..., x_m) = P(X_{t_1}<x_1, X_{t_2}<x_2, ...,X_{t_m}<x_m)
$$
由这些有限维分布函数构成的全体:
$$
\\{F_{t_1, t_2, ..., t_m}(x_1, x_2, ..., x_m),\ \forall m\in 正整数,\ \forall t_1,\ t_2,\ ...,\ t_m\in T\\}
$$
就成为时间序列$\\{X_t,\ t \in T\\}$的概率分布族。

- 自相关系数和自协方差

对于时间序列$\\{X_t,\ t \in T\\}$，任取 $t,s \in T$，**自协方差函数**定义为:

$$
r(t, s) = E(X_t-\mu_t)(X_s-\mu_s)
$$

**自相关系数**定义为:

$$
\rho(t, s) = \frac{r(t, s)}{\sqrt{D(X_t) \cdot D(X_s)}}
$$

### 时间序列的预处理

    - 平稳性检验
    - 纯随机性检验
    
> 平稳性检验

- 特征统计量
- 平稳时间序列的定义
- 平稳时间序列的统计性质
- 平稳时间序列的意义
- 平稳性的检验

> 特征统计量

**特征统计量: 均值、方差、自协方差、自相关系数**

其中自协方差是两个不同时间点的随机变量协方差的计算，自相关系数同。

> 平稳时间序列的定义

- 严平稳时间序列

严平稳是一种条件比较苛刻的平稳性定义，它认为只
有当序列所有的统计性质都不会随着时间的推移而发
生变化时，该序列才能被认为平稳。也就是说所有统计量的联合概率分布均相同（仅与时间长度有关，而与起始点无关）<br>
数学表示为：所有统计性质不随$t$变化。

对时间序列$\\{X_t,\ t \in T\\}$，对于任意正整数$m$，任取$t_1, t_2, L, t_m \in T$，
任意正整数$\tau$，有

$$
F_{t_1, t_2, L, t_m}(x_1, x_2, L, x_m) = F_{t_(1+\tau), t_(2+\tau), L, t_(m+\tau)}(x_1, x_2, L, x_m)
$$

则称时间序列$\\{X_t,\ t \in T\\}$为严平稳时间序列。

- 宽平稳时间序列

宽平稳是使用序列的特征统计量来定义的一种平稳性。
它认为序列的统计性质主要由它的低阶矩决定，所以
只要保证序列低阶矩平稳（二阶），就能保证序列的
主要性质近似稳定。有：同期望、同方差、同自协方差和自相关系数（仅与时间长度有关，而与起始点无关）<br>
数学表示为：对时间序列$\\{X_t,\ t \in T\\}$，如果满足以下三个条件:
1. 任取$t\in T$，有$E(X_t^2)<\infty$;
1. 任取$t\in T$，有$E(X_t)=\mu, \mu$为常数;
1. 任取$t, s, k \in T$，且$k + s - t \in T$，有$r(t, s) = r(k, k+s-t)$.

则称$\\{X_t,\ t \in T\\}$为宽平稳时间序列（弱平稳、二阶平稳）。

⚠️: $(3) \Leftrightarrow r(t, s) = r(t+\tau, s+\tau)$，其中，$t, s \in T, t+\tau \in T, s+\tau \in T$.

- 严平稳与宽平稳
    - 严平稳条件比宽平稳条件苛刻，通常情况下，严平稳（低阶
矩存在）能推出宽平稳成立，而宽平稳序列不能反推严平稳
成立
    - 二阶矩存在的严平稳为宽平稳
    - 序列为多元正态分布时，宽平稳为严平稳（概率密度函数特点所决定的）。
    
> 平稳时间序列统计性质与意义

- **性质1**: 常数均值
- **性质2**: 自协方差函数和自相关函数只依赖于时间的平移长度而与时间的起止点无关
- **意义:**

1. 时间序列数据结构：可列多个随机变量，而每个变量只有一个样本观察值。
2. 在平稳序列场合，`序列的均值等于常数`，这意味着原本含有可列多个随机变量的均值序
列变成了只含有一个变量的常数序列。
3、 原本每个随机变量的均值（/方差/自相关系数）只能依靠唯一的一个样本观察值去估计，
现在由于平稳性，每一个统计量都将拥有大量的样本观察值。
`这极大地减少了随机变量的个数，并增加了待估变量的样本容量。`极大地简化
了时序分析的难度，同时也提高了对特征统计量的估计精度。

**延迟`K`自协方差函数**

对于平稳时间序列 $\\{X_t, t \in T\\}$，任取 $t(t + k \in T)$，该时间序
列的延迟$k$自协方差函数和延迟 $k$ 自相关系数定义如下：

**延迟$k$自协方差函数**
$$
\gamma(k) = \gamma(t, t+k), \forall k 为整数.
$$
平稳随机序列一定具有常数方差，即$DX_t = r(t, t) = r(0), \forall t \in T$

**延迟$k$自相关系数**
$$
\rho_k = \frac{\gamma(t, t+k)}{\sqrt{DX_t \cdot DX_{t+k}}} = \frac{\gamma(k)}{\gamma(0)}
$$

**自相关系数的性质**

- 规范性: $\rho_0 = 1 \ and\ |\rho_k| \le 1, \forall k$
- 对称性: $\rho_k = \rho_{-k}$
- 非负定性: 对任意的正整数$m$,相关阵 $\Gamma_m$ 为对称非负定阵
$$
\Gamma_m = 
\begin{bmatrix}
\rho_0 & \rho_1 & L & \rho_{m-1} \\\\
\rho_1 & \rho_0 & L & \rho_{m-2} \\\\
M & M & \ & M \\\\
\rho_{m-1} & \rho_{m-2} & L & \rho_{0} \\\\
\end{bmatrix}
$$
- 非唯一性: 一个平稳时间序列一定唯一决定了它的自相关系数，
但一个自相关系数未必唯一对应着一个平稳时间序列。


> 平稳性的检验

1. 图检验方法(时序图检验、自相关图检验)
    - 特点：操作简单，应用广泛，但结论带有一定主观性。
    - 自相关图是以自相关系数为横轴，延迟时期数为倒纵轴，
    水平方向的垂线表示自相关系数的大小，是一个二维平面坐标悬垂线图。
    ***根据平稳序列通常具有短期相关性的特点，则随着延 迟期数的增加，平稳序列的自相关系数会很快地衰减为零，而非平稳序列的自相关系数较慢地衰减为零***
2. 统计检验方法（单位根检验ADF检验）

如果一个时间序列是平稳序列，那么它一定能用ARMA（p，q）模型去描述，当然p、q的取值是不确定的。如果描述该时间序列的ARMA模型是稳定的，那么这组时间序列一定是稳定的。因为ARMA模型的平稳性完全取决于它的AR（p）部分，因此可以通过计算特征根的方法来确定该ARMA模型的稳定性。计算特征根的方法呢可以引入滞后算子去计算，AR模型的表现形式是$x_t =  \varphi_1x_{t-1} + \varphi_2x_{t-2} + ... + \varphi_px_{t-p} + \varepsilon_t$，特征根的计算方式是$$\lambda^p-\varphi_1\lambda^{p-1}-...\varphi_p=0$$，特别的当特征根等于1的时候,有$$\varphi_1+\varphi_2+...+\varphi_p=1$$。因此对于p阶自回归过程可以考察可以通过考察自回归系数之和是否小于1来考察序列的平稳性。通过差分方程来考察$\varphi_1+\varphi_2+...+\varphi_p$的值，具体如下：
$$
\begin{aligned}
\begin{aligned}
x_t & =  \varphi_1x_{t-1} + \varphi_2x_{t-2} + ... + \varphi_px_{t-p} + \varepsilon_t\\\\
\Delta x_t & =  （\varphi_1-1）x_{t-1} + \varphi_2x_{t-2} + ... + \varphi_px_{t-p} + \varepsilon_t\\\\
& =  （\varphi_1+\varphi_2-1）x_{t-1} - \varphi_2 \Delta x_{t-2} + ... + \varphi_px_{t-p} + \varepsilon_t\\\\
& = ...\\\\
& =  （\varphi_1+\varphi_2+...+\varphi_p-1）x_{t-1} - (\varphi_2+...+\varphi_p-1)\Delta x_{t-1}- ... - \varphi_p\Delta x_{t-p+1} + \varepsilon_t\\\\
& =\rho x_{t-1}+\beta_1\Delta x_{t-1}+...+\beta_{p-1}\Delta x_{t-P+1}
\end{aligned}
\end{aligned}
$$
也就是说，ADF检验可以通过检验$\rho$是否小于0来判断这个序列是否平稳。ADF检验的原假设就是 H0：$\rho$=0 H1:$\rho$<0。那么通过构建统计量进行检验，观测p值，如果p值小于5%就可以认为序列是平稳的。
    
> 纯随机序列

- 描述性定义：**序列值间没有任何相关性，过去的行为对将来的发展没有丝毫影响，这种序列称为纯随机序列，也称白噪声序列。**
 从统计分析的角度而言，
纯随机序列是没有任何实际分析价值的序列。

- 数学定义

**定义**

如果时间序列$\\{X_t, t \in T\\}$满足如下性质:

$$
EX_t = \mu, \forall t \in T \tag{1}
$$

$$
\gamma(t, s)
\begin{cases}
\sigma^2, t = s\\\\
0, t \ne s
\end{cases}
, \forall t, s \in T \tag{2}
$$

则称序列 $\\{X_t, t \in T\\}$ 为纯随机序列，也称为白噪声序列，简记为 $X_t\sim WN(\mu, \sigma^2)$。

⚠️ 注意:

1. 白噪声序列一定是平稳序列，而且是最简单的平稳序列。
2. 纯随机序列是一种没有分析价值的序列（统计不相关）。

- 性质
   - 纯随机性：指白噪声序列各值之间没有任何相关关系，即为“无记忆”的序列。序列完全无序的随机波动，没有任何值得提取的有用相关信息。
   - 自协方差和自相关系数为0，即
   
$$
\begin{aligned}
\gamma(k) = 0, & \forall k\ne 0 \\\\
\rho_k = 0, & \forall k \ne 0 \\\\
\end{aligned}
$$

   - 方差齐性： 指白噪声序列中每个变量的方差都相等。
根据马尔可夫定理，只有方差齐性的序列，用最小二乘
法得到的未知参数估计值才是准确的、有效的。

- 纯随机序列的应用
1. 对于一个观察值序列，一旦相关信息全部提取（通过拟合模型
进行）完毕，则剩余的残差序列应具有纯随机性。检验残差的纯随机性是用于判定序列相关信息是否提取充分的
标准之一。
2. 对于一个观察值序列，一旦相关信息全部提取完毕，则剩
余的残差序列应具有方差齐性。检验残差的方差齐性是用于判定序列相关信息是否提取充
分（即白噪声序列）的另一标准。

- 纯随机性检验

纯随机性检验也称为**白噪声检验**，是专门用来检验序列是否为纯随机序列的一种方法。

根据纯随机性的定义，只要满足自协方差或自相关系数为$0$，则该序列就具有纯随机性。实际上，由于观察值序列的有限性，
纯随机序列的样本自相关系数不会绝对为0。

**Barlett定理**

对一个`纯随机`的时间序列，得到一个$n$期（观察期数）的观察序列 $\\{x_t, t=1, ..., n\\}$,
则该序列的延迟 $k(k \ne 0)$ 期的样本自相关系数将近似服从均值为$0$，方差为$1/n$的正态分布，即

$$
\hat{\rho}_k \dot \sim N(0, \frac{1}{n}), \forall k \ne 0\\\\
\sqrt{n} \hat{\rho}_k \dot \sim N(0, 1)
$$

**假设检验的基本步骤**

1. 根据问题设定假设检验的原假设与备择假设
2. 构造合适的检验统计量并确定检验统计量的分布
3. 给定显著性水平，一般显著性水平(0.05)，然后根据显著性水平和检验统计量的分布来确定拒绝域
4. 计算检验统计量的大小，根据它是否属于拒绝域来确定原假设的接受还是拒绝。

**统计假设**

- H0：延迟期数小于或等于 $m$ 的序列值间相互独立
- H1： 延迟期数小于或等于$m$的序列值间有相关性

**检验统计量**

- Q统计量(Box和Pierce): 适合大样本，由barlett定理及卡方统计量定义，序列的平方近似服从咖方分布

- LB统计量(Box和Ljung): 适合小样本，是对Q统计量的修正，因其较准确，现一般都采用LB统计量


**判别原则**

- 拒绝原假设: 当检验统计量大于1-α分位点，或该统计量的 $P$ 值
小于 $α$ 时，则可以以$1-α$ 的置信水平拒绝原假设，认
为该序列为非白噪声序列。

- 接受原假设: 当检验统计量小于1-α分位点，或该统计量的 $P$ 值
大于 $α$ 时，则认为在 $1-α$ 的置信水平下无法拒绝原假
设，即不能显著拒绝序列为纯随机序列的假定。

## ARMA模型概念、平稳性判定和统计性质
- AR模型
- MA模型
- ARMA模型

### AR(p)模型
   - 定义
   - 平稳性判定
   - 统计性质

> AR模型的定义

$$
\begin{aligned}
\begin{cases}
x_t = \varphi_0 + \varphi_{1}x_{t-1} + \varphi_{2}x_{t-2} + \varphi_{p}x_{t-p} + \varepsilon_t \\\\
\varphi_p \ne 0 \\\\
E(\varepsilon_t) = 0, Var(\varepsilon_t) = \sigma_{\varepsilon}^2, E(\varepsilon_t \varepsilon_s) = 0, s \ne t \\\\
E(x_s \varepsilon_t) = 0, \forall s<t.
\end{cases}
\end{aligned}
$$

1. 最高阶为$p$
2. 保证残差零均值白噪声
3. 保证$t$期的随机干扰与过去$s$期的序列值无关
4. 当$\varphi_0 = 0$时，成为中心化$AR(p)$模型

- $AR(p)$ 序列中心化变换: 将非中心化的$AR(p)$转化为中心化$AR(p)$。

令$\mu = \frac{\varphi_0}{1-\varphi_1-...-\varphi_p}$
则
$$
\begin{aligned}
x_t & = \mu(1-\varphi_1-...-\varphi_p) + \varphi_1x_{t-1} + \varphi_2x_{t-2} + ... + \varphi_px_{t-p} + \varepsilon_t \\\\
& = \mu + \varphi_1(x_{t-1} - \mu) + ... + \varphi_p(x_{t-p} - \mu) + \varepsilon_t \\\\
x_t - \mu & = \varphi_1(x_{t-1} - \mu) + ... + \varphi_p(x_{t-p} - \mu) + \varepsilon_t \\\\
\end{aligned}
$$
变换$y_t = x_t -\mu$成为中心化变换。

> AR模型的特征根

由AR模型知，$$x_t =  \varphi_1x_{t-1} + \varphi_2x_{t-2} + ... + \varphi_px_{t-p} + \varepsilon_t$$
移项可得(引入滞后算子$Bx_t=x_{t-1}$)$$(1-\varphi_1B-\varphi_2B^2-...-\varphi_pB^p)x_t= \varepsilon_t$$
令$$\Phi(B)=1-\varphi_1B-\varphi_2B^2-...-\varphi_pB^p$$
特征根的计算为$$\lambda^p-\varphi_1\lambda^{p-1}-...\varphi_p=0$$
不妨设有p个特征根满足$$\lambda_i^p-\varphi_1\lambda_i^{p-1}-...\varphi_p=0$$
令$$u_i=\frac 1{\lambda_i}$$
则$$
\begin{aligned}
\Phi(u_i) & =1-\varphi_1\frac 1{\lambda_i}-\varphi_2\frac 1{{\lambda_i}^2}-...-\varphi_p\frac 1{{\lambda_i}^p}\\\\
& = \frac 1{{\lambda_i}^p}({\lambda_i}^p-\varphi_1{\lambda_i}^{p-1}-...\varphi_p)=0
\end{aligned}
$$

> AR模型平稳性判定
- 时序图和自相关图方法：利用已知的AR(p)模型生成一组数据，看这组时间序列数据是否是平稳的
- 平稳域判别法和单位根判别法
    - 特征根判别（特征根判别法的思路是证明AR（p）模型生成的时间序列$x_t$的期望、方差和自相关系数满足宽平稳的条件） 
        1. $AR(p)$模型平稳的充要条件是它的$p$个特征根都在单位圆内，特征根$|\lambda_1 < 1|$
        2. 根据特征根和自回归系数多项式的根成倒数的性质，等价判别条件是该模型的自回归系数多项式的单位根都在单位圆外，$\Phi(\mu) = 0$ 的根$|\mu_i|>1$
    - 平稳域判别
        - 平稳域的定义: 使特征根都在单位圆内的$AR(p)$的系数集合，即$\\{\varphi_1, \varphi_2, ..., \varphi_p|特征根都在单位圆内\\}$
        - 比较适合低阶AR模型，如1，2阶，高阶模型不容易推导平稳域
    - 经验判断准则
        - $AR(p)$ 模型平稳性的必要条件是: $\varphi_{1} + \varphi_{2} + ... + \varphi_{p} < 1$
        - $AR(p)$ 模型平稳性的充分条件是: $|\varphi_{1}| + |\varphi_{2}| + ... + |\varphi_{p}| < 1$

> 统计性质

- 均值

$$E(x_t) = E(\varphi_{0} + \varphi_{1}x_{t-1} + ... + \varphi_{p}x_{1-p} + \varepsilon_t)$$
因平稳序列均值为常数，且$\\{ \varepsilon_t \\}$为白噪声序列，有$E(x_t) = \mu, E(\varepsilon_t) = 0, \forall t \in T$
则:

$$
\begin{aligned}
E(x_t) & = E(\varphi_{0} + \varphi_{1}x_{t-1} + ... + \varphi_{p}x_{1-p} + \varepsilon_t) \\\\
& = E(\varphi_{0}) + \varphi_{1}E(x_{t-1}) + ... + \varphi_{p}E(x_{1-p}) + E(\varepsilon_t) \\\\
& = \varphi_{0} + \varphi_{1}\mu + ... + \varphi_{p}\mu \\\\
\mu & = \varphi_{0} + \varphi_{1}\mu + ... + \varphi_{p}\mu \\\\
& = \frac{\varphi_0}{1 - \varphi_1 - ... - \varphi_p}
\end{aligned}
$$

-方差
AR模型的方差很难计算，因为每个$X_t$都受到之前所有变量的影响，因此需要通过传递公式将AR模型转换为MA模型去计算。根据$\Phi（B）x_t=\varepsilon_t$和
$\Phi（B）=\prod_{i=1}^{p}(1-\lambda_iB)(\lambda_i是特征根)$可以得到：
$$
\begin{aligned}
x_t & =\frac {\varepsilon_t}{\Phi（B）}\\\\
& =\frac {\varepsilon_t}{\prod_{i=1}^{p}(1-\lambda_iB)} \\\\
& = \sum_{i=1}^p{\frac {k_i}{1-\lambda_iB}}\varepsilon_t(用待定系数法展开)\\\\
& = \sum_{j=0}^\infty{\sum_{i=1}^p{k_i{\lambda_i}^jB^j\varepsilon_t}}\\\\
& = \sum_{j=0}^\infty{\sum_{i=1}^p{k_i{\lambda_i}^j\varepsilon_{t-j}}}\\\\
& = \sum_{j=0}^\infty{G_j\varepsilon_(t-j)}\\\\
\end{aligned}
$$
根据上式，可以写成AR和MA模型的转换公式：
$$
\begin{aligned}
x_t & = \sum_{j=0}^\infty{G_jB^j\varepsilon_t}\\\\
& = G(B)\varepsilon_t\\\\
\end{aligned}
$$
- 自相关系数的性质
   - 量呈指数衰减（时序图判断时间序列平稳性的准则：短期相关性）
   - 拖尾性：始终有非0的取值，并不会在K大于某个数后就为0了（原因在于，$X_t$始终会受到$X_{t-1}...X_{t-p}$的影响，而$X_{t-1}$也会受到$X_{t-2}...X_{t-p-1}$的影响，也就是说$X_t$会受到前面所有数据的影响，故无论K取多大自相关系数始终不会为0）
- 偏自相关系数
   - 定义：对于平稳AR(p)序列，所谓滞后K偏自相关系数就是在剔除了中间K-1个随机变量（$X_{t-1}...X_{t-k+1}$）后，$X_{t-k}$对$X_t$的相关度量。
   - 性质：零均值平稳序列$\{X_t\}$为AR(p)序列的充要条件是$\{X_t\}的偏自相关系数p步截尾，即$X_{t-p}$期后边的变量对$X_t$没有影响了。但是实践操作中往往由于样本偏自相关只是对总体的描述，因此可能样本偏自相关系数在0附近波动。***滞后K偏自相关系数实际上就是K阶自回归模型第K个回归系数的值***
   

### MA(q)模型
   - 定义
   - 平稳性判定
   - 统计性质
   - 可逆性
   
> MA(q)模型定义

$$
\begin{cases}
x_t = \mu + \varepsilon_t + \theta_{1} \varepsilon_{t-1} + \theta_{2} \varepsilon_{t-2} + ... + \theta_{q} \varepsilon_{t-q} \\\\
\theta_q \ne 0 \\\\
E(\varepsilon_t) = 0, Var(\varepsilon_t) = \sigma_{\varepsilon}^2, E(\varepsilon_t \varepsilon_s) = 0, s \ne t.
\end{cases}
$$

当$\mu = 0$时，中心化$MA(q)$模型。简$x_t = \Theta(B)\varepsilon_t$。
$q$阶移动平均系数多项式: $\Theta(B) = 1 - \theta_1 B - \theta_2 B^2 - ... - \theta_q B^q $。

> MA(q)模型平稳性判定

方差存在且有界的$MA$模型生成的序列总是平稳的，所以对应的$MA$模型也总是平稳的。

> MA(q)模型统计性质

- 期望：常数期望
- 方差：常数方差
- 自协方差函数和自相关系数：可以直接计算，因为$\{x_t\}$是白噪声序列的线性组合，比较好计算
   - 性质：自协方差函数和自相关系数都是q阶截尾（超过q阶的自协方差函数没有交叉的随机变量$\varepsilon_i$,故值为0）

> MA模型的可逆性

- 自相关系数的不唯一性

不同的MA模型，可能具有完全相同的自相关系数。举例来说$X_t=\varepsilon_t-2\\varepsilon_{t-1}$和$X_t=\varepsilon_t-0.5\\varepsilon_{t-1}$具有完全相同的自相关系数
$$
\rho_k
\begin{cases}
-0.4, k = 1\\\\
0, k \ne 1
\end{cases}
$$
这种自相关系数的不唯一性，也就是模型和自相关系数不唯一性，会给以后建模带来麻烦。因为我们是依靠样本自相关系数显示出来的特征选择合适的模型，但是**如果自相关系数和模型不是一一对应的关系，将导致拟合模型和随机序列也不是一一对应的关系**为了保证一个给定的自相关函数，能够对应唯一的模型，我们要给模型增加约束条件。这个条件称为模型的可逆性条件。

- 逆函数和可逆性

1. MA模型的逆函数

以MA(1)模型为例，用过去序列值的一个线性组合去逼近现在时刻的行为，即
$$
\begin{aligned}
 X_t & = I_1X_{t-1}+ I_2X_{t-2} + ... + \varepsilon_t \\\\
 & = \sum_{j=1}^\infty{I_jX_{t-j}} + \varepsilon_t \\\\
 \varepsilon_t & = (1-I_1B-I_2B^2...I_jB^j)X_t\\\\
 & = (1-\sum_{j=1}^\infty{I_jB^j})X_t
 \end{aligned}
 $$
 
 如果说MA模型生成的序列可以用一个无限阶的自回归模型逼近，即逆函数$I_j$存在，我们就称该模型具有可逆性，也就是可逆的。否则就是不可逆的。**一个自相关系数列对应着唯一一个可逆MA模型**
 
 拓展开来，如果是MA(q)模型的话，可能是（我的猜测）AR模型的方差很难计算，因为每个$X_t$都受到之前所有变量的影响，因此需要通过传递公式将AR模型转换为MA模型去计算。根据$x_t=\Theta（B）\varepsilon_t$和$\Theta（B）=\prod_{i=1}^{q}(1-\lambda_iB)(\lambda_i是特征根)$可以得到：
 
$$
\begin{aligned}
\varepsilon_t & =\frac {x_t}{\Theta（B）}\\\\
& =\frac {x_t}{\prod_{i=1}^q(1-\lambda_iB)} \\\\
& = \sum_{i=1}^q{\frac {k_i}{1-\lambda_iB}}x_t(用待定系数法展开)\\\\
& = \sum_{j=0}^\infty{\sum_{i=1}^p{k_i{\lambda_i}^jB^jx_t}}\\\\
& = \sum_{j=0}^\infty{\sum_{i=1}^p{k_i{\lambda_i}^jx_{t-j}}}\\\\
& = \sum_{j=0}^\infty{G_jx_(t-j)}\\\\
\end{aligned}
$$

根据上式，可以写成MA向AR模型的转换公式：

$$
\begin{aligned}
\varepsilon_t & = \sum_{j=0}^\infty{G_jB^jx_t}\\\\
& = G(B)x_t\\\\
\end{aligned}
$$

2. MA模型可逆的条件

MA的AR形式（MA模型关于白噪声项的部分）的特征根在单位圆内，即特征根$|\lambda_i < 1|$;根据特征根和自回归系数多项式的根成倒数的性质，等价判别条件是该模型的自回归系数多项式的单位根都在单位圆外，$\Phi(\mu) = 0$ 的根$|\mu_i|>1$<br>
   - 若MA可逆，则MA的可逆性与相应的AR形式的平稳性是等价的
   - 在可逆条件下，逆转形式可以实现MA模型转变到无穷阶AR模型，那么无穷阶AR模型所有的性质**偏自相关系数无穷阶拖尾**，MA模型也有

### ARMA模型
   - 定义
   - ARMA模型有效应该具有的性质
      - 因果性
      - 可逆性
      - 无冗余性
   - 平稳性判定
   - 可逆条件判定
   - 统计性质
   
> ARMA模型定义

$$
\begin{cases}
x_t = \phi_0 + \phi_1 x_{t-1} + ... + \phi_p x_{1-p} + \varepsilon_t - \theta_1 \varepsilon_{t-1} - ... - \theta_q \varepsilon_{t-q}\\\\
\phi_p \ne 0, \theta_q \ne 0 \\\\
E(\varepsilon_t) = 0, Var(\varepsilon_t) = \sigma_{\varepsilon}^2, E(\varepsilon_t \varepsilon_s) = 0, s \ne t \\\\
E(x_s, \varepsilon_{t}) = 0, \forall s < t.
\end{cases}
$$

$\phi_0 = 0$ 时，为中心化$ARMA(p, q)$模型，简记为: $\Phi (B) x_t = \Theta (b) \varepsilon_t$
- $p$阶自回归系数多项式: $\Phi (B) = 1 - \phi_1B - \phi_2B^2 - ... - \phi_pB^p$
- $q$阶自回归系数多项式: $\Theta (B) = 1 - \theta_1 B - \theta_2 B^2 - ... - \theta_q B^q$

> ARMA模型有效应该具有的性质
1. 因果性: $ARMA(p,q) \rightarrow MA(\infty)$：$x_t$能够用滞后的$\varepsilon_t$解释
2. 可逆性：$ARMA(p,q) \rightarrow AR(\infty)$：$x_t$能够由无穷阶次滞后的$x_t$的线性组合表达，避免出现$x_t$要被未来的时序表达的可能
3. 无冗余性：$\Phi(B)x_t=\Theta(B)\varepsilon_t$中两边多项式无公因式（没有相同的特征根）

> ARMA模型平稳性判定

$p$阶自回归系数多项式$\Phi(B) = 0$的根都在单位圆外，
即$ARMA(p,q)$模型的平稳性完全由其自回归部分的平稳性决定。

给定原始模型$\Phi(B)X_t = \Theta(B)\varepsilon_t$, 令$z_t = \Theta(B)\varepsilon_t$，
因为$z_t$是关于平稳序列$\varepsilon_t$的线性组合，很容易验证$z_t$是平稳的零均值白噪声序列。
所以$\Theta(B) x_t= z_t$可以看做是一个$AR$模型。

> ARMA(p,q)模型可逆条件判定

MA(q)的系数多项式$ \Theta(B)= 0$的根都在单位圆外，
即$ARMA(p,q)$模型的可逆性完全由其移动平滑部分的可逆性决定。

给定原始模型$\Phi(B)X_t = \Theta(B)\varepsilon_t$, 令$z_t = \Phi(B)X_t$，
因为$z_t$可以看做一个新的时间序列
所以$\Theta(B)\varepsilon_t= z_t$可以看做是一个$MA$模型。

> ARMA模型的统计性质
- 传递形式和逆转形式

$$
\begin{aligned}
 X_t & = {\Phi(B）}^{-1}\Theta(B)\varepsilon_t \\\\
 & = \varepsilon_t +\sum_{j=1}^\infty{G_j\varepsilon_{t-j}}\\\\
 \varepsilon_t & = {\Theta(B)}^{-1}\Phi(B）\varepsilon_t\\\\
 & = X_t+\sum_{j=1}^\infty{I_jX_{t-j}}
 \end{aligned}
 $$
 由传递形式知，ARMA模型可以看做是无穷阶的MA模型；由逆转形式知ARMA模型可以看做无穷阶的AR模型。故ARMA模型的统计性质为：
 

| 模型       | 自相关系数 | 偏自相关系数 |
| ---------- | ---------- | ------------ |
| AR(p)      | 拖尾       | p阶截尾      |
| MA(q)      | q阶截尾    | 拖尾         |
| ARMA(p，q) | 拖尾       | 拖尾         |

## ARMA模型建模过程
- ARMA模型识别
- ARMA模型参数估计
- ARMA模型检验及优化
- ARMA模型的预测（之后看）

### ARMA模型识别

- 模型识别前的说明
- 模型识别方法

> 模型识别前的说明:对零均值平稳序列建立中心化的ARMA模型

   - 对非平稳序列处理，使其平稳
      - 方差非平稳序列：对数变换，平方根变换等（先对数据取对数）
      - 均值非平稳序列：差分变换（均值非平稳序列看图可以看出来）
   - 关于非零均值的平稳序列有两种处理方法 $E(x_t)=\mu$
      - 计算样本均值$\hat x$，令$w_t=x_t-\hat x$
      - 在模型识别阶段，对序列均值是否为0不予考虑，而在参数估计阶段，将序列均值作为一个参数予以估计

> 模型识别方法
   - 根据样本的自相关图（SACF）和偏自相关图（SPACF）表现出来的性质予以拟合。
      - 因为由于样本的随机性，样本的相关系数不会呈现出理论截尾的完美的情况，本应截尾的地方仍然会呈现出小值震荡的情况。
      - 由于平稳时间序列通常具有短期波动性，随着延迟阶数增加样本自相关系数和偏自相关系数会衰减至0附近作小值波动，什么情况下该作为相关系数截尾，什么情况下看作为相关系数在延迟若干阶之后衰减到零值附近。

| 模型       | 自相关系数 | 偏自相关系数 |
| ---------- | ---------- | ------------ |
| AR(p)      | 拖尾       | p阶截尾      |
| MA(q)      | q阶截尾    | 拖尾         |
| ARMA(p，q) | 拖尾       | 拖尾         |

   - 构造原假设：$\rho_k=0, {\hat \phi}_{kk}=0
   
根据巴雷特检验还有Q统计量检验可知，样本的自相关系数和偏自相关系数近似服从$\hat {\rho}_k \sim N(0, \frac{1}{n}),{\hat \phi}_{kk}\sim N(0, \frac{1}{n})$。利用这两个统计量，可以通过设定95%的置信区间看样本观测值是否落在这个区间。
   
   - 模型定价的经验方法：如果样本（偏）自相关系数在最初的d阶明显大于两倍的标准差范围，而后几乎95%的自相关系数都落在2倍标准差范围内，而且非零自相关系数衰减为小值波动的过程非常突然。这种，通常视为（偏）自相关系数截尾，截尾阶数为d。
   
      
### ARMA模型参数估计
   - 矩估计
   - 极大似然估计
   - 最小二乘估计法

$$
\begin{cases}
x_t = \phi_0 + \phi_1 x_{t-1} + ... + \phi_p x_{t-p} + \varepsilon_t - \theta_1 \varepsilon_{t-1} - ... - \theta_q \varepsilon_{t-q}\\\\
\phi_p \ne 0, \theta_q \ne 0 \\\\
E(\varepsilon_t) = 0, Var(\varepsilon_t) = \sigma_{\varepsilon}^2, E(\varepsilon_t \varepsilon_s) = 0, s \ne t \\\\
E(x_s, \varepsilon_{t}) = 0, \forall s < t.
\end{cases}
$$

需要估计的参数有p+q+1个：

$$
\begin{cases}
\phi = （\phi_1,\phi_2,...,\phi_p）\\\\
\theta= （\theta_1,\theta_2,...,\theta_p）\\\\
\sigma_{\varepsilon}^2=E({\varepsilon_t}^2)\\\\
\end{cases}
$$

> 矩估计

1. AR模型矩估计

由$x_t = \phi_1 x_{t-1} + ... + \phi_p x_{t-p} + \varepsilon_t$知道AR模型自相关系数有\rho_t = \phi_1 \rho_{t-1} + ... + \phi_p \rho_{t-p} ，因此可以得到

$$
\begin{cases}
\rho_1 = \phi_1 \rho_0 + \phi_2 \rho_1 ... + \phi_p \rho_{{p-1}}\\\\
\rho_2 = \phi_1 \rho_1 + \phi_2 \rho_0 ... + \phi_p \rho_{{p-2}}\\\\
...\\\\
\rho_p = \phi_1 \rho_{p-1} + \phi_2 \rho_{p-2} ... + \phi_p \rho_0\\\\
\end{cases}
$$

通过用样本自相关系数的值就可以计算出AR模型的参数估计，$\sigma_{\varepsilon}^2=E({\varepsilon_t}^2)$的估计就可以用二阶矩进行估计

2. MA模型矩估计

也是同理，通过一阶/二阶矩的计算来估计参数

3. ARMA模型矩估计

也是同理，通过一阶/二阶矩的计算来估计参数，但是会比较复杂

> 极大似然估计
- 优点<br>
用了概率分布函数，充分利用了每个已知条件。估计准确度高
- 缺点<br>
不好算

> 最小二乘估计法
非常复杂，也是求参数的一个方法

###  ARMA模型检验

在识别了模型（确定了ARMA模型的阶数）并进行参数估计以后，要考虑我们的模型是不是合理的最优的，因此需要对生成的模型进行检验和优化

- 模型的参数估计值是否具有显著性（模型参数的显著性检验）
- 检验模型的残差序列是否为白噪声序列（模型有效性检验）
#### 模型参数的显著性检验

- 目的

检验每个未知参数是否显著非0，删除不显著参数使得模型最精简（类似于多元回归单个参数的显著性检验）

- 假设条件

H0:$\beta_j=0$ H1:$\beta_j\not=0$

- 检验统计量 T分布

- 拒绝域的形式是双边的

#### 模型有效性检验

- 目的

若建立了恰当的模型，那么拟合的残差应该是白噪声序列（实际上就是检验对相关信息是否提取充分），即均值为0，常数方差，彼此不相关

- 检测对象

残差序列

- 原假设

残差序列为白噪声序列H0:$\rho_1=\rho_2=...\rho_m=0,\forall s \geq t$实际上应该取遍所有阶（m值）但实际序列大多都是低阶相关性，因此一般取$\frac n{10}$即可

- 检验统计量

LB修正的Q统计量。在原假设成立的条件下，LB修正的Q统计量会符合咖方分布$X^2（L-p-q）$

- 判定准则($\alpha$为给定的显著性水平)
$$
\begin{cases}
Q\leq X_{1-\alpha}^2(L-p-q), 接受原假设\\\\
Q> X_{1-\alpha}^2(L-p-q), 拒绝原假设
\end{cases}
$$

### ARMA模型优化

- 模型优化

当一个模型通过了检验以后，说明在一定的置信水平下，该模型能够有效的拟合观察值序列的波动，但是这种有效模型并不是唯一的，因此需要引入对模型进行比较的数量准则进行选择模型

- 选择模型方法

确定适当的比较准则，构造适当的统计量，根据统计量选出相对最优模型。这里我们用最佳准则函数定阶法，即确定出一个准则函数，该函数既要考虑某一模型拟合时对原始数据的接近程度，又要考虑模型所含待定参数的个数。建模时**使准则函数达到最小的**就是相对最优模型

> AIC准则（适用于AR\MA）
- 定义<br>
似然函数最大值越大越好，未知参数的个数越少越好。AIC准则函数为AIC(M)=-2ln(极大似然函数)+2M,其中M为模型中参数的个数
- 缺点<br>
当样本容量趋于无穷大时，AIC通常选择的未知参数比真实模型大。

> SBC准则函数（适用于AR\MA）
- 定义<br>
改良了AIC当样本容量趋于无穷大时，AIC通常选择的未知参数比真实模型大很多的缺陷。SBC准则函数为SBC(M)=-2ln(极大似然函数)+Mln(n),其中M为模型中参数的个数,n为样本容量
- 缺点<br>
具有较强的一致性，但不具有有效性。SBC准则在确定模型阶数时平均波动性比AIC的大。

> ARMA模型定阶方法
   - 分别估计所有可能模型，从（0,0）到（5,5）的ARMA模型，并记下他们的信息准则的值，取最小值对应的阶数
   - EACF(延伸自相关函数法) EACF理论上有一个由零构成的三角模式，给你截图看一下，同样，这里也是ARMA(4,10)为顶角恰好构成一个由零构成的三角模式

## GARCH模型（条件异方差模型）

### GARCH模型的重要性
### ARCH(q)
### GARCH(p,q)模型
### 模型的设定
### GARCH模型衍生

### GARCH模型的含义和重要性

当我们计算波动率的时候，如果一个收益率呈现尖峰厚尾的特征（即不是正态分布，正态分布只需要知道方差即可描述它的波动率）它的波动率就不能单纯的用序列的方差的进行描述。以短期市场利率为例，为了计算它的波动性，可以先构建ARMA模型得到它的残差序列（零均值同方差自相关系数和偏自相关系数均为0），根据ARMA的统计特征，该利率{$libor_t$}的方差完全由其残差序列的方差决定，这就意味着我们可以通过研究该残差序列的波动率来反映{$libor_t$}的波动率。在观察这组残差序列和残差平方序列以后，发现该残差平方序列有波动聚集性，也就是说残差序列的波动性在某些时段很大，在另外的时段会很小，因此通过ARCH和GARCH模型分析它的波动率

### ARCH(q)模型

- ARCH(1)模型
   - 模型表达式
   - 统计性质
   - 约束条件和稳定性
- ARCH(q)模型
   - 模型表达式
   - 统计性质
   - 约束条件和稳定性

#### ARCH(1)模型
   
> ARCH(1)模型表达式

$$
\begin{cases}
x_t=\phi_1x_{t-1}+...+\phi_px_{t-p}+u_t+\theta_1u_{t-1}+...\theta_qu_{t-q}（通过ARMA模型生成平稳残差序列）\\\\
u_t=\sigma_t\varepsilon_t(\sigma_t用来描述残差序列的波动性，即波动率)\\\\
\varepsilon_t \sim N(0,1)（残差序列有尖峰厚尾的特征，也可以改成t分布）\\\\
{\sigma_t}^2=\alpha_0+\alpha_1{u_{t-1}}^2（波动率是残差平方序列的组合）
\end{cases}
$$

说明：ARCH模型包括均值方程和   

> ARCH(1)统计性质

   - 期望：$E(u_t)=E(\sigma_t)E(\eposilon_t)$ \rightarrow 零均值
   
   - 条件异方差（波动率的衡量）

利率序列并不是标准的正态分布，所以时间序列尽管是稳定的，方差仍然不能作为波动率。又因为{$libor_t$}的ACF和PACF均不为0，也就是说前期的利率会影响后期的利率。因为我们想知道的是在知道昨天利率的情况下，今天利率会怎么波动作为波动性的考量，所以如果昨天利率对今天利率有影响的话，会对我们考察波动性造成干扰，因此通过ARMA模型消除了这种自相关性和偏自相关性的干扰，因此生成了残差序列{$u_t$}探究波动性。首先{$u_t$}是平稳序列具有同方差，但是{${u_t}^2$}具有波动聚集性的，也就是说明昨天的波动情况可能会对今天的波动情况造成影响。$u_t|u_{t-1}$表示的是在已知$u_{t-1}$的情况下，$u_t$的情况。在$u_{t-1}$已知的情况下，${\sigma_t}^2=\alpha_0+\alpha_1{u_{t-1}}^2$也是已知的，又假设$u_t=\sigma_t\varepsilon_t$，所以很容易就能得到：
$$
\begin{cases}E(u_t|u_{t-1})=E(\sigma_t\varepsilon_t)=0\\\\D(u_t|u_{t-1})=D(\sigma_t\varepsilon_t)={\sigma_t}^2D(\varepsilon_t)=\alpha_0+\alpha_1{u_{t-1}}^2\\\\\end{cases}
$$
即$u_t|u_{t-1} \sim N(0,\alpha_0+\alpha_1{u_{t-1}}^2)$，条件方差会随着时间而变化，即$D(u_t|u_{t-1}) =\alpha_0+\alpha_1{u_{t-1}}^2$

   - 方差

已知$u_t|u_{t-1} \sim N(0,\alpha_0+\alpha_1{u_{t-1}}^2)$，故：
$$
\begin{aligned}
 var(u_t) & =var(E(u_t|u_{t-1}))+E(var(u_t|u_{t-1}))\\\\
 & =0+E(\alpha_0+\alpha_1{u_{t-1}}^2)\\\\
 & = \alpha_0+\alpha_1var(u_{t-1})\\\\
 & ={\frac {\alpha_0}{1-\alpha_1}}
 \end{aligned}
$$

> ARCH模型约束条件和稳定性

   - 约束条件：$\alpha_1$在0-1
   - 稳定性：{$u_t$}的方差要为正，所以$\alpha_0$要大于0
 
#### ARCH(q)模型
   
> ARCH(q)模型表达式

$$
\begin{cases}
x_t=\phi_1x_{t-1}+...+\phi_px_{t-p}+u_t+\theta_1u_{t-1}+...\theta_qu_{t-q}（通过ARMA模型生成平稳残差序列）\\\\
u_t=\sigma_t\varepsilon_t(\sigma_t用来描述残差序列的波动性，即波动率)\\\\
\varepsilon_t \sim N(0,1)（残差序列有尖峰厚尾的特征，也可以改成t分布）\\\\
{\sigma_t}^2=\alpha_0+\alpha_1{u_{t-1}}^2+...+\alpha_q{u_{t-q}}^2（波动率是残差平方序列的组合）
\end{cases}
$$

说明：ARCH模型包括均值方程和   

> ARCH(q)统计性质

   - 期望：$E(u_t)=E(\sigma_t)E(\eposilon_t)$ \rightarrow 零均值
   
   - 条件异方差（波动率的衡量）

$$
\begin{cases}E(u_t|(u_{t-1},...,u_{t-q}))=E(\sigma_t\varepsilon_t)=0\\\\D(u_t|(u_{t-1},...,u_{t-q}))=D(\sigma_t\varepsilon_t)={\sigma_t}^2D(\varepsilon_t)=\alpha_0+\alpha_1{u_{t-1}}^2+...+\alpha_1{u_{t-q}}^2\\\\\end{cases}
$$
即$u_t|(u_{t-1},...,u_{t-q}) \sim N(0,\alpha_0+\alpha_1{u_{t-1}}^2+...+\alpha_1{u_{t-q}}^2)$，条件方差会随着时间而变化，即$D(u_t|(u_{t-1},...,u_{t-q})) =\alpha_0+\alpha_1{u_{t-1}}^2+...+\alpha_1{u_{t-q}}^2$

   - 方差

已知$u_t|(u_{t-1},...,u_{t-q}) \sim N(0,\alpha_0+\alpha_1{u_{t-1}}^2+...+\alpha_1{u_{t-q}}^2)$，故：
$$
\begin{aligned}
 var(u_t) & =var(E(u_t|(u_{t-1},...,u_{t-q})))+E(var(u_t|(u_{t-1},...,u_{t-q})))\\\\
 & =0+E(\alpha_0+\alpha_1{u_{t-1}}^2+...+\alpha_1{u_{t-q}}^2)\\\\
 & = \alpha_0+\alpha_1var(u_{t-1})+...+\alpha_qvar(u_{t-q})\\\\
 & ={\frac {\alpha_0}{1-\alpha_1-...-\alpha_q}}
 \end{aligned}
$$

> ARCH模型约束条件和稳定性

   - 约束条件：$\alpha_1+\alpha_2...+\alpha_q$在0-1之间
   - 稳定性：{$u_t$}的方差要为正，所以$\alpha_0$要大于0  
   
### GARCH(p,q)模型

ARCH模型会遇到p阶太长的一个问题，也就是说要估计的参数比较多，可能会降低模型的准确度，GARCH模型就是用来解决这个问题的，事实上GARCH（1,1）模型可以看做是无穷阶ARCH模型（类比于ARMA模型是无穷阶的MA模型）

- GARCH（1,1）
$$
\begin{cases}
x_t=\phi_1x_{t-1}+...+\phi_px_{t-p}+u_t+\theta_1u_{t-1}+...\theta_qu_{t-q}）\\\\
u_t=\sigma_t\varepsilon_t\\\\
\varepsilon_t \sim N(0,1)\\\\
{\sigma_t}^2=\alpha_0+\alpha_1{u_{t-1}}^2+\beta_1{\sigma_{t-1}}^2
\end{cases}
$$

- 期望
- 方差
- 约束条件和稳定性
