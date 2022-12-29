
# 基于锚点稀疏子空间学习的非完整多视角聚类


## 1. 前言

### 1.1 符号含义

- $A^{(v)} \in R^{n_a \times d^{(v)}}$ 为每个视角的完整子集，即在所有视角都存在的样本，其样本数为 $n_a$；
- $U^{(v)} \in R^{n^{(v)}_u \times d^{(v)}}$ 为每个视角的非完整子集，即不是在所有视角都存在的样本，其样本数为 $n^{(v)}_u, v = 1,2,\cdots,V$；
- $n^{(v)}$ 为每个视角的存在样本数量；
- $n_a$ 为完整子集的样本数量；
- $n^{(v)}_u$ 为非完整子集在每个视角的样本数量；
- $d^{(v)}$ 为和每个视角的原始特征维度；
- $n$ 为样本总数量；
- $V$ 为视角数量；
- $c$ 为类簇数量；


### 1.2 输入

- ${X}^{(v)} \in R^{ n \times d^{(v)} }$ 为每个视角的所有样本（不存在的样本标记为NaN）；
- $\bar{X}^{(v)} = [A^{(v)} ; U^{(v)}] \in R^{ n^{(v)} \times d^{(v)} }$ 为去掉不存在的样本并按照完整子集在前、非完整子集在后排列的数据；
- $W^{(v)} \in \{0,1\}^{n \times n^{(v)}}$ 为缺失指示矩阵；
- $\lambda$ 为损失函数中的权衡参数，是超参数；

$$
W^{(v)}_{i,j} = \begin{cases}
1\text{, if the $i$-th sample is the $j$-th sample presented in $v$-th view}; \\
0\text{, otherwise}.
\end{cases}
$$

$$
{X}^{(v)} = W^{(v)} \bar{X}^{(v)}
$$

$$
n^{(v)} = \left( n_a + n^{(v)}_u \right) < n\\
v = 1,2,\cdots,V
$$


### 1.3 输出

- $Z \in R^{ n \times d^{(v)} }$ 方法学到的共识完整锚点图；
- $C \in \{0,1\}^{n \times c}$ 是聚类结果；


## 2. 本文方法

### 2.1 完整子集的视角间共识稀疏子空间矩阵的学习

由于完整子集中每个样本都拥有全部的视角，因此学到的$Z_a$是视角间的共识子空间，融合了不同视角的信息。

$$
\min \sum_{v=1}^V \left\Vert Z_a^{(v)} A^{(v)} - A^{(v)} \right\Vert _F^2 + \lambda \left\Vert Z_a^{(v)} - Z_a \right\Vert _F^2 \\
\text{s.t. } Z_a^{(v)} \geq 0, Z_a^{(v)} 1 = 1, diag(Z_a^{(v)}) = 0 \\
             Z_a \geq 0, Z_a 1 = 1, diag(Z_a) = 0 \\
$$

- $A^{(v)}$ 为视角$v$的完整子集；
- $Z_a^{(v)} \in R^{n_a \times n_a}$为视角$v$的完整子集上利用自表示性质构建的子空间矩阵，即完整子集的视角专属子空间矩阵；
- $Z_a \in R^{n_a \times n_a}$ 为完整子集的共识子空间矩阵，即 $\{Z_a^{(v)}\}_{v=1}^V$ 的共同的质心；


### 2.1 非完整子集的视角内锚点子空间矩阵的学习

- 非完整子集的各个视角不能完全对齐，无法学习视角间的共识子空间，
- 但在每个视角内可以用完整子集 $A^{(v)}$来线性表示对应的非完整子集 $U^{(v)}$，以学到视角内锚点子空间矩阵 $Z_u^{(v)} \in R^{{n_u}^{(v)} \times n_a}$。
- 然后，在每个视角 $v$内拼接两个子空间矩阵 $Z_a, Z_u^{(v)}$，就能得到该视角的所有存在样本到完整样本的锚点子空间矩阵 $Z^{(v)} \in R^{{n}^{(v)} \times n_a}$。

$$
\min \sum_{v=1}^V \left\Vert Z_u^{(v)} A^{(v)} - U^{(v)} \right\Vert _F^2 \\
\text{s.t. } Z_u^{(v)} \geq 0, Z_u^{(v)} 1 = 1 \\
$$


### 2.2 各视角存在样本锚点图和完整共识锚点图的学习

将上述的1. 完整子集的视角间共识稀疏子空间矩阵的学习；2. 非完整子集的视角内锚点子空间矩阵的学习；放入统一框架中进行优化。

$$
\min \sum_{v=1}^V \left\{ \left\Vert Z_a^{(v)} A^{(v)} - A^{(v)} \right\Vert _F^2 +
\lambda \left\Vert Z_a^{(v)} - Z_a \right\Vert _F^2 +
\left\Vert Z_u^{(v)} A^{(v)} - U^{(v)} \right\Vert _F^2 \right\} \\
\text{s.t. } Z_a^{(v)} \geq 0, Z_a^{(v)} 1 = 1, diag(Z_a^{(v)}) = 0 \\
             Z_a \geq 0, Z_a 1 = 1, diag(Z_a) = 0 \\
             Z_u^{(v)} \geq 0, Z_u^{(v)} 1 = 1
$$

- 各视角存在样本锚点图计算如下： $Z^{(v)} = [ Z_a ; Z_u^{(v)} ] \in R^{n^{(v)} \times n_a}$ ；
- 完整共识锚点图计算如下： $Z = \frac{1}{V} \sum_{v=1}^V W^{(v)} Z^{(v)}$ ；
- 对共识锚点图 $Z$进行快速谱聚类（Fast Spectral Clustering）即可获得最终聚类结果$C$ ；


### 2.3 等价的无约束损失函数及其优化

为了便于使用PyTorch的基于自动微分的优化算法，加快优化效率，将上述目标函数等价改写为一个等价的无约束损失函数。

- 用掩码softmax激活函数$\sigma_M(X)$实现约束条件 $Z \geq 0, Z 1 = 1, diag(Z)=0$，即：
  - $Z$的元素大于等于0；
  - $Z$的每行元素之和都为1；
  - $Z$的对角线元素为0；
- 使用均方误差损失 $\mathcal{L}_{MSE}(X, Y)$来实现F范数误差，即$\left\Vert X - Y \right\Vert_F^2$；
- 待求的子空间矩阵 $\{Z_a,Z_a^{(v)},Z_u^{(v)}\}_{v=1}^V$分别用可学习参数 $\{\Theta_a,\Theta_a^{(v)},\Theta_u^{(v)}\}_{v=1}^V$ 表示；
- 使用Adam优化算法最小化损失函数 $\mathcal{L}_{\text{ABSS-IMC}}$ 即可；


$$
\mathcal{L}_{\text{ABS2-IMC}} = 
\sum_{v=1}^V \left\{ \mathcal{L}_{MSE} \left( \sigma_M(\Theta_a^{(v)}) A^{(v)}, A^{(v)} \right) +
\lambda \mathcal{L}_{MSE} \left( \Theta_a^{(v)} , \Theta_a \right) +
\mathcal{L}_{MSE} \left( \sigma(\Theta_u^{(v)}) , U^{(v)} \right)
\right\}
$$

掩码softmax激活函数定义如下：
$$
[\sigma_M(X)]_{i,j} = \frac{\exp(X_{i,j}) \cdot M_{i,j}}{\sum_{j=1}^n \exp(X_{i,j}) \cdot M_{i,j}}
$$


## 3. 实现细节

### 3.1 前处理步骤

- 归一化：将每个特征归一化为0-1之间的实数；

$$
\tilde{X} = \frac{X - \min_i \{X_i\}}{\max_i \{X_i\}}
$$

- 将归一化后的数据重新组织为$\bar{X}^{(v)} = [A^{(v)} ; U^{(v)}]$，其中；
  - $A^{(v)} \in R^{n_a \times d^{(v)}}$为每个视角的完整子集，即在所有视角都存在的样本，其样本数为 $n_a$；
  - $U^{(v)} \in R^{n^{(v)}_u \times d^{(v)}}$为每个视角的非完整子集，即不是在所有视角都存在的样本，其样本数为 $n^{(v)}_u, v = 1,2,\cdots,V$；
  - 重新组织后的数据$\bar{X}^{(v)}$与原始数据${X}^{(v)}$之间的关系为：${X}^{(v)} = W^{(v)} \bar{X}^{(v)}$；




### 3.2 后处理步骤

- 从可学习参数 $\{\Theta_a,\Theta_u^{(v)}\}_{v=1}^V$得到 $\{Z_a,Z_u^{(v)}\}_{v=1}^V$ 如下：
  - $Z_a = \sigma_M(X)$；
  - $Z_u^{(v)} = \sigma(\Theta_u^{(v)})$；
- 从$Z_a,\{Z_u^{(v)}\}_{v=1}^V$得到完整共识锚点图$Z$ 如下：
  - $Z^{(v)} = [ Z_a ; Z_u^{(v)} ] \in R^{n^{(v)} \times n_a} $ ；
  - $Z = \frac{1}{V} \sum_{v=1}^V W^{(v)} Z^{(v)}$ ；
- 对完整共识锚点图$Z$进行快速谱聚类如下：
  - 对$Z$进行奇异值分解，即 $Z=P\Sigma Q^T$；
  - $Z$ 的左奇异向量$P \in R^{n \times c}$即为$Z$的谱嵌入，取原始的左奇异向量$P$的前$c$个特征；
  - 对$P$的模长进行归一化，即$\tilde{P}_i = \frac{P_i}{||P_i||_F}$；
  - 对归一化后的$\tilde{P}_i$进行K-means聚类，得到最终聚类结果$C \in \{0, 1\}^{n \times c}$；


### 3.3 超参数设置

- 使用 Adam 优化器，学习率设置为0.001；
- 超参数 $\lambda$ 设置为0.1；


