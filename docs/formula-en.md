
# Anchor-based Sparse Subspace Incomplete Multi-view Clustering


## 1. Preliminaries

### 1.1 Symbols

- $A^{(v)} \in R^{n_a \times d^{(v)}}$ is the view-complete subset for each view. Each sample has all the views. Its size is $n_a$.
- $U^{(v)} \in R^{n^{(v)}_u \times d^{(v)}}$ is the incomplete subset for each view. Each sample only has parts of the views. Its size is $n^{(v)}_u, v = 1,2,\cdots,V$.
- $n^{(v)}$ is the number of existent samples of each view.
- $n_a$ is the number of samples of the complete subset.
- $n^{(v)}_u$ is the number of samples of the incomplete subset of each view.
- $d^{(v)}$ is the original feature dimension of each view.
- $n$ is the total number of samples.
- $V$ is the number of views.
- $c$ is the number of clusters, which is given by the user.


### 1.2 Inputs

- $X^{(v)} \in R^{ n \times d^{(v)} }$ are all the samples of each view, where inexistent samples are marked with NaN.
- $\bar{X}^{(v)} = [A^{(v)}; U^{(v)}] \in R^{ n^{(v)} \times d^{(v)} }$ are the reorganized samples of each view, where inexistent samples are dropped and the complete subset is arranged before the incomplete subset.
- $W^{(v)} \in \{0,1\}^{n \times n^{(v)}}$ is the missing indicator matrix.
- $\lambda$ is the balancing parameter in the loss function.

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


### 1.3 Outputs

- $Z \in R^{ n \times d^{(v)} }$ is the complete consensus anchor graph learned by our method.
- $C \in \{0,1\}^{n \times c}$ are the clustering results；


## 2. Our method

### 2.1 Learning an inter-view consensus sparse subspace matrix for the complete subset

Since each sample of the complete subset possesses all the views, the learned $Z_a$ is the consensus subspace across the views, which fuses information from different views.

$$
\min \sum_{v=1}^V \left\Vert Z_a^{(v)} A^{(v)} - A^{(v)} \right\Vert _F^2 + \lambda \left\Vert Z_a^{(v)} - Z_a \right\Vert _F^2 \\
\text{s.t. } Z_a^{(v)} \geq 0, Z_a^{(v)} 1 = 1, diag(Z_a^{(v)}) = 0 \\
             Z_a \geq 0, Z_a 1 = 1, diag(Z_a) = 0 \\
$$

- $A^{(v)}$ is the complete subset of view $v$.
- $Z_a^{(v)} \in R^{n_a \times n_a}$ is the subspace matrix of the complete subset of view $v$ constructed by self-representative property.
- $Z_a \in R^{n_a \times n_a}$ is the consensus subspace matrix of the complete subset, that is, the centroid of $\{Z_a^{(v)}\}_{v=1}^V$.


### 2.2 Learning the intra-view anchor-based subspace matrices for the incomplete subset

- Since the samples of the incomplete subset are not perfectly aligned, no inter-view consensus subspace can be learned.
- Yet within each view, we can linearly combine samples of $A^{(v)}$ to approximate $U^{(v)}$ to learn their relationships as $Z_u^{(v)} \in R^{{n_u}^{(v)} \times n_a}$.
- Then, concatenate two subspace matrices $Z_a, Z_u^{(v)}$ within each view to obtain the anchor graph (bipartile graph) of that view $Z^{(v)} \in R^{{n}^{(v)} \times n_a}$.

$$
\min \sum_{v=1}^V \left\Vert Z_u^{(v)} A^{(v)} - U^{(v)} \right\Vert _F^2 \\
\text{s.t. } Z_u^{(v)} \geq 0, Z_u^{(v)} 1 = 1 \\
$$


### 2.3 Learning the complete consensus anchor graph

Putting the above 1. inter-view consensus sparse subspace learning and 2. intra-view anchor-based sparse subspace learning into a unified optimization problem.


- Compute the anchor graph of the existent samples of each view as $Z^{(v)} = [ Z_a; Z_u^{(v)} ] \in R^{n^{(v)} \times n_a}$.
- Compute the complete consensus anchor graph as: $Z = \frac{1}{V} \sum_{v=1}^V W^{(v)} Z^{(v)}$ .
- Perform Fast Spectral Clustering on $Z$ and obtain the final clustering results $C$.


$$
\min \sum_{v=1}^V \{ \left\Vert Z_a^{(v)} A^{(v)} - A^{(v)} \right\Vert _F^2 +
\lambda \left\Vert Z_a^{(v)} - Z_a \right\Vert _F^2 +
\left\Vert Z_u^{(v)} A^{(v)} - U^{(v)} \right\Vert _F^2 \} \\
$$

$$
\text{s.t. } Z_a^{(v)} \geq 0, Z_a^{(v)} 1 = 1, diag(Z_a^{(v)}) = 0 \\
             Z_a \geq 0, Z_a 1 = 1, diag(Z_a) = 0 \\
             Z_u^{(v)} \geq 0, Z_u^{(v)} 1 = 1
$$


### 2.4 Equivalent unconstrained loss and its optimization

We transform the above objective function into an equivalent unconstrained form to take advantage of the auto-gradient and hardware acceleration of PyTorch. 

- Use masked softmax activation function $\sigma_M(X)$ to substitute constrains $Z \geq 0, Z 1 = 1, diag(Z)=0$.
- Use mean squared error $\mathcal{L}_{MSE}(X, Y)$ to implement F-norm error, i.e., $\left\Vert X - Y \right\Vert_F^2$.
- The expectant subspace matrices $\{Z_a,Z_a^{(v)},Z_u^{(v)}\}_{v=1}^V$ are solved by the learnable variables $\{\Theta_a,\Theta_a^{(v)},\Theta_u^{(v)}\}_{v=1}^V$, respectively.
- Usee Adam to minimize the loss $\mathcal{L}_{\text{ABS2-IMC}}$.

$$
\mathcal{L}_{\text{ABS2-IMC}} = 
\sum_{v=1}^V \{ \mathcal{L}_{MSE} \left( \sigma_M(\Theta_a^{(v)}) A^{(v)}, A^{(v)} \right) +
\lambda \mathcal{L}_{MSE} ( \Theta_a^{(v)} , \Theta_a ) +
\mathcal{L}_{MSE} \left( \sigma(\Theta_u^{(v)}) , U^{(v)} \right) \}
$$

The definition of the masked softmax activation function is as follows:

$$
\sigma_M(X_{i,j}) = \frac{\exp(X_{i,j}) \cdot M_{i,j}}{ \sum_{j=1}^{n} \exp(X_{i,j}) \cdot M_{i,j}}
$$


## 3. Implementation details



### 3.1 Preprocessing steps

1. Normalize each feature to a real number in [0, 1];

$$
\tilde{X} = \frac{X - \min_i \{X_i\}}{\max_i \{X_i\}}
$$

2. Reorganize the normalized features into the form of $\bar{X}^{(v)} = [A^{(v)} ; U^{(v)}]$, where:
   - $A^{(v)} \in R^{n_a \times d^{(v)}}$ is the complete subset for each view.
   - $U^{(v)} \in R^{n^{(v)}_u \times d^{(v)}}$ is the incomplete subset for each view.
   - The relationship between $\bar{X}^{(v)}$ and ${X}^{(v)}$ is: ${X}^{(v)} = W^{(v)} \bar{X}^{(v)}$；



### 3.2 Postprocessing steps

1. Obtain $\{Z_a,Z_u^{(v)}\}_{v=1}^V$ from learnable parameters $\{\Theta_a,\Theta_u^{(v)}\}_{v=1}^V$ as follows:
   - $Z_a = \sigma_m(\Theta_a, M)$ .
   - $Z_u^{(v)} = \sigma(\Theta_u^{(v)})$ .


2. Obtain $Z$ from $Z_a,\{Z_u^{(v)}\}_{v=1}^V$ as follows:
   - $Z^{(v)} = [ Z_a ; Z_u^{(v)} ] \in R^{n^{(v)} \times n_a} $.
   - $Z = \frac{1}{V} \sum_{v=1}^V W^{(v)} Z^{(v)}$ .


3. Perform clustering on $Z$ as follows:
   - Perform singular decomposition on $Z$, i.e., $Z=P\Sigma Q^T$.
   - The left singular matrix of $Z$ (i.e., $P \in R^{n \times c}$) is the spectral embeddings of $Z$.
   - Take the first $c$ features of $P$ as we have $c$ clusters.
   - Perform normalization on $P$, i.e., $\tilde{P}_i = \frac{P_i}{||P_i||_F}$.
   - Perform K-means on the normaliezd $\tilde{P}_i$ to get $C \in \{0, 1\}^{n \times c}$.


### 3.3 Hyper-parameters settings

- Use Adam optimizer with a learning rate set to 0.1.
- Hyper-parameter $\lambda$ is set to 0.1.
