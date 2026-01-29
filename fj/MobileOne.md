这是一个非常硬核且高质量的问题！**结构重参数化（Structural Re-parameterization）** 听起来很玄乎，但一旦拆解开来，它的底层原理其实就是**线性代数的基本性质**。

我们要解决的核心问题是：**如何把一堆乱七八糟的分支（3x3, 1x1, Identity, BN）合并成唯一的一个 3x3 卷积核？**

整个过程分为三步走：
1.  **消灭 BN 层**（把 BN 的参数“吸”进卷积核里）。
2.  **统一形状**（把 1x1 和 Identity 全部伪装成 3x3 的形状）。
3.  **矩阵加法**（权重相加）。

为了方便演示，我们假设输入输出通道都是 1（单通道）。

---

### 第一步：吸星大法 —— 融合 Batch Norm (BN)

在训练时，卷积层后面通常紧跟一个 BN 层。
*   **卷积公式：** $y_{conv} = w \cdot x + b$ （$w$是权重，$b$是偏置）
*   **BN 公式：** $y_{bn} = \frac{y_{conv} - \mu}{\sigma} \cdot \gamma + \beta$
    *   $\mu$ (均值), $\sigma$ (方差) 是统计出来的。
    *   $\gamma$ (缩放), $\beta$ (平移) 是训练出来的参数。

**怎么合并？**
你可以把 BN 的公式展开，把 $y_{conv}$ 代进去：
$$ y_{bn} = \frac{(w \cdot x + b) - \mu}{\sigma} \cdot \gamma + \beta $$

经过简单的数学变换（把常数项合并），它依然是一个 $W \cdot x + B$ 的形式：
$$ y_{bn} = (\frac{\gamma}{\sigma} \cdot w) \cdot x + (\frac{\gamma}{\sigma} \cdot (b - \mu) + \beta) $$

**结论：**
无论你在卷积后面接了什么 BN，我都可以算出“一套新的权重 ($w'$) 和偏置 ($b'$)”来代替它们俩的组合。
*   **新权重** $w' = w \cdot \frac{\gamma}{\sigma}$
*   **新偏置** $b' = \beta + (b - \mu) \cdot \frac{\gamma}{\sigma}$

**现在的状态：**
原来的 `Conv -> BN` 变成了单纯的 `Conv(w', b')`。所有的分支现在都只剩下“纯卷积”了。

---

### 第二步：伪装术 —— 统一变成 3x3

现在的难题是：我们要把一个 $3\times3$ 卷积、一个 $1\times1$ 卷积和一个 Identity（恒等映射）相加。矩阵加法要求**形状（Shape）必须一样**。

#### 1. 把 1x1 变成 3x3
这很好理解。$1\times1$ 卷积只作用在中心像素上。
如果我们有一个 $3\times3$ 的卷积核，只有**中心**那个格子有值，周围一圈全是 **0**，那它的效果是不是和 $1\times1$ 一模一样？

*   假设 $1\times1$ 卷积核是 $[k]$。
*   我们把它填充（Pad）成 $3\times3$：
    $$
    \begin{bmatrix}
    0 & 0 & 0 \\
    0 & k & 0 \\
    0 & 0 & 0
    \end{bmatrix}
    $$

#### 2. 把 Identity (x=x) 变成 3x3
Identity 层就是直接把输入复制到输出，不做任何改变。
在卷积的世界里，什么样的卷积核能做到“复制输入”？
答案是：**中心为 1，其余全为 0 的卷积核**（这叫狄拉克 $\delta$ 函数的离散形式）。

*   我们构造一个这样的 $3\times3$ 核：
    $$
    \begin{bmatrix}
    0 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 0
    \end{bmatrix}
    $$
*   *注意：如果是多通道，只有在同一个通道（Depthwise）或者对应通道（Pointwise）的位置中心是 1，其他通道也是 0。*

---

### 第三步：万法归一 —— 权重相加

现在，训练时的三个分支（3x3分支，1x1分支，Identity分支）都已经处理好了：
1.  都先融合了各自的 BN 层，变成了纯粹的权重矩阵。
2.  都通过补 0 的方式，变成了标准的 $3\times3$ 矩阵。

**卷积的线性性质：**
$$ \text{Conv}(x, W_1) + \text{Conv}(x, W_2) = \text{Conv}(x, W_1 + W_2) $$

**最后的动作：**
我们只需要把这三个 $3\times3$ 的矩阵**对应位置相加**，得到一个最终的矩阵 $W_{final}$ 和最终的偏置 $B_{final}$。

$$ W_{final} = W_{3\times3} + \text{Pad}(W_{1\times1}) + \text{IdentityMatrix} $$

**推理时：**
我们扔掉所有的分支，只新建一个标准的 $3\times3$ 卷积层，把 $W_{final}$ 和 $B_{final}$ 填进去。搞定！

---

### 举个直观的数字例子

假设训练时某一层学到了以下参数（简化版）：

1.  **3x3 分支：** 权重全为 1（假设）。
    $$
    K_A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
    $$
2.  **1x1 分支：** 权重是 2。
    $$
    K_B = [2] \xrightarrow{\text{补0}} \begin{bmatrix} 0 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 0 \end{bmatrix}
    $$
3.  **Identity 分支：** 等效权重是 1。
    $$
    K_C = \text{Id} \xrightarrow{\text{构造}} \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
    $$

**重参数化后的最终卷积核：**
$$
K_{final} = K_A + K_B + K_C =
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1+2+1 & 1 \\
1 & 1 & 1
\end{bmatrix} =
\begin{bmatrix}
1 & 1 & 1 \\
1 & 4 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$

你在手机上跑的时候，就只跑这一个 $K_{final}$。

### MobileOne 的独特之处

MobileOne 之所以被称为 MobileOne，是因为它把这个技术用到了极致。它针对 MobileNet 的结构（Depthwise Conv + Pointwise Conv）分别做了重参数化：

1.  **Depthwise 重参数化：**
    *   训练时：$3\times3$ DW + $1\times1$ DW + Identity。
    *   融合后：一个 $3\times3$ DW。

2.  **Pointwise 重参数化：**
    *   训练时：$1\times1$ Conv + ... (可能有多个冗余的 $1\times1$ 分支)。
    *   融合后：一个 $1\times1$ Conv。

### 为什么要这么麻烦？

你可能会问：**“既然最后等效于一个 3x3 卷积，那我训练时直接用一个 3x3 不就好了吗？为什么要搞三个分支再合并？”**

这是一个深刻的深度学习理论问题：**过参数化（Over-parameterization）有助于优化。**

*   **损失函数的曲面：** 单个 $3\times3$ 卷积的解空间可能比较“陡峭”，容易陷入局部最优解。
*   **多分支的优势：** 加入 $1\times1$ 和 Identity 分支，相当于给梯度下降提供了“多条捷径”。Identity 分支特别重要，它保证了梯度可以无损地传导到前面的层（像 ResNet 那样），让网络更容易训练得很深。

**总结 MobileOne 的哲学：**
*   **训练时：** 我要“复杂”，因为复杂好训练，精度高。
*   **推理时：** 我要“简单”，因为简单跑得快。
*   **手段：** 数学变换（线性代数）是连接这两者的桥梁。