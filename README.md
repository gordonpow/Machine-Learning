# Machine-Learning
# HW1 推導與實作說明

本作業主要針對以下兩個模型進行訓練過程的數學推導與程式實現說明：

1. SVM（Support Vector Machine）
2. MLP（Multilayer Perceptron）

---

## 📘 1. SVM 的訓練過程

### 🎯 目標

對於訓練資料 \( (x_i, y_i) \)，其中 \( y_i \in \{+1, -1\} \)，SVM 希望找到一個分類超平面：

\[
f(x) = \vec{w}^T x + b
\]

使得所有資料點被正確分類且**間隔最大**。

---

### ✅ 最佳化問題（硬間隔 SVM）

\[
\min_{\vec{w}, b} \frac{1}{2} ||\vec{w}||^2 \\
\text{subject to } y_i (\vec{w}^T x_i + b) \geq 1 \quad \forall i
\]

透過拉格朗日乘數法可以導出：

\[
\vec{w} = \sum_i \alpha_i y_i x_i
\]

---

### 🧠 梯度下降形式（Soft Margin + Hinge Loss）

將問題轉為 soft-margin SVM 的形式：

\[
\min_{\vec{w}} \frac{\lambda}{2} ||\vec{w}||^2 + \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_i \vec{w}^T x_i)
\]

透過 SGD 更新權重：

\[
\vec{w}^* = \vec{w} + \Delta \vec{w}
\]

其中：

\[
\Delta \vec{w} = 
\begin{cases}
-\eta \lambda \vec{w} + \eta y_i x_i & \text{if } y_i \vec{w}^T x_i < 1 \\
-\eta \lambda \vec{w} & \text{otherwise}
\end{cases}
\]

---

## 📘 2. MLP 的訓練過程

### 🎯 目標

透過多層神經網路對輸入 \( x \) 預測輸出 \( \hat{y} \)，並最小化損失函數（如 MSE 或交叉熵）。

---

### ✅ 前向傳播（Forward Pass）

每層的計算為：

\[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \\
a^{(l)} = \sigma(z^{(l)})
\]

最後輸出：

\[
\hat{y} = a^{(L)}
\]

---

### ✅ 損失函數（以 MSE 為例）

\[
L = \frac{1}{2} ||y - \hat{y}||^2
\]

---

### ✅ 反向傳播（Backpropagation）

從最後一層往前推：

\[
\delta^{(L)} = \nabla_a L \odot \sigma'(z^{(L)}) \\
\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})
\]

---

### ✅ 權重更新

對每層使用梯度下降：

\[
\Delta W^{(l)} = -\eta \cdot \delta^{(l)} (a^{(l-1)})^T \\
W^{(l)} \leftarrow W^{(l)} + \Delta W^{(l)}
\]

程式中表示為：

```python
W = W + delta_W  # delta_W = -learning_rate * gradient
