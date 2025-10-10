# Transformer-KAN: Replacing Feed-Forward Layers with Kolmogorov–Arnold Networks

This project explores a **novel Transformer architecture** in which the standard Feed-Forward Neural Networks (FFNs) in both the encoder and decoder blocks are replaced by **Kolmogorov–Arnold Networks (KANs)**.  
The goal is to evaluate how this substitution affects the model's performance — particularly its accuracy, loss, and perplexity — on the **Tiny Shakespeare dataset**.

---

## 🧠 Background

### 1. The Kolmogorov–Arnold Representation Theorem

The **Kolmogorov–Arnold Representation Theorem (KART)** states that any multivariate continuous function  
\\( f : [0,1]^n \to \mathbb{R} \\)  
can be represented as a finite superposition of continuous univariate functions and addition:

\[
f(x_1, x_2, \dots, x_n) = \sum_{q=1}^{2n+1} \phi_q \left( \sum_{p=1}^{n} \psi_{pq}(x_p) \right)
\]

Here:
- \( \psi_{pq}(\cdot) \) are **inner functions** applied to individual input dimensions,  
- \( \phi_q(\cdot) \) are **outer functions** applied after summation,  
- and the entire model is a **superposition of simpler functions** rather than a pure composition as in traditional MLPs.

---

### 2. FFN vs. KAN: Mathematical Contrast

#### Conventional Feed-Forward Layer (MLP)
The MLP-based Feed-Forward Network within a Transformer block performs:
\[
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
\]

where:
- \( x \in \mathbb{R}^{d} \) is the input vector,  
- \( W_1, W_2 \) are learnable weight matrices,  
- \( b_1, b_2 \) are biases, and  
- \( \sigma(\cdot) \) is a nonlinear activation such as ReLU or GELU.  

This operation applies **linear transformations followed by nonlinearities** — a *composition* of functions.

---

#### Kolmogorov–Arnold Network (KAN) Feed-Forward Layer
By contrast, the **KAN-based layer** models each connection between nodes as a **learnable univariate spline function**, forming a *superposition* of additive univariate mappings:

\[
\text{KAN}(x) = \sum_{i=1}^{m} \phi_i \left( \sum_{j=1}^{d} \psi_{ij}(x_j) \right)
\]

where:
- \( \psi_{ij}(\cdot) \) are inner spline functions acting on individual input coordinates,  
- \( \phi_i(\cdot) \) are outer spline functions combining these contributions,  
- \( m \) is the number of functional aggregations.  

Thus, **each connection encodes a continuous, piecewise polynomial relationship** instead of a static weight — giving the model smoother and more interpretable transformations.

---

### 3. KANs vs. MLPs Summary

| Aspect | MLP (Feed-Forward) | KAN (Kolmogorov–Arnold Network) |
|:-------|:-------------------|:--------------------------------|
| Representation | Linear → Nonlinear → Linear | Superposition of univariate spline functions |
| Mapping | Matrix multiplication | Additive spline aggregation |
| Interpretability | Black-box weights | Explicit functional mapping |
| Smoothness control | Implicit via activation | Explicit via spline degree/knots |

---

## ⚙️ Architecture Overview

The modified architecture maintains the **Transformer encoder–decoder structure**, but with its feed-forward sublayers swapped out:

