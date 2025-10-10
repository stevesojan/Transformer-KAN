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

## 2. FFN vs. KAN: Mathematical Contrast

### Conventional Feed-Forward Layer (MLP)

The MLP-based Feed-Forward Network within a Transformer block performs the familiar two-layer mapping:

$$
\mathrm{FFN}(x) \;=\; \sigma\big(x W_1 + b_1\big)\, W_2 + b_2
$$

where:
- \(x \in \mathbb{R}^{d}\) is the input vector,  
- \(W_1 \in \mathbb{R}^{d \times d_{ff}},\, W_2 \in \mathbb{R}^{d_{ff} \times d}\) are learnable weight matrices,  
- \(b_1 \in \mathbb{R}^{d_{ff}},\, b_2 \in \mathbb{R}^d\) are biases, and  
- \(\sigma(\cdot)\) is a nonlinear activation such as ReLU or GELU.

This operation applies **linear transformations followed by nonlinearities** — i.e. a *composition* of functions.

---

### Kolmogorov–Arnold Network (KAN) Feed-Forward Layer

By contrast, a KAN-based layer uses the Kolmogorov–Arnold superposition principle and models the mapping as a sum of univariate function compositions:

$$
\mathrm{KAN}(x) \;=\; \sum_{i=1}^{m} \phi_i\!\left(\sum_{j=1}^{d} \psi_{ij}(x_j)\right)
$$

where:
- \(\psi_{ij}(\cdot)\) are **inner univariate functions** (e.g., learned splines) applied to each input coordinate \(x_j\),  
- \(\phi_i(\cdot)\) are **outer univariate functions** that combine the inner results,  
- \(m\) is the number of aggregated terms (controls expressivity).

Thus each connection is modeled as a **continuous, piecewise-polynomial (spline) mapping** rather than a fixed scalar weight; the layer is an explicit *superposition* of simpler univariate components.

---

### Quick summary

| Aspect | MLP (FFN) | KAN |
|---|---:|:---|
| Core operation | \( \sigma(xW_1+b_1)W_2 + b_2 \) | \( \sum_i \phi_i(\sum_j \psi_{ij}(x_j)) \) |
| Parameterization | Dense matrices + activations | Univariate functions (splines) + additive aggregation |
| Smoothness control | Implicit (via activations) | Explicit (spline degree / knots) |
| Interpretability | Lower | Higher (functional form per connection) |

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

