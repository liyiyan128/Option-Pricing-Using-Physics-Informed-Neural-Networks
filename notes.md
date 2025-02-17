# Deriving the PDE Formulation for American Options

Below is a common way to see how one derives the partial-differential-equation (PDE) formulation (often called a "complementary formulation" or "free-boundary problem") for pricing an American option under the standard Black–Scholes framework. For concreteness, let us consider an American put option with strike $K$, although the same reasoning applies (with small adjustments) to American calls (especially if there are dividends) or other payoffs.

---

## 1. Setup and Notation

We assume:
- The underlying asset $S_t$ follows the (risk-neutral) stochastic differential equation:
  $$
  dS_t = rS_t \, dt + \sigma S_t \, dW_t,
  $$
  where $r$ is the constant risk-free interest rate, $\sigma$ is the volatility, and $W_t$ is a standard Brownian motion under the risk-neutral measure.
- The payoff of an American **put** option is:
  $$
  g(S) = (K - S)^+ = \max(K - S, 0).
  $$
- Let $V(t, S)$ be the fair value of the American put at time $t$ when the underlying’s price is $S$. We wish to find $V(t, S)$ for $0 \le t \le T$.

Because it is an **American** option, it can be exercised at any time $\tau \le T$. Hence,
$$
V(t, S) = \sup_{t \le \tau \le T} \mathbb{E}^\mathbb{Q} \left[ e^{-r(\tau - t)} g(S_\tau) \mid S_t = S \right],
$$
where $\mathbb{Q}$ is the risk-neutral measure.

---

## 2. Inequality Constraints (Early Exercise Condition)

Because the holder of the American put can choose whether to continue holding the option or to exercise immediately, the value $V$ must satisfy:

1. **Value never less than the immediate payoff** (no one would voluntarily accept less than the intrinsic payoff):
   $$
   V(t, S) \ge g(S) = (K - S)^+.
   $$

2. **If the option is not exercised immediately (i.e., in the continuation region)**, the value $V$ should satisfy the same PDE as in the European case, namely the Black–Scholes PDE:
   $$
   \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0.
   $$

3. **At exercise** (in the exercise region), we have:
   $$
   V(t, S) = g(S).
   $$

Hence, the American option value must solve a **free-boundary problem**: part of the $(t, S)$ domain is a "continuation region" (where it is optimal to hold), and part is an "exercise region" (where it is optimal to exercise). Across the boundary between these two regions, certain smooth-pasting conditions or inequalities must hold.

---

## 3. The "Complementary Formulation"

A convenient way to encode both "the PDE holds unless it’s optimal to exercise" and "the option value is at least the payoff" is via an inequality/PDE pair or, more compactly, via a "$\max$ (or $\min$)" formulation.

### 3.1 Inequality Form

One way (common in financial mathematics texts) is to write two conditions simultaneously:

1. **Linear PDE Inequality**:
   $$
   \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \le 0.
   $$

2. **Early Exercise Payoff Constraint**:
   $$
   V \ge g(S).
   $$

Along with the terminal (final) condition at maturity:
$$
V(T, S) = g(S),
$$
and appropriate boundary conditions for $S \to 0$ or $S \to \infty$.

In the region where $V > g(S)$ (strictly better to hold than to exercise), the inequality in the PDE must be tight (becomes an equality). Conversely, where $V = g(S)$, one typically gets slack in the PDE (i.e., one no longer needs to satisfy equality there).

### 3.2 “$\min$” or “$\max$” Formulation

Another compact way is to write a **single** equation with a minimum or maximum operator. A common version is:
$$
\min \left\{ V - g(S), \; \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \right\} = 0.
$$

Equivalently (depending on the author),
$$
\max \left\{ g(S) - V, \; -\left( \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \right) \right\} = 0.
$$

These formulations encode the same constraints:
1. **Either** $V - g(S) = 0$ (i.e., exercise now)  
   **or** $\frac{\partial V}{\partial t} + \cdots - r V = 0$ (i.e., continue holding).

2. On any boundary separating exercise/continuation regions, "smooth pasting" or consistency conditions typically apply to ensure that $V$ is continuous and sufficiently differentiable.

---

## 4. Final Statement of the American Option Pricing PDE

Putting it all together, **the PDE formulation for an American put** (with strike $K$) is frequently stated as:

1. **Value satisfies**  
   $$
   \begin{cases}
     \max \left( (K - S)^+ - V, \; -\left( \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \right) \right) = 0, \\[6pt]
     V(T, S) = (K - S)^+,
   \end{cases}
   $$
   together with standard boundary conditions for large or small $S$.

2. Equivalently, in an **inequality system** form:  
   $$
   \begin{cases}
     V \ge (K - S)^+, \\[6pt]
     \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \le 0, \\[6pt]
     V(T, S) = (K - S)^+.
   \end{cases}
   $$

---

### Summary

- **American vs. European**: A European put satisfies the standard Black–Scholes PDE everywhere (with a terminal condition). An **American** put must additionally satisfy the constraint $V \ge (K - S)^+$ at all times and allows for exercise prior to $T$.  
- **Mathematically**: This leads to a *free boundary* (the early exercise boundary) that is not known in advance but is determined as part of the solution.  
- **In PDE form**: The solution satisfies a *complementary* system of PDE/inequalities or, more compactly, a **variational inequality** or **min/max** condition.  

This is the **PDE formulation for American option pricing** under the standard Black–Scholes assumptions.
