# Unified CLF–CBF Rate-Optimized Variable Impedance Control for Safe Robotic Manipulation

This repository implements a simulation benchmark framework evaluating **Unified Control Lyapunov Function (CLF) and Control Barrier Function (CBF)** optimization against state-of-the-art contact force control methods. The environment simulates contact-rich wiping tasks using a 7-DoF Franka Emika Panda robot in `robosuite` / MuJoCo, featuring real-time joint-space torque control, operational-space dynamics, force-torque (FT) sensor noise/gravity compensation, and dynamic parameter optimization.



## 🚀 Key Theoretical Contributions

### 1. Proposed Framework: Unified CLF–CBF Rate-Optimized VIC
* **Dual-Layer Co-Optimization:** Solves a real-time Quadratic Program (OSQP) at $200\text{ Hz}$ over an 8-dimensional decision vector $\mathbf{y} = [u_n, \mathbf{u}_K^T, \delta]^T$, co-optimizing normal approach rate $u_n$ and 6D operational-space stiffness rate $\mathbf{u}_K \in \mathbb{R}^6$.
* **Formal Safety (CBF):** Hard Control Barrier Function constraints ($\dot{h}_F \ge -\gamma_f h_F$ and $\dot{h}_P \ge -\gamma_p h_P$) strictly prevent contact force overshoot ($F_e \le F_{\max}$) and excessive surface penetration depth.
* **Exponential Convergence (CLF):** Control Lyapunov Functions drive force tracking error energy $V_F(\mathbf{e}_F) \to 0$ with exponential stability, enabling zero-phase-lag tracking of time-varying forces ($F_d(t) = F_{\text{mean}} + A \sin(\omega t)$).

### 2. Benchmark 1: ART-AVIC (Wang et al., 2027)
* **Architecture:** Adaptive Reference Trajectory & Segmented Adaptive Variable Impedance Control.
* **Mechanism:** Adjusts virtual reference position $x_r(t) = x_r(t-T) + k_x(f_d - f_e)$ via discrete integration, alongside a segmented damping update law $\Delta b = \phi(f_d - f_e)$.
* **Comparative Bottleneck:** Fixed stiffness ($K_p = \text{const}$) creates a trade-off between impact safety and dynamic tracking speed; pure integral updates induce phase lag and transient overshoot under time-varying targets.

---

## 💻 Environment & Setup

### Dependencies
* Python 3.8+
* `robosuite` (Panda arm environments)
* `mujoco` ($\ge 2.3.0$)
* `osqp` & `scipy` (Convex Quadratic Programming)
* `numpy`, `matplotlib`, `cma`

### Installation
```bash
pip install numpy scipy matplotlib robosuite mujoco osqp cma



### Step 3: Stage, Commit, and Push to GitHub

Run the following Git commands in your terminal to commit and push the new file:

```bash
git add README.md
git commit -m "Add comprehensive README.md with architecture and benchmark descriptions"
git push origin main