# 从 0 开始：Diffusion Model 与 Flow Matching（含细胞扰动预测入门）

这个仓库是一个**循序渐进**的教程，目标是帮助你：

1. 先掌握 diffusion model（扩散模型）的最小可运行代码；
2. 再掌握 flow matching（流匹配）的最小可运行代码；
3. 最后把两者迁移到一个“玩具版细胞扰动预测”任务中，理解怎么落地到真实单细胞场景。

> 说明：为了学习友好，这里用的是低维 toy data + 简化版模型，不追求 SOTA 指标。

---

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 教程结构

### 1) Diffusion 基础（二维分布生成）

```bash
python src/diffusion_toy.py --steps 600 --batch-size 256 --sample-size 1500 --plot
```

你会看到：
- 训练 loss 下降；
- 采样点逐步逼近训练分布（两个高斯团）。

核心知识点：
- 前向加噪：`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`
- 训练目标：预测噪声 `eps`
- 反向采样：从 `x_T ~ N(0, I)` 开始逐步去噪

---

### 2) Flow Matching 基础（二维分布生成）

```bash
python src/flow_matching_toy.py --steps 600 --batch-size 256 --sample-size 1500 --plot
```

你会看到：
- 模型学习一个速度场 `v_theta(x_t, t)`；
- 通过常微分方程积分从噪声流到数据分布。

核心知识点：
- 路径定义（线性插值）：`x_t = (1-t)x_0 + t x_1`
- 目标速度：`u_t = x_1 - x_0`
- 训练目标：让 `v_theta(x_t,t)` 拟合 `u_t`

---

### 3) 细胞扰动预测（玩具版）

```bash
python src/cell_perturbation_demo.py --method diffusion --steps 800
python src/cell_perturbation_demo.py --method flow --steps 800
```

任务设定（简化版）：
- 给定“细胞基因表达”向量 `x_ctrl`（未扰动）和扰动类型 `p`；
- 预测扰动后表达 `x_pert`。

实现方式：
- `diffusion`：训练条件噪声预测网络（条件= `x_ctrl` + `p`），再采样得到 `x_pert`；
- `flow`：训练条件速度场网络，积分得到 `x_pert`。

输出包括：
- 训练 loss；
- 每个方法在验证集上的 MSE。

---

## 学习路径建议

1. 先读 `src/diffusion_toy.py`，理解“加噪-去噪”；
2. 再读 `src/flow_matching_toy.py`，对比“概率流 ODE”的思路；
3. 最后读 `src/cell_perturbation_demo.py`，看条件生成怎么接入细胞扰动任务。

如果你后续要做真实单细胞数据（如 scRNA-seq）：
- 输入可从 PCA/latent 表征开始（避免直接高维稀疏计数造成训练不稳）；
- 条件变量可加入 cell type、batch、dose、time；
- 评估除 MSE 外建议加：DE genes 命中、通路富集一致性、OOD 扰动泛化能力。

---

## 你应该如何在 Diffusion vs Flow Matching 中做选择（先验建议）

- 如果你更想快速复用生态（调参经验、现成设计）：先试 diffusion。
- 如果你更关注采样效率、连续轨迹解释、ODE 风格建模：优先试 flow matching。
- 在细胞扰动预测里，建议做小规模对比实验：
  - 指标：MSE / Pearson / DE 方向一致性；
  - 代价：训练时间、推理时间、稳定性。

这个仓库就是为了让你能低成本跑出第一版对比。
