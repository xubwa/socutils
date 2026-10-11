# Stage 3 设计笔记：四分量 Dirac–Coulomb no-pair MP2-F12（推导与待定选择）

已定（你的选择）：两种 geminal 都实现并比较；负能态投影 = 完备化 RKB 空间中的 DHF Furry；anchor (b) 两种都做；关联因子以 h(c·r12) 为主。

## 1. 空间与投影
- OBS：RKB 四分量基（大分量 2-spinor χ_μ，小分量 σ·p χ_μ/2c）。"1" = RKB(OBS ∪ CABS 基函数) 的正交完备化空间。
- 在 "1" 中把空间分成 O（占据，正能）⊕ V₊（OBS 正能虚轨道）⊕ V₋（OBS 负能态）⊕ C₊ ⊕ C₋，
  其中 C± 由 DHF Fock 在 CABS 补空间块内对角化的正/负能本征矢给出。
- 强正交投影（no-pair + ansatz 3）：Q12 = (V₊⊕C₊)⊗(V₊⊕C₊) − V₊⊗V₊ = V₊C₊ + C₊V₊ + C₊C₊。
  它是精确的投影算符，且把 OBS 与 CABS 中的负能分量都投影掉。
  （备选：在整个 "1" 中对角化 Fock 定义 Λ₊；那时 O、V₊ 不严格落在 Λ₊ 的值域里，Q12 不再是精确投影。差别随 CABS 完备而消失。我采用分块定义。）

## 2. 两种 geminal
- A（先 KB 再相关，Liu）：|w_kl⟩ = Q12 f12 Ŝ |kl⟩，f12 作用于四分量 pair 的全部 16 个分量。
- B（先相关再 KB，Ten-no–Yamaki/Pestka 思路）：|w_kl⟩ = Q12 K12 f12 Ŝ |k^L l^L⟩，
  K12 = (1, σ₂·p₂/2c, σ₁·p₁/2c, σ₁·p₁σ₂·p₂/4c²)。

**非相对论极限（推导，[推断，待数值检验]）**：对 A，单粒子 h_D = cα·p + V − 2c²(小分量块)，
⟨f ψ|h_D|f ψ⟩（ψ 小分量 = σ·pψ^L/2c）在 c→∞ 时 = ½⟨σ·pψ^L|f²|σ·pψ^L⟩ + ⟨(σ·p f)ψ^L|f σ·pψ^L⟩，
而非相对论 ⟨fψ|T|fψ⟩ 多出 ½⟨ψ^L|(∇f)²|ψ^L⟩。两电子合计正好缺 U = f′²（Stage 2 的 B1 项）：
标量 f 乘小分量破坏了 kinetic balance（正确的小分量是 σ·p(fψ^L)/2c）。
四分量里 [f,[cα·p, f]] = 0，approximation C 本身不再产生 U；只有 Q12 中的 Λ₊ 在完备 "1" 里能把它恢复（通过 RI）。
**预期：A 的 c→∞ 极限只有在 CABS 完备时才回到 Stage 2；有限 CABS 下 anchor (a) 会有可量化的偏差。**
B 在 c→∞ 时解析地回到 Stage 2 的几何子（含 f′²），前提是 B 矩阵中的动能部分按解析方式处理（见 §5 选择 3）。

## 3. 需要的积分（全部用 libcint，现有 gout + F12 初始化）
libcint 的 gout 函数与算符核无关；F12 积分只是把 CINTinit_int2e_EnvVars 换成 stg/yp 版本。
- A：(LL|K|LL)、(SS|K|LL)、(SS|K|SS)，K ∈ {stg, yp}：现成 gout int2e、int2e_ipvip1（∇μ·∇ν 与 iσ·(∇μ×∇ν)）、
  int2e_ipvip1ipvip2 → 需新增 stg/yp 包装（spinor 输出或在 sph 中组装）。
- B：σ·p σ·p = p²，积分不含自旋：需要 (∇²μ ν|K|..)（int2e_stg_ipip1 已有）与 (∇²μ ν|K|∇²κ λ)（ipip1ipip2，需新增包装）。
- 验证：yp(ζ→0) 必须等于 Coulomb 的同类积分；spinor 组装与 sph+自旋耦合两条路径对比；有限差分检查导数积分。

## 4. 参考与 anchor
- DHF 参考（pyscf DHF，Dirac–Coulomb）；类氢零阶模式（裸核 Dirac 作 H0，含单激发），其 E2 即 Stage 1 的量。
- (a) c→∞：在 c/c₀ = 10, 20 处计算，按 1/c² 外推（或用 qdhf 的 quad），与 Stage 2 比；
- (b) He 样离子：类氢模式 vs Stage 1 分波外推（µEh 级），DHF 模式 vs 大基组 DHF-MP2 CBS 外推；
- (c) 小基组 F12 vs 大基组 CBS 外推。

## 5. 需要你决定的三处选择
1. **旋量 pair 的固定振幅**。非相对论 SP 振幅（单态 1/2、三重态 1/4）依赖空间交换 p12，四分量中没有。
   (a) Ŝ = 3/8 + 1/8 P^Σ₁₂，P^Σ = (1 + Σ₁·Σ₂)/2 为自旋交换（Σ = diag(σ,σ)）：作用在反对称旋量 pair 上等价于 3/8 + 1/8 p12，
       c→∞ 时严格回到 Stage 2；需要插入 Σ 的积分（在 sph+自旋分解中容易）。
   (b) 放弃固定振幅，对每个 pair 优化振幅（不需要选 cusp 系数，但在大体系里可能不稳定，Stage 2 没有对应的 anchor）。
2. **h(c·r12) 的表示**（libcint 只能算 Slater 型 e^{−ζr} 与 Yukawa e^{−ζr}/r 及其线性组合）。
   Stage 1 的模型：h_rel − h_NR 在 r ≪ 1/c 处是 (1/4πc) ln r，在 r ≫ 1/c 处是 1/(4c² r)（长程 1/r 尾，O(α²)）。
   (a) f_rel = −e^{−γr}/γ + 2[h_rel − h_NR](c·r)·e^{−γr}：保留 STG 的长程衰减，相对论修正只在短程；用 Slater 函数和拟合（log 需要几何分布的 ζ）。
   (b) 直接把 2h_rel(c·r) 本身拟合成 Slater 和，长程不加阻尼（1/r 尾会进入 geminal，强正交与收敛性未知）。
   (c) 先只用 NR STG 做两种 ansatz 的比较，h 作为第二步加入（把"KB 方式"与"关联因子形式"两种效应分开）。
3. **ansatz B 的 B 矩阵**。⟨K12 f Φ|(F̂₁+F̂₂)|K12 f Φ⟩ 中 cα·p 与 K12 的组合解析处理时会出现 f²·V_nuc 之类的三体积分（libcint 没有）。
   (a) 动能/c 项解析（保证 c→∞ 回到 Stage 2 的 f′²），核吸引与 J/K 部分在 "1" 中做 RI（类似 approximation C 处理 K 的方式）；
   (b) 全部在 "1"⊗"1" 中做 RI（实现最简单，但 RI 误差大、c→∞ 只在 CABS 完备时回到 Stage 2）。
