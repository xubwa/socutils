# Stage 3 推导笔记：四分量 Dirac–Coulomb no-pair MP2-F12

（与 `stage3_design.md` 配合；本文件记录已确定的定义和已有的数值检验。标注 [数值] / [推导] / [推断]。）

## 1. 空间与投影（`rf12/dc_space.py`）
- "1"：OBS+CABS 基函数的 RKB 空间（大分量 2-spinor χ，小分量 σ·pχ/2c），正交归一。
- Λ₊("1") = 整个 "1" 中 DHF Fock 矩阵正能本征矢张成的空间；
  C₊ = Λ₊ ⊖ (O ⊕ V₊)（SVD 求零空间后在 C₊ 内正则化），N = "1" ⊖ (O ⊕ V₊ ⊕ C₊)。
- Q12 = (V₊⊕C₊)⊗(V₊⊕C₊) − V₊⊗V₊ = (1−O′)⊗(1−O′) − V₊⊗V₊，O′ = O ⊕ N（"填满的 Dirac 海"）。
  在 "1" 的这一组基中 Q12 是 pair 指标上的 0/1 掩码。
- **[数值] 被 anchor 否定的定义**：在"与全部 OBS 轨道正交的补空间"内对角化 Fock 来定义 C±（设计笔记里我原来倾向的分块定义）。
  补空间被迫与 OBS 负能态正交，混入 O(1/c) 的负能成分，代价 O(1/c)²·2c² = O(1)，c→∞ 时也不消失（Brown–Ravenhall 型污染）：
  He 的 c→∞ CABS singles 为 −1.080 mEh，正确值（Stage 2）−0.468 mEh。改用整体 Λ₊ 后：He 一致到 < 10⁻⁹ Eh，Ne 0.02 µEh。

## 2. Pair 与固定振幅
- 旋量 pair i<j，Φ_ij = (|ij⟩ − |ji⟩)/√2。E2 = Σ_{i<j} E_ij。
- S12 = 3/8 + 1/8 p12（p12：空间交换）。在反对称 pair 上 p12 = −P^Σ，P^Σ = (1 + Σ₁·Σ₂)/2，故 S12 = 3/8 − 1/8 P^Σ：
  三重态（P^Σ = 1）1/4，单态（P^Σ = −1）1/2。展开：
  S12 Φ_ij = (1/√2)(1 − P12)[5/16 |ij⟩ − 1/16 Σ_c |Σ_c i, Σ_c j⟩]。
- [推导] Σ_c 作用于 RKB 函数会离开 RKB 空间：σ_c σ·p = 2p_c − σ·p σ_c。ansatz A 中用投影到 "1" 的 Σ̃_c（只改变 O(1/c²)）；
  ansatz B 中 Σ_c 作用在 2-spinor 大分量上，是精确的。

## 3. Ansatz A（标量 geminal，`rf12/dc_mp2f12.py`）
u_ij = Q12 f12 S12 Φ_ij，f12 = −e^{−γr12}/γ 作用于全部 16 个分量。
- V = ⟨γ|f Q12 g|Φ⟩，X = ⟨γ|f Q12 f|γ⟩，C_ab = ⟨γ|f Q12 (F̂₁+F̂₂)|Φ_ab⟩，
  B = ⟨f F̂₁₂ f⟩_C − ⟨f P F̂ f⟩ − ⟨f F̂ P f⟩ + ⟨f P F̂ P f⟩（P = 1 − Q12）。
- **[推导] 四分量里 approximation C 没有 U 项**：h_D = cα·p + V 中 cα·p 是一阶微分算符，[f,[cα·p,f]] = 0，
  所以 f F̂ f = ½{f², F̂+K̂} − f K̂ f 精确成立（K̂ 用 RI）。
- **[推导] 非相对论极限**：标量 f 乘小分量破坏 kinetic balance（正确的小分量是 σ·p(fψ^L)/2c）。
  单电子 c→∞：⟨fψ|h_D|fψ⟩ − ⟨fψ|T|fψ⟩_NR = −½⟨ψ^L|(∇f)²|ψ^L⟩；两电子合计正好缺 U = f′²。
  只有 Q12 中的负能态（O′ ⊃ N）在完备 "1" 中通过 RI 把它补回（⟨f P F̂ P f⟩ 中 F_NN ≈ −2c² 乘 |⟨N|fψ⟩|² = O(1/c²)）。
- **[数值] He，aug-cc-pVDZ，c→∞（c = 10c₀、20c₀ 外推）**：
  MP2 部分与 Stage 2 一致（< 0.001 µEh）；Stage 2 中 U 项贡献 +32.79 mEh；

  | CABS | Stage 2 E_F12 | ansatz A E_F12(c→∞) | A − Stage 2 |
  |---|---|---|---|
  | aug-cc-pVDZ-OptRI | −9.8967 mEh | −12.0771 mEh | −2180 µEh |
  | aug-cc-pVTZ-OptRI | −9.9630 mEh | −11.0742 mEh | −1111 µEh |

  负能态 RI 补回了 U 的约 93%（DZ）/ 97%（TZ）；剩余随 CABS 增大而减小，与推导一致。
  结论：**A 的 c→∞ 极限只在 CABS 完备时回到 Stage 2，且误差落在 B 的最大一项上（mEh 量级），远大于我们要量化的那些"小"相对论效应。**

## 4. Ansatz B（先相关再 KB）的定义："approximation C-KB"
γ_p = K12 f12 S12 Φ^L_p，Φ^L 由 DHF 旋量的大分量（2-spinor，不重新归一）组成，K = (1, σ·p/2c)ᵀ，K12 = K₁K₂。
记 A = S12 Φ^L_p（2-spinor pair 函数），γ = K12 f A。

**设计原则**：c→∞ 时存活的项与 Stage 2 的计算方式完全相同（解析 FG、F²、U；单轨道 RI 处理 f²(T+V+J)；pair RI 处理 fKf），
只有 O(1/c²) 的余项用 RI。这样 c→∞ 时 B 与 Stage 2 逐项相同（anchor (a) 是恒等式而不是收敛性质）。

关键恒等式 [推导]：对任意 2-spinor 函数 X、Y，⟨K₁X|T_D|K₁Y⟩ = ½⟨X|p²|Y⟩ **精确成立**（T_D = cα·p + (β−1)c²），
⟨K₁X|W|K₁Y⟩ = ⟨X|W|Y⟩ + (1/4c²)⟨σ·pX|W|σ·pY⟩（W 为局域势 V 或 J）。
与 "1" 中 RKB 函数 I = (I^L, σ·pξ_I) 的重叠：⟨IJ|K12 f A⟩ = ⟨Ĩ J̃|f|A⟩，Ĩ = I^L + (1/2c) p² ξ_I（p² = −∇²）。

- **⟨γ|IJ⟩**（投影项、C、RI 用）：上式，积分类 plain + ipip1 迹。
- **V** = ⟨γ|g|Φ⟩ − ⟨γ|P|IJ⟩⟨IJ|g|Φ⟩，⟨γ|g|Φ⟩ = ⟨fA|g|Φ^{LL}⟩（解析，yp 核）+ O(1/c²) 交叉项（在 2-spinor RI 空间 R⊗R 中计算）。
- **X** = ⟨γ|γ⟩ − ⟨γ|P|IJ⟩⟨IJ|γ⟩，⟨γ|γ⟩ = ⟨A|f²|A⟩ + (1/4c²)⟨fA|p₁²+p₂²|fA⟩ + (1/16c⁴)⟨fA|p₁²p₂²|fA⟩；
  第二项用 ⟨fA|p₁²|fA⟩ = ⟨A|{f², T₁}|A⟩ + ⟨A|f′²|A⟩（单轨道 RI 处理 T₁A），第三项 R⊗R RI。
- **B_C** = U + ½⟨A|{f², (T+V+J)₁ + (T+V+J)₂}|A⟩（单轨道 RI，R 空间）− ⟨γ|K̂₁+K̂₂|γ⟩（"1" 中 pair RI）
  + (1/4c²)⟨fA|p₁²p₂²|fA⟩（精确动能的 1/c² 余项，R⊗R RI）
  + (1/4c²) Σ_e [⟨σ_e·p_e fA|W_e|σ_e·p_e fA⟩ + ⟨σ_{e′}·p_{e′} fA|W_e|σ_{e′}·p_{e′} fA⟩] + O(1/c⁴)（R⊗R RI）；
  B = B_C − ⟨γ|P F̂ |γ⟩ − h.c. + ⟨γ|P F̂ P|γ⟩（"1" 中）。
- **C_ab** = ⟨γ|Q12 F̂₁₂|Φ_ab⟩（"1" 中）。
这里 R = OBS+CABS 大分量 2-spinor 基的正交化空间（c→∞ 时与 Stage 2 的 "1" 同一张成）。

## 5. Ansatz B 的 c→∞ 检验：负能态投影的有限基伪影
**[数值] He，aug-cc-pVDZ / aug-cc-pVDZ-OptRI，frozen = 0**（`examples/rf12/stage3/s3_08`–`s3_11`）：

| c | E_F12(B) − Stage 2 |
|---|---|
| c₀ | +26.81 µEh |
| 10c₀ | +23.30 µEh |
| 20c₀ | +23.27 µEh |
| c→∞（Richardson） | +23.265 µEh |

按设计应为恒等式（0）。20c₀ 下逐项对 Stage 2（`s3_10`，B 的 pair 归一化下 V ↔ ½V_S2，X、B ↔ ¼X_S2、¼B_S2）：
VFG −0.020，X_f2 +0.002，U +0.002，½{f²,T+V+J}（RI）+0.030，fKf +0.042，V 投影 −0.073，X 投影 +0.014 µEh；
V、X 总量 ≤ 0.003 µEh；**只有 B 的投影项差 −23.29 µEh**（B 总量 +23.28 µEh）。

诊断：
- 把 N 当作不投影（`project_negative=False`，N 归入 Q 一侧）：10c₀ 下 +23.30 → +3.00 µEh；剩下的来自 (N, O) 型 pair（O 仍被投影，F_NN 作用其上）。
- c·max|⟨γ|N J⟩| = 3.991×10⁻⁴，10c₀ 与 20c₀ 相同 → ⟨γ|NJ⟩ ∝ 1/c；乘 F_NN ≈ −2c²，贡献 c²·|O(1/c)|² = O(1)，c→∞ 不消失。
  符号：−⟨γ|PF̂|γ⟩ − h.c. + ⟨γ|PF̂P|γ⟩ 中 N 部分 = −Σ_N |⟨γ|N·⟩|² F_NN > 0，与 +23 µEh 一致。

**[推导] 精确理论中 ⟨γ|NJ⟩ = O(1/c³)**：负能态 N = (u^L, u^S)，u^L = −σ·p u^S/2c + O(1/c³)；
⟨N|Kφ⟩ = ⟨u^L|φ⟩ + ⟨u^S|σ·pφ⟩/2c，O(1/c) 项因 σ·p 厄米而抵消。贡献于能量 2c²·O(1/c⁶) = O(1/c⁴)，低于我们关心的 O(1/c²)。
**[推断] 有限 RKB 基中抵消不完全**：u^L 只能是大分量基中的最佳表示，(1 − P_L) p²ξ ≠ 0；geminal fA 不在基组张成中，
所以 ⟨Ĩ J̃|f|A⟩（Ĩ = I^L + p²ξ_I/2c）对 N 留下 ε/c，ε 为基组不完备度。预期随 CABS 增大而减小——**尚未验证**：
aug-cc-pVTZ-OptRI 的计算在 15 GB 容器内 OOM（exit 137），需要在超算上跑（`s3_11 aug-cc-pvdz aug-cc-pvtz-optri`）。
（cusp：高阶项含 p³f，判断可积，属推断。）

**这是一个需要用户决定的推导选择**（尚未决定）：
1. 保持 "1" 中一致的 no-pair 投影，把该项作为 RI 误差量化（随 CABS 的表）；
2. 对含 N 的 pair 重叠用 1/c 展开的精确值（O(1/c³) → 置零），c→∞ 严格回到 Stage 2，(1) 保留为诊断。
我的建议是 (2)，前提是 TZ-OptRI 显示伪影随 CABS 减小（若不减小，说明不只是基组问题，须停下讨论）。
**对 Stage 4 的意义**：负能态 2c² 能隙把有限基 RI 的微小泄漏放大为 O(1)——正是"RI 不完备被 F12 放大"的机制，在 4c 就已出现。
