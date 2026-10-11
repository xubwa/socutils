# Stage 0：文献核查（相对论 F12 的 parent theory）

标注：**[已核实:来源]** = 实际读过该文字（注明摘要/全文）；**[二手]** = 通过他人转述；
**[推断]** = 我们自己的推导，未见于文献。元数据（作者/卷/页/DOI）均经 Crossref 核对。
注意：几篇关键原文只拿到摘要（下面逐条注明），凡是依赖全文细节的说法都还需要你手头有全文时复核。

## 1. W. Kutzelnigg 1989
"Generalization of Kato's cusp conditions to the relativistic case", in D. Mukherjee (ed.),
*Aspects of Many-Body Effects in Molecules and Extended Systems*, Lecture Notes in Chemistry **50**,
Springer 1989, pp. 353–366, doi:10.1007/978-3-642-61330-2_19。**只读到摘要。**

- 相对论多电子波函数在点核及 r12→0 处有"弱奇点" r^ν, ν<0；有限核无奇点 [已核实:摘要]。
- Dirac 方程中 ν 对 Z/c 是二次的；square-root 算符和 **free-particle** no-pair 投影的 Dirac 算符中 ν 对 Z/c 是线性的（无光滑非相对论极限） [已核实:摘要]。
- 电子–电子：Dirac–Coulomb r12^λ, λ<0；Dirac–Gaunt λ>0；Dirac–Breit λ=0 [已核实:摘要]。
- 相对论效应只改变聚合点极小邻域，其余由 Kato cusp 支配 [已核实:摘要]。
- 显式指数 q = √(1−α²/4) − 1（Ψ ~ a r12^q + b r12^(q+1)）只见于 Pestka/Bylicki/Karwowski 2007 会议幻灯片对本文的转述 [二手]；幻灯片给的数值 −1.3×10⁻⁵ 与该公式（−6.7×10⁻⁶）不符，内部自相矛盾。**未能核实原文。**

## 2. Z. Li, S. Shao, W. Liu, J. Chem. Phys. **136**, 144117 (2012)
"Relativistic explicit correlation: Coalescence conditions and practical suggestions", doi:10.1063/1.3702631。
**读到完整摘要；全文未获取**，实践部分细节来自 W. Liu, Phys. Rep. **537**, 59 (2014)（全文已读）的转述。

- 用约化两电子 Hamiltonian 的内部对称性推导 DC、DCG、DCB、MDC、ZORA 的电子–电子聚合条件 [已核实:摘要]。
- DCG 在聚合点正则；DC 与 DCB 有弱奇点 r12^ν, ν<0 且 O(α²) [已核实:摘要]。
- 渐近展开收敛半径 R_c = O(α²)；α→0 与 r12→0 不对易，因此直接微扰理论（DPT）在 r12<R_c 对 Ψ^SS 失效 [已核实:摘要]。
- 结论：no-pair 近似下相对论显式相关可与非相对论"完全平行"（以 MP2-F12 为例），但须用 **extended no-pair projector** [已核实:摘要]。
- Liu 2014 转述：Q12 把 O₊ 与 V₋ 都视为占据；推荐直接用非相对论 STG f12 = −e^{−γ r12}/γ；两种 ansatz（"先 kinetic balance 再相关" f12|kl⟩ 与 "先相关再 KB" Q12 K12 f12|k^L l^L⟩），"没有先验理由偏好哪一种" [已核实:Liu 2014 全文]。

## 3. B. Simmen, E. Mátyus, M. Reiher, J. Phys. B **48**, 245004 (2015)
"Relativistic kinetic-balance condition for explicitly correlated basis functions", doi:10.1088/0953-4075/48/24/245004。
**摘要 + arXiv:1507.08556 全文（末尾约 15% 未读）。**

- 由线性方程组行约化导出显式相关基函数的 KB：对两费米子、一般几何子 Θ，
  |ll⟩=(m₁p₁²+m₂p₂²−4m₁m₂m₁₂c²)Θ，|ls⟩=(σ₂·p₂/2c)(p₂²−p₁²−4m₁m₁₂c²)Θ，
  |sl⟩ 对称，|ss⟩=−m₁₂(σ₁·p₁)(σ₂·p₂)Θ [已核实:arXiv 全文 式(50)–(53)]。
- 单粒子 KB 足以给出正确的非相对论极限，但一般不足以保证变分稳定 [已核实:arXiv 全文]。

## 4. Ten-no 组
- S. Ten-no, D. Yamaki, J. Chem. Phys. **137**, 131101 (2012), doi:10.1063/1.4757415，四分量显式相关 MP2：
  no-pair 下由 Lévy-Leblond 型方程得到 kinetically balanced 几何子，非相对论极限满足 cusp；
  用 large-component 基的正交补构造强正交投影 (2α/2β)；MP2 试验实现 [已核实:摘要；全文未获取]。
  Liu 2014 批评其 Q12 在有限 c 下混入负能态 [已核实:Liu 2014 全文]。
- 检索（Crossref、Semantic Scholar 引文表、Web）**未找到 Ten-no 组 2012 年后的相对论 F12 原创论文**；
  Grüneis/Hirata/Ohnishi/Ten-no, JCP **146**, 080901 (2017) 把"超越 DC 的相对论显式相关"列为未来挑战 [已核实:全文]。
- 其它相关：Bischoff/Valeev/Klopper/Janssen, JCP **132**, 214104 (2010)（标量相对论 DKH-R12）[已核实:摘要]；
  Pestka/Bylicki/Karwowski, J. Math. Chem. **50**, 510 (2012)（DC 几何子）[元数据已核实，摘要未读到]。

## 5. 两篇 Chem. Rev. (2012)
- Hättig, Klopper, Köhn, Tew, Chem. Rev. **112**, 4–74, doi:10.1021/cr200168z。**全文未获取，内容未核实**——本项目不引用其具体说法，直到核实为止。
- Kong, Bischoff, Valeev, Chem. Rev. **112**, 75–107, doi:10.1021/cr200204r。**全文已读**：
  Schwartz 公式 δE_l^(2) = −(45/256)(l+½)⁻⁴[1 − (5/4)(l+½)⁻² + …]（PDF 提取丢失负号，负号为推断）；
  Kutzelnigg–Morgan：自然宇称单态 (l+½)⁻⁴、三重态 ⁻⁶；Kato cusp 单态 (1+r12/2)Φ；
  一阶对函数单态 ½ r12 ψ⁽⁰⁾(r,r)；SP ansatz 源自 Ten-no；STG、RI/CABS、approximations A/B/C。
  相对论只有一句（R12 技术可用于相对论两电子算符），无四分量内容 [已核实:全文]。

## 6. 分波收敛的直接前驱
- W. Kutzelnigg, Int. J. Quantum Chem. **108**, 2280 (2008), doi:10.1002/qua.21747 [已核实:摘要]：
  ¹S 两电子离子主相对论修正的分波增量 ∝ (l+½)⁻²（非相对论 ⁻⁴）；系数
  DC: 2α²π²∫|ψ(r,r,0)|² r³ dr，完整 DCB 相同；Gaunt 单独 −4α²π²∫…；
  只依赖非相对论 cusp，"不需要 no-pair 投影或短程正则化的假设"；不给出超出主阶的信息。
- S. Salomonson, P. Öster, Phys. Rev. A **40**, 5548 (1989), doi:10.1103/PhysRevA.40.5548 [已核实:摘要]：
  He、DC no-pair 全阶对函数，相对论修正分波主项 ≈ 0.075 α²(l+½)⁻²。
- Kutzelnigg, Morgan, JCP **96**, 4484 (1992) [已核实:摘要]：E^(2) 只含偶次幂；(l+½)⁻⁶/(l+½)⁻⁴ 系数比在 1/Z 展开中一般为 −5/4。
- Schwartz, Phys. Rev. **126**, 1015 (1962) [元数据已核实；45/256 公式经 Kong 综述二手核实]。

## 矛盾与空白
1. **DCB 指数冲突**：Kutzelnigg 1989 说 Breit λ=0；Li 2012 说 DCB 有 ν<0。DCG 二者大致一致（正则/λ>0）。
2. **DC 指数的显式值无一手来源**；唯一公式（二手）数值自相矛盾。各分量（LL/LS/SS）的指数需要 Li 2012 全文。
3. **"奇点对能量无关紧要"（Liu）与"相对论分波尾 (l+½)⁻²"（Kutzelnigg 2008, Salomonson–Öster）并存**：后者来自相对论算符作用在非相对论 cusp 上，比非相对论尾慢两阶。非相对论 STG 能否吸收 α²(l+½)⁻² 尾，**无任何发表的数值检验**。
4. **DPT 失效的位置**：Li 2012 说 DPT 在 r12<R_c 失效；Kutzelnigg 2008 的 (l+½)⁻² 正是 DPT 主阶。(l+½)⁻² 律在多大 l 失效、如何依赖 Z，**无数据**。
5. **no-pair 投影的种类**（free-particle / 裸核 Furry / DHF Furry / extended）对聚合行为的影响无统一说法；free-particle 投影据 Kutzelnigg 1989 有 O(Z/c) 线性效应。
6. **几何子 KB 有三种不等价版本**（Ten-no–Yamaki、Liu 的两种顺序、Simmen 2015），从未数值比较。
7. 核实缺口：Kutzelnigg 1989、Li 2012、Ten-no–Yamaki 2012、Hättig 2012 全文均未读到。

## 供 Stage 1 检验的定量预测 [推断，已手算复核]
在 1/Z 展开（标度单位，核电荷 1、光速 c/Z）下，把零阶 ψ₀ = e^{−r₁−r₂}/π 代入：
- 非相对论 Hill 系数 6π²∫|ψ₀(r,r)|² r⁵ dr = 6·120/4⁶ = **45/256** —— 精确复现 Schwartz 系数（自洽性检验）。
- 同样代入 Kutzelnigg 2008：A₂ = 2α²π²∫|ψ₀|² r³ dr = 2·6/4⁴·(Zα)² = **(3/64)(Zα)² ≈ 0.0469 (Zα)²**，
  即 δE_l ≈ −(45/256)(l+½)⁻⁴ + A₂(l+½)⁻²（A₂ 的符号按摘要的写法为正，摘要未明示符号约定）。
- (l+½)⁻² 项与 ⁻⁴ 项相等处 (l+½)* ≈ 1.94/(Zα)：Z=2 → 133，10 → 27，30 → 8.9，50 → 5.3，80 → 3.3。
