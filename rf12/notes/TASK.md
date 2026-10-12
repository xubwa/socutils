# 原始任务说明（用户原文，未改动）

# 任务：建立有 parent theory 的相对论 explicitly correlated (F12) 方法

## 背景
你在 socutils 代码库中工作。现有：quadruple-precision HF/DHF/sf-X2C-1e
（socutils.qdhf）、relativistic high-order CC、relativistic NEVPT2/MRCI。
我做相对论电子结构，但对 F12 不熟。我主要通过数值结果和你沟通，
不逐行读代码，所以结论必须由可检查的数值支撑。

目标：先在 four-component no-pair Dirac–Coulomb 框架下建立 F12，
two-component 形式必须从它推导出来。不接受把 X2C 和 F12 直接拼接。

## 原则
1. Parent theory 优先。任何 two-component 近似都要说清楚它近似的是
   four-component 理论里的哪一项。
2. 先写慢而透明的 brute-force reference implementation（直接对公式），
   再做优化；优化版必须逐位对上 reference。
3. 每个阶段设 numerical anchor，anchor 没对上不进入下一阶段。
4. 推断和事实分开标注；所有文献引用先核实再使用。

## Stage 0：Literature
核实并精读：Kutzelnigg 1989（Kato cusp condition 的相对论推广）；
Li, Shao, Liu 2012 JCP（relativistic coalescence conditions）；
Simmen, Mátyus, Reiher 2015（kinetic balance for explicitly correlated
basis functions）；Ten-no 组的 relativistic F12 相关工作（核实具体文献）；
Hättig/Klopper/Köhn/Tew 2012 与 Kong/Bischoff/Valeev 2012 两篇 Chem. Rev.。
交付：每篇一段摘要，再列出它们之间的矛盾与空白。

## Stage 1：Coalescence behavior 的数值检验
He-like ions（Z = 2, 10, 30, 50, 80），Dirac–Coulomb no-pair，
spinor partial-wave expansion 到高 l，用 quadruple precision。
测定 partial-wave increment 的衰减指数和 r12→0 的局部行为，
与 Kutzelnigg 的预测对比。
Anchor：c→∞ 时必须回到 non-relativistic 的 (l+1/2)^-4（singlet）。
这一阶段的结论决定 correlation factor 该取什么形式。

## Stage 2：Non-relativistic baseline
MP2-F12（Slater-type geminal，fixed-amplitude ansatz，CABS），
体系 He、Ne、H2O。积分用 libcint。
Anchor：对 ORCA 或 Molpro 的数值对到 µEh。

## Stage 3：Four-component MP2-F12
设计满足 kinetic balance 的 explicitly correlated geminal；
strong orthogonality projector 要同时投影掉 negative-energy states；
在 RKB 下构造 CABS。
Anchor：(a) c→∞ 回到 Stage 2；(b) He-like ions 对高精度 Dirac–Coulomb
reference，或对 Stage 1 的 large-l extrapolation；(c) F12 在小基组下
应接近大基组 CBS extrapolation。

## Stage 4：推导 two-component 形式
从 Stage 3 出发，在 complete one-particle space 施加 exact decoupling，
逐一列出近似：one-electron decoupling、忽略 two-electron picture change、
complement space 中 decoupling transformation 的 finite-matrix
representation。每个近似都与 Stage 3 对比定量。
注意：已知 X2C-1e 在有限基组下的 resolution of identity 不完备
会产生 spurious long-range tail，而 F12 依赖 RI insertion，
要专门检查这个问题会不会被放大。

## Stage 5
前四步都站住之后，再讨论 CCSD(T)-F12。

## 每阶段交付
Derivation notes（关键步骤可检查）、代码、把 anchor 写成 automated tests，
以及一页 summary：对上了什么、哪里没对上、下一步打算做什么。

## 遇到以下情况先停下来问我
- 推导出现两种以上看起来都合理的选择（correlation factor 形式、
  projector 定义等）
- Anchor 对不上，且两轮排查后仍找不到原因
- 数值结果和文献结论矛盾
