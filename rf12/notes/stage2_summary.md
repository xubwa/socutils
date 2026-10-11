# Stage 2 一页总结：非相对论 MP2-F12/3C(FIX) baseline

**实现**：`rf12/nr/mp2f12.py`（brute-force reference：ansatz 3、SP 固定振幅、approximation C、CABS、CABS singles；
积分全部来自 libcint：int2e / int2e_stg / int2e_yp，**精确 Slater geminal**），`rf12/nr/mp2f12_brute.py`（按算符定义直接施加 Q12 的交叉检验），
推导 `notes/stage2_derivation.md`，测试 `rf12/test/test_stage2_mp2f12.py`（5 项，约 4 分钟）。

**Anchor（按你的选择：Psi4 1.11，conventional，β = γ = 1，aug-cc-pVDZ / aug-cc-pVDZ-OptRI）**
ORCA/Molpro 本容器不可用；Psi4 从 conda-forge 安装。He 没有 cc-pVDZ-F12-OptRI，因此三个体系统一用 aug-cc-pVDZ。

| 体系 | HF / MP2 / CABS singles vs Psi4 | 用 Psi4 自己的积分（6G geminal）组装 E_F12 vs Psi4 | 精确 STG：libcint vs libint Laplace 展开 | 精确 STG − Psi4 默认（6G 拟合） |
|---|---|---|---|---|
| He | < 0.001 µEh | 0.0000 µEh | 0.000004 µEh | −0.71 µEh |
| Ne（全电子） | < 0.001 µEh | 0.0002 µEh | 0.0001 µEh | −241.0 µEh |
| Ne（冻结芯） | < 0.001 µEh | 0.0000 µEh | （计算中） | +16.8 µEh |
| H2O（全电子） | < 0.001 µEh | 0.0000 µEh | （计算中） | −51.9 µEh |
| H2O（冻结芯） | < 0.001 µEh | 0.0000 µEh | （计算中） | +60.3 µEh |

**对上了**：
- 公式层面与 Psi4 一致到 ≤ 0.0002 µEh（同一 geminal、同一 CABS）。
- 精确 Slater geminal 的积分在两个独立积分库（libcint、libint）之间一致到 ≤ 0.0001 µEh。
- B 的逐项展开（B2–B8）与直接施加投影算符的定义一致到 10⁻¹⁴。

**没对上 / 需要知道的**：
- 我们的生产结果（精确 STG）与 Psi4 默认输出差 1–241 µEh：这完全来自 Psi4 的 6-Gaussian 拟合（芯电子越多越大）。
  如果之后要对 ORCA/Molpro，需要确认它们用的 geminal（Molpro 也用 Gaussian 拟合）和 DF 设置。
- Psi4 的 geminal 没有 1/β 因子而振幅固定为 3/8、1/8，所以只在 β = 1 时满足 cusp；我们用 −e^{−γr}/γ，任意 γ 都正确。
- 3C(FIX)、γ = 1 时 Ne 的冻结芯 F12 修正（−0.09522）比全电子（−0.09123）更负，两个独立程序一致；
  这是固定振幅 + γ = 1 对芯 pair 不合适造成的方法性质，不是程序错误。
- pip 版 pyscf 的 libcint 没有 F12 积分；已用同版本 libcint v6.1.3 加 WITH_F12 重新编译替换（普通 ERI 逐位不变），代码会检测并提示。
- 目前只有 reference 实现（H2O/aug-cc-pVDZ 约 110 s）；还没有写优化版本。

**下一步**：Stage 3（四分量 MP2-F12）。有几个推导上的选择需要你决定（见对话）。
