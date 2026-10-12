# HANDOFF：相对论 F12 项目（rf12）从云端 session 转到超算

新 session 请按顺序读：本文件 → `TASK.md`（用户原始任务，原文）→ 各 stage 的 summary / derivation notes。
分支：`ccr-7737f46e-ovnzz2`（xubwa/socutils）。所有工作已提交在此分支上。

## 0. 与用户协作的约定（必须遵守）
- 用户做相对论电子结构，不熟 F12，**通过数值结果沟通，不逐行读代码**；结论必须有可检查的数值支撑。用中文交流。
- 原则（见 `TASK.md`）：parent theory 优先；先 brute-force reference 再优化（逐位对上）；每阶段 anchor 对上才前进；
  推断与事实分开标注（笔记里用 [数值]/[推导]/[推断]），文献先核实再引用。
- **停下来问用户**：推导出现两种以上合理选择；anchor 两轮排查找不到原因；数值与文献矛盾。
- 每阶段交付：derivation notes、代码、anchor 写成 automated tests、一页 summary（对上了什么/没对上/下一步）。
- 用户在对话中做过的决定：
  - Stage 2 anchor：Psi4 1.11，3C(FIX)，非 DF（当时容器内没有 ORCA/Molpro）。**用户超算上有 ORCA 和 Molpro**，
    希望补做原始要求的 ORCA/Molpro 对照（输入草稿见 `examples/rf12/stage2_external/`，关键字需核实）。
  - Stage 3：两种 geminal（A：标量 f 作用于 4c pair；B：先相关再 KB）**都实现再比较**；
    projector = **完备化 RKB 空间中的 DHF Furry**；主相关因子 = **相对论 h(c·r12)**（先 NR STG，再加 h 修正）；
    anchor (b) **两种都做**（hydrogenic-H0 模式对 Stage 1；DHF 模式对 CBS）。
  - 旋量固定振幅、h 的表示、B 矩阵的处理：用户无偏好，按我的建议走（P^Σ 振幅；动能解析 + 其余 RI，full-RI 作对照）。
  - 用户的物理判断：外层相关电子基本是非相对论的，相对论效应主要是内层诱导外层形变；"高动量双电子贡献"不重要。
    但用户明确要求：**"Stage 3 还是应该该做的都做了，这样才能说明我们认为小的东西它是小的。"**
- git：提交信息不写模型名；不主动开 PR。

## 1. 超算上的环境搭建
1. **代码可导入**：仓库根目录就是 `socutils` 包。建一个包装目录：
   `mkdir -p ~/socutils-env && ln -s /path/to/socutils-checkout ~/socutils-env/socutils`，`export PYTHONPATH=~/socutils-env`。
2. **Python 依赖**（云端所用版本）：Python 3.13，pyscf 2.14.0（pip），numpy 2.5.3，scipy 1.18.1。
3. **启用 F12 的 libcint（必需）**：pip 版 pyscf 附带的 libcint 没有 F12 积分。
   ```bash
   git clone https://github.com/sunqm/libcint && cd libcint && git checkout v6.1.3
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_F12=1 -DWITH_RANGE_COULOMB=1 -DWITH_4C1E=1 \
            -DWITH_FORTRAN=0 -DWITH_CINT2_INTERFACE=0 -DKEEP_GOING=1
   make -j
   # 备份并替换 pyscf 的 libcint（路径用 python -c "import pyscf,os;print(os.path.dirname(pyscf.__file__))" 查）
   # pip 版 pyscf 2.14 的 deps/lib 里只有一个 libcint.so（云端如此）
   cp $PYSCF/lib/deps/lib/libcint.so $PYSCF/lib/deps/lib/libcint.so.orig_nof12
   cp -L libcint.so $PYSCF/lib/deps/lib/libcint.so
   ```
   替换后普通 ERI 应逐位不变（H2O/cc-pVTZ：‖ERI‖ = 71.97117841575987）。若共享 pyscf 不便改动，建议用自己的 venv。
4. **编译 socutils 的 C 库**：仓库根目录 `make`（或 `cd lib && ./build.sh`）。rf12 需要 `lib/librf12.so`（quad 精度本征值，需 libquadmath）
   和 `lib/librf12_f12.so`（F12 导数积分类 + 收缩驱动；CMake 通过 python 找 pyscf 的 libcint，**必须先完成第 3 步**）。
5. **基组**：`rf12/nr/gbs.py` 读 Psi4 的 .gbs 文件以保证与参考程序逐字节相同。查找顺序：`$RF12_BASDIR` → 可导入 psi4 的 `basis/` → 原容器路径。
   超算上没有 psi4 时，把 psi4 的 `share/psi4/basis` 目录拷过去并设 `RF12_BASDIR`。
6. **Psi4（可选，仅 Stage 2 复现用）**：云端是 conda-forge 的 psi4 1.11（独立 env）。
7. **Stage 1 缓存**：`export RF12_CACHE=/some/scratch/dir`（B-spline 积分缓存，可选）。
8. **资源**：Stage 3 brute-force 很吃内存。He aug-cc-pVDZ/aug-cc-pVTZ-OptRI 的 ansatz B 在 15 GB 内 OOM（exit 137）。
   建议在计算节点上跑，≥ 64 GB；并行用 OMP_NUM_THREADS。

**验证环境**（依次通过即可）：
```bash
python rf12/test/test_rf12.py              # stage 1 基础，~18 s
python rf12/test/test_ints4c.py            # F12 导数积分（FD、Yukawa→Coulomb、收缩模式）
python rf12/test/test_fourc.py             # 4c MO 积分 vs pyscf、DHF/MP2 c→∞、complete-space singles c→∞
python rf12/test/test_stage2_mp2f12.py     # stage 2 vs Psi4 数据（~4 min）
python rf12/test/test_stage1_anchors.py    # ~90 s
```

## 2. 各阶段状态
| Stage | 状态 | 文档 |
|---|---|---|
| 0 文献 | 完成 | `stage0_literature.md` |
| 1 He-like coalescence | 完成，anchor 全过 | `stage1_results.md`、`stage1_summary.md` |
| 2 NR MP2-F12/3C(FIX) | 完成，对 Psi4 到 ≤0.0002 µEh（同 geminal）；**ORCA/Molpro 对照待做** | `stage2_derivation.md`、`stage2_summary.md` |
| 3 4c MP2-F12 | 进行中 | `stage3_design.md`、`stage3_derivation.md` |
| 4 2c 推导 | 未开始 | — |
| 5 CCSD(T)-F12 | 未开始 | — |

### Stage 3 已完成并有测试的部分
- 积分层：`lib/rf12/f12ints.c` + `rf12/ints4c.py`（stg/yp × plain/ipvip1/ipvip1ipvip2/ip1/ip1ip2/ipip1/ipip1ipip2，四种收缩模式），
  FD 验证；`rf12/twoc_ints.py`（2-spinor 上带 L/∇²/σ·p 算符的 ERI，厄米性与模式一致到 1e-12–1e-16）。
- 4c 轨道与 MO 积分：`rf12/fourc.py`（对 PySCF 1e-15）；DHF/MP2 c→∞ 回到 NR（≤0.01 µEh）。
- 完备 RKB 空间与 no-pair 投影：`rf12/dc_space.py`（整体 Λ₊ 定义；分块定义被 anchor 否定，见 derivation §1）。

### Ansatz A（`rf12/dc_mp2f12.py`）：已实现，已表征，测试待写
c→∞ 不回到 Stage 2：4c 中没有 U 项，U 只通过负能态 RI 补回（推导 §3）。He aug-cc-pVDZ：A − Stage 2 = −2180 µEh（DZ-OptRI）、−1111 µEh（TZ-OptRI）。
QZ-OptRI 在云端被 60 分钟限制杀掉 → 超算上补跑（`examples/rf12/stage3/s3_07_ansatzA_cabs.py aug-cc-pvqz-optri`）。

### Ansatz B（`rf12/dc_mp2f12_kb.py`）：已实现；c→∞ 偏差已定位，**等用户决定**
详见 `stage3_derivation.md` §5。要点：c→∞ 时 B − Stage 2 = +23.27 µEh（He，DZ/DZ-OptRI）；
除 B 的投影项外全部中间量与 Stage 2 一致到 ≤0.07 µEh；偏差来自有限 RKB 基中 geminal 与负能态 N 的重叠 ∝ 1/c
（精确理论中是 O(1/c³)），乘 F_NN≈−2c² 后不随 c→∞ 消失。
- 待用户决定：(1) 保持一致投影、当作 RI 误差量化；(2) 含 N 的 pair 重叠按 1/c 展开置零（c→∞ 严格回到 Stage 2），(1) 作诊断。
  我的建议是 (2)，前提是下面的 TZ-OptRI 显示伪影随 CABS 减小。
- 诊断开关：`DCMP2F12KB.project_negative = False`；`m.ov_neg` = c·max|⟨γ|NJ⟩|；`m.parts` 存各项中间量。

## 3. 下一步（按顺序）
1. **[超算，第一件事]** `python examples/rf12/stage3/s3_11_ansatzB_negproj_cabs.py aug-cc-pvdz aug-cc-pvtz-optri`
   （≥ 64 GB）。DZ-OptRI 对照值：projN=True +23.27 µEh，projN=False 约 +3.0 µEh，c·maxOvN = 3.99e-4。
   - 若伪影随 CABS 明显减小 → 把结果和 (1)/(2) 的选择交给用户。
   - 若不减小 → 不是单纯基组问题，**停下来问用户**（两轮排查规则）。
2. ORCA/Molpro 的 Stage 2 对照（`examples/rf12/stage2_external/`，用户在超算上跑）；注意 DF 和 geminal 高斯拟合会造成 µEh 级差别。
3. Ansatz A：写自动测试（He DZ c→∞ 偏差 −2180.4 µEh 回归、MP2 一致）；补 QZ-OptRI。
4. Ansatz B：按用户决定定稿；anchor (a) 写成测试；full-RI 变体对照；RI 随 CABS 收敛；A 与 B 比较（含有限 c 的相对论位移，预期 He 为正、几 µEh）。
5. S3.6：h(c·r12) 修正（Slater 求和 geminal）；anchor (b) hydrogenic-H0 模式对 Stage 1、DHF 模式对 CBS；anchor (c) 小基组 F12 vs CBS。
6. Stage 3 一页 summary（`stage3_summary.md`），然后 Stage 4（含 X2C RI 尾检查——§5 的机制正是"RI 不完备被 F12 放大"的 4c 版本）。

## 4. 文件地图
- 代码：`rf12/`（stage 1：bspline/xp/pairpw/angular/relmodel；stage 2：`nr/`；stage 3：ints4c/twoc_ints/fourc/dc_space/dc_mp2f12/dc_mp2f12_kb）。
- C：`lib/rf12/`（qlinalg.c、f12ints.c、vendored `include/cint.h`、CMakeLists.txt）。
- 测试：`rf12/test/`；数据 `rf12/test/data/`（Psi4 参考值、Psi4 积分 npz）。
- 脚本：`examples/rf12/`（stage 1/2）、`examples/rf12/stage3/`（下表）。

| 脚本 | 用途 |
|---|---|
| s3_01_ints_fd.py | F12 导数积分有限差分检验 |
| s3_02_twoc_ints_check.py | twoc_ints.eri2 对全张量 brute force |
| s3_03_fourc_eri_check.py | 4c MO 积分 vs PySCF |
| s3_04_dhf_mp2_cinf.py | DHF/MP2 c→∞ |
| s3_05_singles_cinf.py | complete-space CABS singles c→∞ |
| s3_06_ansatzA_cscan.py | ansatz A 的 c 扫描（`He 0 10 20`） |
| s3_07_ansatzA_cabs.py | ansatz A c→∞ 偏差随 CABS |
| s3_08_ansatzB_cscan.py | ansatz B 的 c 扫描 |
| s3_09_ansatzB_negproj.py | B：投影/不投影 N 的对比 |
| s3_10_ansatzB_vs_stage2_terms.py | B 的逐项中间量对 Stage 2 |
| s3_11_ansatzB_negproj_cabs.py | B：负能态伪影随 CABS（超算任务 1） |
