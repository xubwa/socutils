# CFOUR X2C 性质积分 → socutils 接口

调研结论与实施方案。所有标注「实测」的数字都是在 `~/scr4/zmcscf/sm` 的真实 Sm 算例上跑出来的。

上游格式备忘：`~/apps/cfour-x2cso/doc/x2cprop_integral_files.md`（本文档在 §2.5 列出它需要修正的地方）。

---

## 0. 新 session 上手须知

**这个 repo 的位置**。`~/apps/socutils-git` 是活的 repo（repo 根目录**就是** `socutils` 包）。
`~/apps/socutils` 是 2024-03 的陈旧副本，`.git` 已损坏（只剩 `objects/`），它嵌套的
`socutils/` 会在 cwd 是它时**静默遮蔽 `import socutils`**。`CLAUDE.md` 里写着这个副本
"has been removed"，实际还在。跑任何 import socutils 的东西一律用：

```bash
PYTHONPATH="/home/xwang405/apps/socutils-git-env:/home/xwang405/apps/pyscf:$PYTHONPATH" \
  /home/xwang405/apps/conda/bin/python ...
```

（`~/apps/pyscf` 才是可信的 pyscf，conda site-packages 里那个不是；不要临时建符号链接，
`socutils-git-env/socutils -> socutils-git` 是常设的。另外 `python -c` 会把 cwd 放在
`sys.path` 最前，所以**不要**从含有 `socutils` 目录的地方跑。）

**验证数据**：`~/scr4/zmcscf/sm`。单个 Sm 原子，C1，未收缩 Dyall v3z，`nc = 308`，`natms = 1`。
里面既有 CFOUR 写出的性质积分文件，也有 socutils 的驱动脚本（`Sm_hf.py` / `Sm_cahf.py`）和
SHCI 的 `FCIDUMP` / `input.dat` / `states_splitting.csv`。**不要碰这个目录**（`cderi.chk` 是 2.8 GB）。

**注意**：`~/scr4/zmcscf/sm/output.dat` 是 Dice/ZFCI 的输出，**不是 CFOUR 的输出**。里面没有
任何 CFOUR 打印的性质数值。所以没有现成的参考数字可对，验证只能靠「CFOUR 文件 vs socutils
自己算的同一个算符」这种内部一致性检验（§6）。

---

## 1. TL;DR

1. **reader 的大半已经存在**。`tools/label.py` 里的 `carr_f2py` 就是这个 Fortran 格式的读取器，
   `Labels.spinor_cfour2pyscf(mol)` 就是 CFOUR→pyscf 的 AO 重排 + 旋量变换。
   `somf/x2cmp.py:864 read_qed()` 是一个现成的 6 行消费模板。剩下要做的是**分类学 + 语义**，
   不是解析。

2. **AO 顺序问题已经彻底解决并验证到 1e-15**（§3）。CFOUR 的 `SM:DYVTZUNC` 与 pyscf 的
   `basis='dyallv3z'` 逐位一致，置换矩阵精确，无需任何符号翻转。

3. **CFOUR 写出来的是 `x_response=True` 的图像变换算符**（实测：`SOX2CDPZ` 对
   `get_hfw1(..., x_response=True)` 吻合到 1.1e-9，对普通图像变换只有 3.8e-5）。
   这意味着它们是**能量导数性质**的算符，**不能**直接用于跃迁矩——而 `prop/int1e.py` 和
   `tdscf/gks.py` 恰恰刻意对跃迁矩用 `x_response=False`。**且这个约定在文件之间不统一**
   （§4.1），接口必须逐文件带元数据，不能全局设一个开关。

4. **socutils 的性质计算确实坏了，而且比表面看到的更严重**（§5）。有两条互不相干的车道：
   新的 `prop/int1e.py`（未提交、只注册了一个算符）和旧的 `prop/{efg,hfc,gfac,...}`
   （与 2024-03 快照逐字节相同、多处直接抛异常）。

5. **缺的关键接口是「算符已经是 2c 了，直接缩并」这个入口**——今天不存在。这正是 CFOUR
   文件需要的入口，也正是相关联的 CAHF/SHCI 密度需要的入口。

---

## 2. 文件格式：实测确认的规范

### 2.1 容器

Fortran unformatted sequential，**整个数组一条 record**（前后各 4 字节长度标记）。
`scipy.io.FortranFile` 可直接读。

**但有例外，文档没提**：以下 9 个文件是**裸 dump，没有 record 标记**，
`FortranFile` 会抛 `FortranFormattingError`：

```
X2CATMXM  X2CAX1EM  X2CD4CLL  X2CD4CLS  X2CD4CSS
X2CDEN2C  X2CFOCKM  X2CMFSOM  X2COVLPM
```

判据很简单：读头 4 字节整数，若 == `filesize - 8` 则是 record，否则是裸 dump。
（表现为 `SO*` 文件是 6071304 字节而 `X2C*` 是 6071296 —— 差的正是那 8 字节。）

### 2.2 数据布局

- `complex*16`，**实虚交错**（numpy `.view(complex128)`）。
  实测：交错解释给出厄米性 1e-16，「前半实后半虚」解释给出 ~1.0 的相对误差 —— 交错是确定的。
- **Fortran 列主序**，reshape 成 `(2nc, 2nc)`。代码里就是 `.reshape(2*nc, 2*nc).T`。
- 旋量基排列：**前 nc 个 α ⊕ 后 nc 个 β**（确认）。
- `nc` 可由文件大小唯一定出：`6071296 = 64·nc²` → `nc = 308`，与 `NTOTAL` 一致。

> **陷阱**：C-order/F-order 搞反时，厄米矩阵会静默地变成它的复共轭 —— **能通过任何厄米性自检**。
> 所以必须显式 `order='F'`，并且用已知数值验证，而不是用厄米性验证。

### 2.3 分类（文档的 A–F 分类经实测确认）

| 类 | 布局 | 偏移（以 double 计） |
|---|---|---|
| A 单矩阵 | `8·nc²` | — |
| B 3 分量、分文件（x/y/z 在文件名里） | 每文件 `8·nc²` | — |
| C 有原子项、无分量 | `8·nc²·natms` | `(j-1)·8nc²` |
| D 原子 × 3 分量 | `8·nc²·3·natms` | `((j-1)·3 + i-1)·8nc²` |
| E 原子 × 6 分量（`SOEFG`） | `8·nc²·6·natms` | `((j-1)·6 + i-1)·8nc²` |
| F 实数文件 | `EFGNUC`: `6·natms`；`OVRLPGRD`: `nc(nc+1)/2·3·natms` | — |

原子在外、分量在内，实测确认。`SOEFG` 的 6 分量顺序 `(zz, yy, xx, zx, zy, yx)` 确认。
`SOM01`/`UTM01`/`SOPSNEC`/`SOSFMO2C` 的 3 分量顺序 `(x, y, z)` 确认。

算符文件本身**不带原子标签**，原子索引纯粹是字节偏移 —— 原子的身份要由调用方从
`mol` 那边提供，接口必须保证两边的原子顺序一致。

### 2.4 本算例里恒为零的文件

`SONUCGRD`、`EFGNUC`、`OVRLPGRD` 在这个单原子（位于原点）算例里**恒等于零**，
拿它们验证不了任何东西。多原子的情形本算例覆盖不到，这是验证数据的固有缺口（§6.4）。

### 2.5 上游文档需要修正的地方

1. **「矩阵为 Hermitian」不成立于全部文件**。`SOLZ` 是实反对称矩阵（存的是 `i·L_z`），
   **反厄米**，期望值取迹的**虚部**。
2. **9 个 `X2C*` 文件不是 `wfile` 格式**（§2.1），文档未提。
3. **`SOEFG` 的物理注解不对**。文档说 i=1 是无迹组合而 i=2,3（yy、xx）各含一项 `4π/3` 接触项；
   实测**三个对角分量之和精确为零**（相对 1.1e-14 ~ 1.5e-14），即三个槽**都**是无迹组合。
4. 文档没有说明 `x_response` 约定逐文件不同（§4.1）—— 这是最容易出错的一点。
5. **两份文档都没列的三个文件**（2026-08-31 全目录扫描）：`ATMFOCKM`、`ATMOVLPM`、`SOX2CHFW`，
   均为 record 格式、各一个 2c 矩阵。
6. `SONEFG4C` 的自我覆盖问题**由文件大小独立证实**：它只有 **3** 个 4c 矩阵，而写盘端按 6 分量
   索引。接口里已拉黑（`prop/cfour.py:BLACKLIST`）。

---

## 3. CFOUR ↔ pyscf 转换（已验证到 1e-15）

### 3.1 AO 顺序

CFOUR 的收缩球谐 AO 排列：**原子 → l 从小到大 → m → 收缩函数（指数从大到小）**，
每个 (l,m) 组连续。m 的顺序是：

```
+l, -l, +(l-1), -(l-1), ..., +1, -1, 0     （|m| 降序，正在负前，m=0 放最后）
```

Sm/DYVTZUNC 计数核对：`30 s + 3×24 p + 5×16 d + 7×11 f + 9×3 g + 11×2 h = 308` ✓

**这个规则 `tools/label.py:_cfour_sph_rules` 已经实现了**（`m_sort_key` 把 `m==0` 映射到
`+inf`，其余映射到 `(-|m|, 0 if m>0 else 1)`）。

pyscf 的 `basis='dyallv3z'` 对 Sm 与 CFOUR 的 `SM:DYVTZUNC` **逐位一致**（本来就是未收缩的），
所以这个算例不需要自己解析 GENBAS。

### 3.2 现成的转换代码

```python
from socutils.tools import label

# (2*nao_sph, n2c) 的酉矩阵：CFOUR 的 α⊕β 球谐基 → pyscf 的 j-adapted 旋量基
C = label.Labels.spinor_cfour2pyscf(mol)

# 读一个 (2nc, 2nc) 的性质矩阵并转到 socutils 的旋量基
n2c = mol.nao_2c()                       # == 2*nc
flat = label.carr_f2py(mol, 'SOX2CDPZ')  # 交错复数 → complex128 一维数组
M_cfour = flat.reshape(n2c, n2c).T       # .T 就是 Fortran 列主序
M_pyscf = C.conj().T @ M_cfour @ C
```

`somf/x2cmp.py:864 read_qed()` 就是这个模式的现成用例（它读的是 4c 的 QED 修正，
用 `block_diag(C, C)` 拼成 4c 版本）。

### 3.3 实测吻合度

| 检验 | 误差 |
|---|---|
| CFOUR 重叠 vs pyscf `int1e_ovlp`（球谐基，加置换） | max 1.4e-15 |
| `SOLZ` vs pyscf `int1e_cg_irxp[z]` | 4.9e-16 相对 |
| `X2COVLPM` → 旋量基 vs `int1e_ovlp_spinor` | 1.3e-15 |
| 旋量配方 vs `int1e_ovlp_spinor` | 4.4e-16 |
| `Tr(D·S)`，D 取自 `Sm.chk` 的真实 socutils SCF 密度 | **60.000000000000014**（应为 60）|

**不需要任何符号翻转** —— 实球谐的符号约定两边一致。

---

## 4. 语义陷阱（决定正确性的部分）

格式已经解决了；接下来这些是真正会算错的地方。

### 4.1 `x_response` 约定逐文件不同 ⚠️ 最重要

CFOUR 内部走两条路径：`sox2c1o0`（含 X 响应）和 `sox2c1onox`（不含）。

- **绝大多数文件** = `x_response=True`（能量导数算符）。
  实测：`SOX2CDPZ` 对 `X2CMP(...).get_hfw1(op4c, x_response=True)` 吻合到 **1.1e-9**，
  对普通图像变换 `[U†OU]_++` 只有 **3.8e-5**；`SOEFG` 槽 0 分别是 **5e-11 相对** vs **0.4%**。
- **`SODEN2C`、`SOMOSS2C`** = `x_response=False`。
- **`SODEN2CG`** = 同一个接触密度但响应打开。
- **`SOSIGZ`、`SOSIGX`、`SOLZ`** = **完全不做图像变换**。

后果：**接口不能设一个全局开关**，必须每个文件带自己的元数据。
另外，因为主流文件是 `x_response=True`，它们是能量导数算符，
**用于跃迁矩是错的** —— `prop/int1e.py` 和 `tdscf/gks.py:126` 对跃迁矩刻意用
`x_response=False`，CFOUR 的偶极文件在那里是错的对象。

### 4.2 单位与前因子全部由 Python 侧负责

`sox2cprop/evaprp.f90` 极其单薄：每个性质就是一次 `Tr(D·O)`，加上三种后处理 ——
电子电荷的符号翻转、核贡献相加（**只有电偶极和 EFG 有**）、以及**仅对偶极**的
Debye 换算（`/0.3934303`）。

Fortran 里**没有**任何 MHz / mm·s⁻¹ / V·m⁻² 换算，**没有**核 g 因子，**没有**核四极矩。
CFOUR 把超精细/EDM/NEC 全部当作「a.u. 的裸电子学参数」输出。**所有同位素相关的因子都是
Python 侧的工作**。

### 4.3 KB vs UT（`SOM*` vs `UTM*`）

差别是真实且巨大的：`UTM01` 逐元素约为 `SOM01` 的 **10 倍**（超精细），g 因子上约 **7%**。
参考：Cheng, *Mol. Phys.* **121**, e2113567 (2023)。

好消息：CFOUR 的本征基公式 `V + (ε_p T_pq + ε_q T*_qp)/2c²` 与 socutils
`prop/gfac`、`prop/hfc/dhf.py` 里已经实现的对易子形式 `h m⁻¹ τ − τ m⁻¹ h`
**代数上等价** —— 只要喂同样的 `h4c`/`m4c`，两边按构造就一致。

### 4.4 已解码的算符恒等式（可直接当验证用）

- `SOSIGZ` = `σ_z ⊗ S`（精确），`SOSIGX` = `σ_x ⊗ S`
- `X2CEFGZZ` 与 `SOEFG` 的槽 1 **逐字节相同**（实测差为**精确 0**）
- 自由原子球对称性自洽（可当 `SOEFG` 分量序的独立佐证）：三个对角槽的 Frobenius 范数彼此相等、
  三个非对角槽彼此相等，比值 **2/√3 = 1.1547**；六个槽两两相对差 1–2 量级（确实互不相同），
  各自厄米到 ~1e-16
- ~~`UTM10Z − UTML0Z = (g_e/4)·σ_z·S`~~ ⚠️ **这条是错的，不能当验证用**（2026-08-31 实测）。
  最小二乘拟合给出的系数是 **0.465**（而 `g_e/4 = 0.5006`，差 7%），max-norm 残差 20–70%。
  原因：`UTM10*`/`UTML0*` 是**图像变换过**的算符（`x_response=True`），而 `SOLZ`/`SOSIGZ`
  **完全不做图像变换**（§4.1），Sm（Z=62）下两者的差不可能小。
  **结构是对的**：把 `UTM10Z`/`UTML0Z` 同时对 `SOLZ` 和 `σ_z S` 做二元拟合，`SOLZ` 系数在
  两者中都是 ≈ 0.4866（≈ ½），而 `UTML0Z` 的 `σ_z S` 系数是 0.0014（≈ 0）——
  即 `½(L + g_e S)` 与 `UTML0* = g_e→0` 是**非相对论极限**的结构，不是精确恒等式。

---

## 5. socutils 现状：性质计算为什么「有点问题」

### 5.1 两条互不相干的车道

**新车道**：`prop/int1e.py` + `SpinorSCF.prop_int1e` + `tdscf/gks.py` + `fci/addons.py`。
文档写得好，思路正确（4c 组装 → 用建 Hamiltonian 的同一套解耦变换 → 收缩回来，
并区分 `x_response`），自检通过（PC(identity)−S = 4e-11，PC(r) 对 pyscf 独立实现 7e-13）。

但是：
- **只注册了一个算符**（`ELECTRIC_OPS = {'r': ...}`，`int1e.py:72-75`）
- **在 git 里是未跟踪状态**（`?? prop/int1e.py`）
- 没有 per-atom / `rinv` 机制
- **对 GHF（自旋轨道基）参考给出静默错误**：`prop_int1e` 会返回**旋量基**矩阵，
  然后被拿去和 GHF 的自旋轨道密度缩并。实测：z 轴上的分子电子偶极 x 分量给出
  **−0.43 a.u.**（正确值 0）。

**旧车道**：`prop/{efg,hfc,gfac,effective_field,contact_density}`。
11 个文件与 2024-03 快照**逐字节相同**，没有 `__init__.py`、没有文档、没有 example、没有测试。

### 5.2 旧车道的实际状态（逐个跑过）

直接抛异常的：

- `hfc/spinor_hf.kernel` 和 `hfc/dhf.kernel` → `NameError: name 'mol' is not defined`
  （`hfc_nuc is None` 时命中，应为 `method.mol`）
- `contact_density/spinor_hf` → **连 import 都失败**，`_decontract_spinior` 拼写错误
- **所有 `Xresp=False` 分支** → `ModuleNotFoundError: No module named 'socutils.somf.eamf'`
  （该模块已从活 repo 删除）。而 `Xresp=False` 是两个接触密度 kernel 的**默认值**。

能跑的只有 `efg/{rhf,spinor_hf}`、`gfac/{dhf,spinor_hf}`、`effective_field/*`，其中两个返回 `None`。

结构性问题：

- 五份手抄的同一条五步流水线（decontract → 建 4c 算符 → `get_hfw1` → 反收缩 → `einsum('ij,ji->')`）
- **重复且被浪费的 X2C 构建**：`efg/spinor_hf.py:48`、`hfc/spinor_hf.py:29,61`、
  `effective_field/spinor_hf.py:45` 各自调一次 `x2c_grad.x2c1e_hfw0`（未收缩基下的完整 4c 对角化），
  然后立刻被 `method.with_x2c.h4c/m4c` **覆盖**。既白算几秒，又埋雷 ——
  本地那个是裸 X2C-1e，缓存那个含 AMF/Gaunt/Breit，两者不是一回事。
- **`hfc/spinor_hf.py:50-54` 的 `contr_coeff` 是错的**：用标量收缩系数按 `0::2/1::2` 交错拼 α/β，
  而同文件的 `int_hfc_2c:22` 用的是正确的 `_decontract_spinor`。同一文件两条路径不一致。
- `efg/spinor_hf.py:73` 在循环体内重绑定了循环变量 `efg_nuc`
- 关键字名分裂：`x_response` vs `Xresp`；同一物理开关默认值分裂：
  2c kernel 里 `utm=True`，4c kernel 里 `utm=False`
- 复制粘贴错的 docstring / 日志串（`effective_field/spinor_hf.py` 的 docstring 写着 "Contact density"）
- 遗留调试痕迹（`print('hehe')`）、遗留 profiling 代码
- **就地修改 X2C helper 缓存的 `h4c`**
- 导入时副作用：`warnings.warn(...)` + 猴补丁 `JHF.EFG = ...` / `JHF.HFC = ...`
- `prop/` 下**所有 `__init__.py` 缺失**（靠 namespace package 侥幸工作）

### 5.3 最要命的缺口

**只吃 `method.make_rdm1()`，没有任何入口接受外部密度矩阵。**
而 Sm 这个 case 要的恰恰是 CAHF/SHCI 的态分辨密度和跃迁密度。
`hci/shci.py` 的 `transition_dipole` 是注释掉的，而且用的是裸的非相对论球谐
`mol.intor('int1e_r')`。

**以及：不存在「算符已经是 2c 了，直接缩并」的入口** —— 这正是 CFOUR 文件需要的入口。

---

## 6. 验证阶梯

因为没有 CFOUR 打印的参考数值（§0），验证靠内部一致性。按风险从低到高：

1. **容器**：每个文件读出来，长度能被 `8·nc²` 整除，商等于预期的 item 数。
2. **AO 顺序**：CFOUR 重叠 ↔ pyscf `int1e_ovlp`（已做，1.4e-15）。
3. **单算符、无图像变换**：`SOLZ` ↔ `int1e_cg_irxp[z]`（已做，4.9e-16）；
   `SOSIGZ` ↔ `σ_z ⊗ S`（已做，精确）。
4. **图像变换语义**：`SOX2CDPZ` ↔ `with_x2c.get_hfw1(op4c, x_response=True)`（已做，1.1e-9），
   并确认 `x_response=False` 明显更差（3.8e-5）—— 这一步同时验证了约定和 socutils 自己的实现。
5. **多分量 + 原子项**：`SOEFG` 6 个槽各自厄米、互不相同、三个对角槽之和为零（已做，1.1e-14）。
6. **密度缩并**：`Tr(D·S) = 60` 用 `Sm.chk` 的真实密度（已做，60.000000000000014）。
7. **端到端**：`Tr(D·X2COVLPM) = 60.000000000000`，用 `Sm.chk` 的真实 socutils SCF 密度
   经 `prop/cfour.py` 全链路（读盘 → 旋转到 pyscf 旋量基 → 缩并），与 pyscf 自己的重叠一致。
   ✅ 2026-08-31 由 `prop/cfour.py` 复现，第 1–6 级全部通过（数字见下）。
8. **多原子**：✅ **2026-09-01 已验证**（§6.2）。旋转 H₂O（一般欧拉角、离原点、`SYM=OFF`、
   `FIXGEOM=ON`、`UNIT=BOHR`、非收缩 cc-pVDZ、点核、纯 SO-X2C-1e 不跑 `xx2camf`）：
   CFOUR `xsox2cprop` 打印的 HF 偶极 ↔ socutils 用 CFOUR 的 `SOX2CDP*` + 自己的 `X2CMP('1e')`
   SCF 密度经 `prop.observables.dipole` 算出的偶极，**1.5e-12**；HF 能量 3e-13。
   多原子的原子顺序（`atom_map=[0,1,2]`）、p/d 壳层 m 序、复数 MO 系数一次覆盖。

---

### 6.2 多原子端到端：CFOUR 自己算的 HF 偶极（2026-09-01）

算例在 `~/scr4/zmcscf/h2o_x2cprop_1e`（`ZMAT`、五个 `*.out`、全部 `SO*` 文件、
`compare_socutils.py`）。序列 `xjoda → xsox2catm → xsox2chf → xsox2cprop`（`~/apps/cfour-x2cso/bin`；
`xx2camf` 需要 `LD_LIBRARY_PATH=~/apps/conda/lib` 才能起，这个算例刻意不跑它）。

| 检验 | 实测 |
|---|---|
| HF 能量 socutils `X2CMP('1e')` vs CFOUR `xsox2chf` | 3.4e-13 |
| 核偶极 `Σ Z_A R_A` vs CFOUR 打印的 "Nuclear dipole" | 2.3e-13 |
| 总偶极 `dipole(mf, store)` vs CFOUR "Total Electric Dipole Moments" | **1.5e-12** |
| 同一几何、`xx2camf` + `%amfiMethod* 0 1 0 0` vs socutils `x2camf(with_gaunt=True, with_breit=False)` | 能量 1e-12，偶极 1.7e-9（`h2o_x2cprop/`；那 1.7e-9 是下面的质心问题，未重跑） |

同一算例、同一密度，其余时间反演为偶的性质（`compare_even_properties.py`，2026-09-01）：

| 检验 | 实测 |
|---|---|
| `efg(mf, store)` 六槽总 EFG（`-Tr` + `EFGNUC`）vs CFOUR "Total Electric Field Gradients"，O / H / H | 5.2e-12 / 1.1e-13 / 4.1e-13 |
| `efg_principal` 本征值（\|V\| 升序）vs CFOUR "Eigenvalues of EFG"（只印 6 位） | 3.2e-7（= 打印舍入） |
| `EFGNUC` vs 用 mol 点电荷公式 `Σ_B Z_B (3r r − r²δ)/r⁵` | 1.1e-12 |
| `nuclear_density(..., 'contact' / 'contact_derivative' / 'effective')` vs CFOUR 三段 "Density at atm#" | 相对 5.1e-13（O 上 300.545 差 6e-11） |
| 奇宇称零检验：`SOX2CEFD`、`SOMQM` 对闭壳层密度 | −5e-15、1.5e-11（CFOUR 1.4e-14、1.3e-10） |

#### 6.2.1 时间反演为奇的算符：CFOUR UHF 的 g 因子 / 超精细（2026-09-01）

CFOUR 不算 g 张量，但 `REFERENCE=UHF` 的开壳层 2c-HF 会打印 `<μ>` 期望值（"g fac"、`UTML0*`、
"MHC-KB/UT"）。两个算例，同一基组/核模型/坐标约定，开壳层 `xsox2chf` 必须加
`SCF_DAMPING=500 SCF_EXPSTART=30 SCF_MAXCYC=600` 才收敛（不加：201 步残差 1e-4，属于 DIIS 在
Kramers 对取向那个极平方向上游荡）。

**三层对比**（`compare_odd_properties.py`）：
(a) 直接用 CFOUR 的 MO（`scf2ccmo`，`C_py = c† C_c`，最低 N 个旋量占据）构密度；
(b) socutils `X2CMP('1e')` SCF 从 (a) 的密度起步；(c) socutils 从默认猜测起步。

| 算例 | 层 | E | 偶：μ / EFG / ρ | 奇：g(UT,KB) / UTML0 / MHC(KB,UT) |
|---|---|---|---|---|
| H₂O⁺ ²B₁（旋转，离原点） | (a) | 5e-13 | 2e-13 / 5e-15 / 2e-13 | 1e-15 / 2e-16 / 2e-13 |
| | (b) 2 步收敛到同一态 | 1e-13 | 1e-12 / 2e-11 / 6e-11 | 1e-10 / 1e-12 / 7e-11 |
| | (c) | **未收敛**，1.6e-7 | 4e-7 / 1e-5 / 2e-6 | **g 方向完全不同**（\|g\| 0.5026 vs 0.5005） |
| OH ²Π₃/₂（轴沿一般方向） | (a) | 2e-12 | 2e-13 / 2e-15 / 2e-12 | 7e-16 / 3e-16 / 3e-13 |
| | (b) 5 步 | 2e-12 | 2e-13 / 2e-10 / 2e-11 | 1e-10 / 2e-13 / 1e-10 |
| | (c) | 2e-12 | 1e-11 / 5e-9 / 3e-11 | **全部严格反号**，绝对值 4e-9；g·n = ∓1.000474257，\|g×n\| 1e-11 |

结论：
- 算符、旋转、符号约定在奇宇称这边同样确认（(a) 层 1e-13～1e-16），包括 KB 与 UT 两套、
  轨道部分 `UTML0*`（OH 上 ≈ 0.5·n，Λ=1）。
- 一个 Kramers 二重态的 `<μ>` 不是态的性质而是**所选 Kramers 分量的性质**：非线性分子（H₂O⁺）
  赝自旋方向任意，两套程序各落一个方向，能量面在这个方向上平到 SCF 都收不住；线性分子
  Ω 是好量子数，轴向被对称性固定，只剩 Ω=±3/2 的**符号**歧义——这正是用户建议用线性分子的原因。
  这也就是为什么 g 张量必须走 Kramers 对内的 2×2 块（跃迁密度），单个期望值给不出。

#### 6.2.2 CI 级 g 因子 / 超精细 vs CFOUR UHF：物理层面对比（2026-09-01）

`oh_x2cprop_1e/ci_g_factor.py`。OH，同一 CFOUR 算符文件。轨道：OH⁻ 闭壳层 KR 轨道起步，
`cahf_diis.mc_diis`（`canonicalize_=natorb=False`，level_shift 0.5）做 CAHF(1 in 2)（active =
SOC 分裂后的 ω=±3/2 π 对）和 CAHF(3 in 4)（4 个 π 旋量等价平均）。CI 级的 `<μ>` **不能按根取**——
Davidson 在简并对内给任意复组合——而是在简并能级内构 d×d 块 `M_a[i,j] = Σ O_pq <i|p†q|j>`
（`zfci.FCISolver.trans_rdm1`，全空间 γ：core 单位阵只进对角块），沿轴/垂直分解；A 在
g_∥>0 的本征态里取值以保留相对符号。

| 波函数 | E − E_UHF (mHa) | 能级 (cm⁻¹) | g_∥ | L_∥ | A_O∥ | A_H∥ |
|---|---|---|---|---|---|---|
| CFOUR UHF（Ω=3/2 分量） | 0 | — | +1.000474 | +0.499977 | −1.970589 | −0.084500 |
| CASCI(1,2) @ CAHF(1,2) | +2.15 | 0 (×2) | +1.000476 | +0.499972 | −1.799217 | −0.118507 |
| CASCI(3,4) @ CAHF(3,4) | +2.26 | 0 (×2) / 203 (×2) | +1.000478 / +0.000575 | +0.499972 / −0.499833 | −1.774704 / +2.666874 | −0.118027 / +0.045466 |
| CASCI(5,8) @ CAHF(3,4) | +1.27 | 0 / 203 / 36991 | +1.000478 / +0.000575 / +0.500533 | +0.499972 / −0.499833 / 0.000004 | −1.762102 / +2.679567 / −1.938672 | −0.117808 / +0.045679 / −0.333191 |
| CASCI(5,8) @ CAHF(1,2) | +1.16 | 0 / 311 / 37194 | 同上到 1e-6 | | −1.786659 | −0.118289 |

- **g_∥ 在 CI 与 UHF 之间一致到 2e-6**（case (a) ²Π₃/₂ 的 g_∥ 由 Λ、Σ 定，对相关不敏感）；
  Ω=1/2 能级 g_∥ ≈ 6e-4（²Π₁/₂ 的著名 g≈0），Ω=3/2 能级垂直块 1e-9～1e-11，Ω=1/2 能级垂直块非零
  （2e-3 / 1.3，物理上允许）；37000 cm⁻¹ 的 ²Σ 能级 g_∥ = 0.5005、L = 0。
- **SOC 分裂** Ω=1/2 − Ω=3/2 = 203（3in4 轨道）/ 311（1in2 轨道）cm⁻¹，实验 139 cm⁻¹，符号（倒置）正确。
- **超精细差别是真的物理**：KR 的 CAHF core 没有自旋极化，O 上差 9%，H 上差 40%（H 的 Fermi
  接触项主要来自 core 自旋极化）；UHF 把它算进去了，CAHF/CASCI 没有。
- CAHF(1in2) 轨道上做 CASCI(5,8) 把 Ω=1/2 推到 311 cm⁻¹（两 π 对被 1in2 弄得不等价），CAHF(3in4)
  才是平衡的起点；CASCI(1,2) @ CAHF(3in4) 会按能量排序选到 ω=±1/2 那对——active 必须显式指定。

**取向检验（`oh_xy_x2cprop_1e`，轴放在 xy 平面、与旋量量子化轴 z 正交）。** socutils 的
CAHF(1,2)/(3,4) 与 CASCI(1,2)/(3,4)/(5,8) 块结果与上一取向**逐位相同**（能量 1e-8，g/L/A 到 1e-6，
Ω=3/2 能级垂直块 1e-10～1e-11）：块方法取向无关。而 CFOUR 的 UHF 在这个取向落到了**另一个解**：
E = −75.447620，比 Ω=3/2 解低 1.9 mHa，g = (0.006, −0.021, −0.456) 沿实验室 z 而非分子轴——
π 空穴取实组合、自旋沿 z 极化的对称性破缺行列式（Π 态实轨道 HF 本来就比 Λ 本征的复轨道 HF 低
mHa 量级，而 OH 的 SOC 只有 0.6 mHa）。用它的 MO 构密度，算符仍 1e-15 一致，socutils 从它起步 3 步
收敛到同一解；socutils 默认猜测则停在 g≈1e-7 的自旋淬灭非共线旋量上（未收敛）。
**结论：开壳层 UHF 的 "g 因子" 对 Π 态依赖初猜与取向，第一取向的 1.000474 只是碰巧落在 Ω=3/2 解；
做标定要用 CI 的简并能级块。**

**g 张量与超精细张量（`oh_x2cprop_1e/gtensor_oh.py`、`gA_oh.py`）。** 在简并能级块上，
`T_ab = Tr(M_a σ_b)` 给出任意赝自旋基下的 g̃、L̃、A；换基是同一个三维旋转 R 右乘所有张量，
所以 g̃g̃ᵀ = 2Tr(M_aM_b)、g̃Aᵀ 是不变量。用左极分解 g̃ = P·R 把赝自旋系钉在 g 上（P ≥ 0），
A' = A·Rᵀ。两取向结果逐位相同、所有张量对称到 1e-12、主轴自动落在分子轴：Ω=3/2 能级
g = 2g̃ = (4.00191, 0, 0)（case (a) 理论 4.0023），L̃ = (0.99994, 0, 0)，A_O = (−3.549407, 0, 0)，
A_H = (−0.236054, 0, 0)，A 相对 g 为负；Ω=1/2 能级 g = (0.0023, 0.0079, 0.0079)。
g̃Aᵀ_nn：+4.003821 / +2.000844 / −7.102205 / −0.472334（g/L/A_O/A_H），两取向一致到打印位。
注意：g̃ 为秩 1 时（g_⊥ = 0）绕赝自旋轴的旋转未定，A 的垂直赝自旋系无法钉到 g 上——此处
A_⊥ 也为 0 所以无害；G 只给 |g|，A 的符号是相对 g（取 P ≥ 0）的。

**收进 `prop/` 的接口（2026-09-01，约定定稿）。** `densities.level_densities(mc)` 逐简并能级
yield `(d, d, nmo, nmo)` 的态/跃迁密度（`trans_rdm1`，⟨p†q⟩ 约定，core 单位阵只在对角项）；
`observables.level_block` 给 `⟨i|O_a|j⟩` 块并检查厄米性；`pseudo_spin_tensor` T_ab = Tr(M_a σ_b)；
`pseudo_spin_frame` 左极分解 T_g = P·R；**`g_tensor` 返回 g = 2P**（因子 2 = 1/μ_B，自由电子给 g_e；
对称、主值 ≥ 0，det 符号不声称）；**`hyperfine_tensor` 返回 A' = T_A·Rᵀ**（H = Σ A_ab I_a S̃_b，裸 a.u.、
不乘核因子，符号相对 g）。d ≠ 2 的能级给 NaN 并记日志。`balance='UT'|'KB'` 选文件。两取向上与脚本
逐位一致；自包含单元测试 `prop/test/test_gtensor.py`（pyscf 旋量 X2C 的 SOC + 裸 (L+g_eS)/2）。

#### 6.2.3 NH ³Σ⁻ 的 g 位移 vs 文献 CC（自旋三重态，2026-09-02）

算例 `~/scr4/zmcscf/nh_gtensor`（纯 X2C-1e）与 `nh_gtensor_amf`（X2CAMF + Gaunt），
`g_tensor.py`。NH 沿 z，r = 1.957794 Bohr，非收缩 aug-cc-pVTZ（GENBAS 缺 `N:APVTZUNC`，
按 pyscf 的 `gto.uncontract(gto.load('aug-cc-pvtz','N'))` 生成了一条，H 用 GENBAS 现成的
`H:APVTZUNC`，指数与 pyscf 逐位相同；83 个 AO ↔ CFOUR 83）。轨道：NH²⁺ 闭壳层 KR 轨道 →
CAHF(2 in 4)（1π 壳层）→ CASCI。参考：Gauss 等的 CC 表（aug-cc-pVDZ，同键长）。

| SOC | 活性空间 | ZFS (cm⁻¹) | Δg_⊥ (ppm) | Δg_∥ (ppm) |
|---|---|---|---|---|
| X2C-1e | (4,6) / (4,10) / (6,12) | 0.46–0.55 | 2077 / 2080 / 2071 | −120.4 / −121.0 / −121.2 |
| **X2CAMF + Gaunt** | (4,6) / (4,10) / (6,12) | 0.16–0.20 | **1261 / 1265 / 1261** | **−118.6 / −119.2 / −119.4** |
| 文献 HF / CCSD / CCSDT / CCSDTQ | — | — | 1081 / 1314 / 1329 / **1331** | −110.7 / −106.2 / −106.0 / **−106.0** |

- **二电子 SOC 是主项**：1e → AMF 把 Δg_⊥ 从 2077 压到 1261（−39%），正是第一行原子 2p 的
  spin-other-orbit 屏蔽量级；Δg_∥ 几乎不动（−120 → −119），因为 ³Σ⁻ 的 g_∥ 没有轨道贡献，
  只剩相对论质量修正。
- 我们的 Δg_⊥ = 1261 落在文献 HF(1081) 与 CCSD(1314) 之间：CAS 只有静态相关（文献里相关把
  Δg_⊥ 抬了 23%），而活性空间从 (4,6) 到 (6,12) 只动 4 ppm，说明差的是动态相关不是 CAS 大小。
- Δg_∥ = −119 vs −106：X2C 图像变换的 RMC 与一阶 Breit-Pauli 的差别（1e 与 AMF 都给 −119，
  与 SOC 无关）。
- 激发态 ¹Δ 14560、¹Σ⁺ 26810 cm⁻¹（实验 12690 / 21200），小 CAS 偏高；ZFS 0.16 cm⁻¹ 只含二阶
  SOC，实验 1.86 cm⁻¹ 主要是自旋-自旋偶极项，不在这个哈密顿量里。
- CFOUR 的 UHF 在这个算例收到 M_S=0 分量（g 全 1e-12），拿不到 HF 参考数——又一次说明
  开壳层 UHF 的 g 因子依赖初猜，标定要用 CI 流形块。

**代码上的推广**：`spin_invariant(block)` 给 `g gᵀ = (4/K)·Tr(M̃_a M̃_b)`（K = S(S+1)(2S+1)/3，
M̃ 是减去迹的部分——迹是整体能移不是塞曼项），对任意 d 成立且与流形基无关；`g_tensor` 在
d = 2 时仍走极分解（SVD，不平方，g_⊥ = 0 能给到 1e-11，不变量路线开方后只有 1e-8），d > 2 用
不变量。超精细张量仍只对 d = 2 给（d > 2 时一般 SU(d) 混合不是自旋矩阵的转动，g 定不出系）。

#### 6.2.4 Sm³⁺ 4f⁵ ⁶H_5/2 的 g 因子 vs 原子实验（J = 5/2，d = 6 流形，2026-09-02）

数据全部来自 `~/scr4/zmcscf/sm`（只读）：`Sm.chk` 的 CAHF(5 in 14) 轨道
（`cahf_mo_coeff_x2camf_dcb`，活性空间列 54:68）、`cderi.chk` 的 DF 向量、CFOUR 的
`UTM10*`/`UTML0*` 算符。脚本 `sm_gtensor.py`（CASCI + `observables.g_tensor`）和
`sm_decomp.py`（轨道/自旋分解），**必须在别的目录里跑**——CASCI 会往当前目录写 FCIDUMP，
会覆盖 sm 里 Dice 那次的输入输出。CASCI(5 in 14)、24 根、< 2 分钟。

| J | d | E 计算 | E NIST ASD | g 计算 | g Landé | g NIST ASD |
|---|---|---|---|---|---|---|
| 5/2 | 6 | 0 | 0 | **0.291193** | 0.284058 | （空，ASD 没给基态 g） |
| 7/2 | 8 | 990.66 | [1080] | 0.826836 | 0.824992 | 0.79 |
| 9/2 | 10 | 2181.01 | [2290] | 1.070117 | 1.070869 | 1.00 |

各向异性 4e-9～5e-8（自由原子必须各向同性，这是流形块自身的一致性检验）；简并度 6/8/10
与 2J+1 完全对上；`degen_tol` 1e-6 与 1e-4 Ha 给同样的分组。

轨道/自旋分解（`UTML0*` 给纯轨道部分 a，b = (g − a)/g_e）：

| J | a 计算 | a (LS) | b 计算 | b (LS) |
|---|---|---|---|---|
| 5/2 | 1.703840 | 1.714286 | −0.705506 | −0.714286 |
| 7/2 | 1.170569 | 1.174603 | −0.171667 | −0.174603 |
| 9/2 | 0.928316 | 0.929293 | +0.070818 | +0.070707 |

**偏差是波函数的还是算符的**（`sm_opcheck.py`，同一批 CI 向量换算符）：

| J | g Landé | g（CFOUR 图像变换） | g（裸 (L+g_eS)/2） | a（CFOUR） | a（裸） | a (LS) |
|---|---|---|---|---|---|---|
| 5/2 | 0.284058 | 0.291193 (+2.51%) | 0.291882 (+2.75%) | 1.703840 | 1.706480 | 1.714286 |
| 7/2 | 0.824992 | 0.826836 (+0.22%) | 0.827714 (+0.33%) | 1.170569 | 1.171887 | 1.174603 |
| 9/2 | 1.070871 | 1.070117 (−0.07%) | 1.071086 (+0.02%) | 0.928316 | 0.929079 | 0.929293 |

图像变换只值 −0.24 个百分点（方向对：相对论修正压低 g），**+2.5% 全在波函数里**：a 比 LS 小
0.61%、|b| 小 1.23%，自旋投影缩得快一倍，而 g = a − g_e|b|，净效果是抬高。

**结论**：机器本身没问题（各向同性 5e-8、简并度 6/8/10、⟨L⟩/⟨S⟩ 分解、算符/波函数分离都自洽），
基态 g ≈ 0.29 这个高度非平凡的数量级（L 与 S 几乎抵消）也出来了。

**但"和实验对不对得上"这个问题在这个体系上问不干净**：
- 我们相对 Landé 偏**高** 2.5%（J=5/2），ASD 给出 g 的两个能级相对 Landé 偏**低** 4–7% —— 符号相反，
  所以不能用"中间耦合太弱"一句话解释（那只会让偏差变小、不会变号）。
- ASD 这些数的出处是 Martin, Zalubas & Hagan 1978（NSRDS-NBS 60，稀土能级汇编）。Sm IV 的能级
  本身在汇编里就是**方括号**的（内插/估计值），g 只给两位小数，且**基态 J=5/2 根本没有 g**。
  ASD 页面没有区分实测与推算的说明。挂在一个内插能级上的两位小数 g，几乎不可能是自由离子
  Zeeman 实测，更像是同一套半经验中间耦合拟合的产物 —— 也就是说这不是一个能判定 ±3% 的基准。
- 真正干净、可判定的差异是**能级间隔**：我们的 SOC 劈裂比汇编值低 8%（991 vs 1080）和 5%
  （2181 vs 2290），而 ⟨L⟩、⟨S⟩ 投影离纯 LS 只有 0.6–1.2% —— ab initio 的 ζ_4f/F^k 偏小
  （CASSCF 层面 F^k 系统性偏大是已知问题），波函数几乎停在 LS 极限。这一条与 g 的符号问题
  独立成立，也是先该修的东西（更大活性空间 / 动态相关），不是性质接口的问题。

#### 6.2.5 磁超精细的诊断手段（Sm³⁺ 4f⁵，2026-09-02）

`sm_hfc.py`。**超精细没有 Landé 那样的绝对参考**：A 里含 ⟨r⁻³⟩（径向）和核矩，两者都不是
角动量代数能给的。所以下面全是**内部一致性检验**，外加两条可做的外部对比。

| J | A_KB | 各向异性 | A_UT | UT/KB |
|---|---|---|---|---|
| 5/2 | 10.688920 | 3.2e-07 | 10.694537 | 1.000525 |
| 7/2 | 7.382061 | 9.7e-08 | 7.383190 | 1.000153 |
| 9/2 | 5.978965 | 4.0e-08 | 5.978119 | 0.999858 |

（裸 a.u.，不含同位素因子。）

1. **自由离子的各向同性**：3e-7 —— 检验 CI/流形块本身，不检验物理。
2. **KB vs UT**：同一可观测量的两套图像变换方案，**一致到 5e-4**。注意这与 §6.1 记的
   "`UTM01`/`SOM01` 逐元素比 10.323" 不矛盾：那是**裸矩阵**的统计（由深核区元素主导），
   4f 的期望值上两套方案没有那个差别。判断超精细处理是否可靠要看期望值，不是矩阵元。
3. **多重态内的投影检验**（⟨r⁻³⟩ 在比值里约掉）：对任意矢量算符，单一 LS 项 + 单一径向参数下
   ⟨V·J⟩ = α⟨L·J⟩ + β⟨S·J⟩。三个 J 能级、两个未知数 ⇒ 一个无参数预言。

| 量 | 用哪两个能级拟合 | 预言第三个的偏差 |
|---|---|---|
| A (KB) | 7/2+9/2 / 5/2+9/2 / 5/2+7/2 | −2.06% / +0.93% / −1.67% |
| g | 同上 | +0.15% / −0.02% / +0.02% |
| g（用精确的 α=1, β=g_e，零自由参数） | — | −2.45% / −0.22% / +0.07% |

   读法：g 既有内部一致性检验**又有绝对锚点**（α、β 已知），A 只有前者。A 的残差（1–2%）
   比 g 的（0.02–0.15%）大一个量级，说明**超精细对偏离纯 LS 的敏感度比 g 高得多**，是更
   灵敏的耦合方案探针；而拟合出的 g 的 β = 1.992 比 g_e 低 0.5%，偏差集中在 J=5/2。

外部对比只有两条路：(a) **code-to-code**——CFOUR 自己打印的 MHC-KB/UT，但只对单行列式
（已在 OH / H₂O⁺ 上做到 1e-13，Sm 需要 CFOUR 收敛一个开壳层 UHF）；(b) **实验**——需要具体
同位素的 μ_I 和实测 A（¹⁴⁷Sm/¹⁴⁹Sm），文献数由使用者提供，不从记忆里编。

另有两个物理层面的诊断（已在 OH 上见过）：**core 自旋极化**——KR 的 CAHF core 给不出 Fermi
接触项，UHF 与 CAHF/CASCI 在轻原子上差 9%（O）/ 40%（H）；**接触密度 `SODEN2C`** 与超精细
共享核附近的 s 密度，可交叉验证 core 处理。

两个必须对上的约定，任何一个不对就停在 1e-8 量级：

- **坐标系**。`FIXGEOM=ON` 保留取向但 xjoda 仍**平移到质心**（joda.out "Coordinates (in bohr)"）。
  pyscf 的 mol 必须用同一组质心坐标；而且质心要用 **CFOUR 的质量表**（`joda/pertable.f`：
  O 15.99491463，H 1.007825035）。用 pyscf/NIST 的质量差 1.6e-9 a.u.，全部落在核偶极上。
- **核模型**。`sox2catm` 的有限核开关是 `iflags2(80)==2`（`NUC_MODEL=GAUSSIAN`），默认点核，
  与 pyscf 默认一致；算例里显式写了 `NUC_MODEL=POINT`。

另外确认：`O:PVDZUNC`/`H:PVDZUNC` 与 `gto.uncontract(gto.load('ccpvdz', ·))` 指数逐位相同；
CFOUR 的 `SCF_CONV=13` 与 socutils 的 `conv_tol=1e-14, conv_tol_grad=1e-10` 是达到 1e-12 所需的
（`SCF_CONV=10` / `conv_tol=1e-12` 时差 2e-8，全是密度收敛）。

### 6.1 `prop/cfour.py` 实测复现（2026-08-31）

| 检验 | 实测 | 文档原值 |
|---|---|---|
| 40 个文件的容器类型 + 项数 vs 分类学 | 全对；9 个裸 dump 全部正确识别 | §2.1 / §2.3 |
| `X2COVLPM` vs `int1e_ovlp_spinor` | 1.332e-15 | 1.3e-15 |
| `SOSIGZ` / `SOSIGX` vs `int1e_sigma` | 1.221e-15 / 9.992e-16 | 「精确」 |
| `SOLZ` 反厄米性 | 8.882e-17 | — |
| `SOLZ` vs `int1e_cg_irxp[z]`（相对） | 8.882e-16 | 4.9e-16 |
| `SOEFG` 三对角槽和为零（相对） | 1.457e-14 | 1.1e-14 |
| `X2CEFGZZ` vs `SOEFG[zz]` | 精确 0 | 「逐字节相同」 |
| `UTM01` / `SOM01` 逐元素比 | 10.323 | ~10 |
| `Tr(D·S)`，真实 `Sm.chk` 密度 | 60.000000000000 | 60.000000000000014 |

⚠️ **这些容差本身有个未解决的基准问题**：`~/apps/pyscf/pyscf/data/nist.py:22` 的
`LIGHT_SPEED = 137.0359991` 是**未提交的本地修改**（上游 `137.03599967994`）。
c 差 4.23e-9 → **1/c² 差 8.46e-9**，比 §6 第 4 级当作数值地板的 `SOX2CDPZ` 吻合度 **1.1e-9 还大 8 倍**。
在把 c 和 CFOUR 对齐之前，1.1e-9 不能当作任何其他检验的容差基准。

---

## 7. 实施方案（2026-08-31 定稿）

设计经一轮多 agent 方案竞争 + 对抗批判（operator-first 胜出），并经与用户逐条讨论确定。
下面每条都是**已决**，不是建议。

### 7.1 分层与硬规矩

```
prop/cfour.py        算符层来源 A：CFOUR 目录（本轮做）
prop/int1e.py        算符层来源 B：socutils 自造（本轮不动）
  ↓ 裸 a.u. 的 (n2c, n2c) ndarray
缩并层               唯一一份 Tr(O·D)
  ↓ 裸 a.u. 的标量 / 数组
可观测量层           符号、核项、3×3 组装、（可选）对角化
```

**硬规矩（唯一需要的那条）**：**性质对象不得把波函数存成属性**。
密度适配器可以接收波函数作参数，但自己无状态。这条成立，单体就长不回来。
`prop/cfour.py` 的 `CfourPropStore` 只持有 `mol`，够不着任何 `make_rdm1`。

### 7.2 已决的设计要点

1. **算符与密度解耦。** SCF 密度不是特例 —— 上游 `evaprpscf` 就是 `evaprp` 在 `D = 1_occ` 时的
   特例却被抄成另外 400 行，socutils 抄了 5 份，同一个病。
2. **保留 CFOUR 的数据模型，丢掉它的控制流。** 它的模式是「把算符物化到 store，一次流一个出来」，
   store 才是解耦机制；过程式的坏味道（硬编码 `inquire` 链、手算偏移、`write(*,*)` 拿不到值、
   手工内存 arena）全是 Fortran 逼出来的，不属于数据模型。
3. **元数据不骑在 ndarray 上。** `lib.tag_array` 实测在**取分量、翻符号、加核项、reshape**
   时全部丢失（只有 `.conj()` 幸存），机制是 `NPArrayWithTag` 没有 `__array_finalize__`
   （`pyscf/lib/numpy_helper.py:1442`）。改由**名字**承载：算符端靠文件名查 `FILES` 表。
4. **「哪种密度合法」做进 API 表面，不做运行时检查。** 上游 `evaprptran ⊂ evaprp` 已经这么选了：
   有 `transition_dipole` 而**没有** `transition_contact_density` —— 用户打不出这个调用。
5. **表管到裸 a.u.，函数管从裸 a.u. 到物理量。** 这条线是数据给的（§4.2）。
6. **算符来源经 `mf.with_props` 挂载**，与 `with_x2c` / `with_df` 同构；声明式，构造时说明要算
   哪些性质、哪些原子，于是 `mc.EFG()` 可以不带参数。
7. **门面不放在波函数类上。** 那是 N 个波函数 × M 个性质份复制粘贴。改用**密度适配器**：
   每种波函数一个无状态适配器，性质对象可调用、波函数是**调用参数**而非属性 →
   **N + M**。加一种波函数 = 一个适配器；加一种性质 = 一个类。

### 7.2b 缩并层：默认 MO 基（2026-09-01 定）

**默认在 MO 基缩并**，AO 只作逃生口：

```python
contract_dm(O, gamma, mo_coeff=mo)   # 主路径：算符转到 MO 基，einsum('pq,pq->', O_mo, gamma)
contract_dm(O, dm_ao)                # mo_coeff=None 走 AO（mf.make_rdm1() 就是 AO 密度）
```

**弛豫归波函数自己管。** 缩并层不认识「弛豫」这个概念,也没有对应的参数或元数据 ——
波函数给什么密度就缩什么,要弛豫密度就由波函数交出弛豫密度。

数学上两者恒等（`Tr(O_AO·D_AO) = Tr(O_MO·γ_MO)`），**选 MO 是为了把两类静默错误从路径里
结构性地拿掉**，而不是靠检查去防：

1. **AO 有转置歧义，MO 没有。** AO 密度要写 `C·γ^T·C†` 还是 `C·γ·C†`，树里两种都有（§7.5），
   而两个候选**都厄米、都 `Tr(S·D)=N`**，没有便宜的检查能分开 —— 实测代价是 Kramers 禁阻
   跃迁上 7 个数量级的假信号。MO 基下没有 `C` 参与，`einsum('pq,pq->', O, γ)` 只有一种写法。
2. **AO 有基的歧义，MO 没有。** 旋量 vs 自旋轨道，而 `nao_2c() == 2·nao_nr()` **精确相等**，
   形状永远分不出来（§5.1 那个 −0.43 a.u. 的机制）。MO 基下 γ 的维度是 nmo，与用哪套 AO 无关。
3. 成本方向相反但都不是瓶颈：AO 密度 6 MB/态、算符只转一次；MO 密度小、算符每换 `mo_coeff`
   重转一次。实测真正贵的是**建**算符（0.53 s）而不是缩并（0.004 s）。

**接口不做 core/active 切分。** 适配器给全空间的 `(γ_MO, mo_coeff)`，性质层一条 einsum、
零分支；core/active 是适配器内部的事。core 也用密度矩阵缩并而不是 `Tr(O_core)` ——
两者在 `γ_core = I` 时恒等，但 core **不总是**单位阵（CAHF 分数占据、`scf/addons.even_occ`、
`lo/atomic_ref`），把「core 满占据」焊死在代码里会在那些情形悄悄出错。

AO 逃生口不是可选项：SCF 的 `mf.make_rdm1()` 本来就只给 AO 密度。走 AO 时语义由调用方负责，
与 CFOUR 那条独立路径同一个原则。

### 7.3 已决的范围（本轮）

| | |
|---|---|
| CFOUR 算符读取层 `prop/cfour.py` | ✅ **已完成并验证**（§6.1）|
| 缩并层 + 密度适配器 | 本轮做 |
| 可观测量层 | 本轮做 |
| socutils 自造算符（扩 `prop/int1e.py`）| **不做**；只保留它现有的三个 early-return 分支 |
| 旧车道五个 kernel | **不碰** |
| CC / EOM 性质 | **不做**（但适配器形状不得堵死以后加）|
| CFOUR picture 一致性校验 | **不做** —— 用户决定：CFOUR 吐出来的当作正确 |
| 同位素因子 / 单位换算 | **不做** —— 裸 a.u. 输出，使用者自己拉表乘核素常数 |

### 7.4 被这个范围消解掉的「物理选择」

多 agent 综合提出了 5 个物理选择，其中 **4 个因「CFOUR 的选择即正确」而消解** ——
它们全是 CFOUR 写文件时已经做完的决定，我们只**记录**不**选择**：
KB/UT（`SOM01` vs `UTM01` 就是表里两行，`balance` 只是名字选择器）、
EFG 无迹/有迹（CFOUR 是无迹；有迹那个是 socutils 自造算符的事）、
`pc`/`energy_derivative`（§4.1 的逐文件约定是既成事实）、
AMF/Gaunt/Breit 响不响应微扰（在文件里面）。
第 5 个（核数据）因「不做单位换算」而消解。

### 7.4b pyscf 的先例：MO 与 AO 是两个接口，且两族默认方向相反

| | 默认返回 | 另一个怎么拿 |
|---|---|---|
| CC（`cc/ccsd.py:1157`） | **MO 基**（docstring: "in MO space"） | `make_rdm1(ao_repr=True)` |
| CASCI/CASSCF（`mcscf/casci.py:1057`） | **AO 基** | 没有开关，AO 是唯一选项 |
| fcisolver（`fcisolver.make_rdm1`） | **活性空间 MO** | 只有这一种 |

AO 那个本质是 MO 那个 + 一次基变换（`cc/ccsd_rdm.py:274` 的 `if ao_repr:`）。CASCI 因为要拼
core 才当场做（`casci.py:1069-1071`；其中的 `* 2` 是非相对论闭壳层每个空间轨道两个电子，
**旋量基下没有这个 2**）。

**`make_rdm1` 是导出格式，不是内部货币。** `mc1step.py` 里 `casdm1` 出现 45 次，每个优化点用的
都是 `fcisolver.make_rdm12`（活性 MO）；AO 的 `make_rdm1` 在整个 `mc1step.py` 里只出现在注释里
—— **CASSCF 轨道优化从头到尾不碰 AO 密度**。pyscf 里调它的全是把总 AO 密度交给外部消费者的场景：
`solvent/_attach_solvent.py`（5 处）、`qmmm/pbc/itrf.py:375`、`mcscf/addons.py:708`（自然占据分析）。

socutils 一个这样的消费者都没移植，所以 `zcasci`/`zcasbase` 的 `make_rdm1` **零调用是结构上正常的**
（`somf/somf.py:373` 那个唯一疑似调用点接的是**标量 pyscf CASSCF** —— 全走 `_sph` 积分且
`mo_coeff` 不取共轭）。这也是为什么我们的性质层走 MO 基不是抄近路：**那就是 pyscf 内部自己的做法**。

继承上 pyscf 只在 `casci.CASBase` 定义一次 `make_rdm1`，`CASCI` 和 `CASSCF` 都继承不重写
（`CASSCF` 继承的是 `CASBase` 而非 `CASCI`）。socutils 在 `zcasbase.CASBase:202` 和
`zcasci.CASCI:466` **各写了一份**，派生类那份重写了基类那份，而两份只差 `.conj().T` 与 `.T.conj()`
的写法 —— 行为完全相同，所以这次重写没有意义，后果是同一个转置 bug 抄成两份、只改一份等于没改。

### 7.5 仍然悬着的（socutils 自造算符那半边接进来时才活）

- **socutils 的 EFG 和 CFOUR 的不是同一个算符**：`prop/efg/spinor_hf.py:96-99` 和 `:105-108`
  的去迹在 LL 和 SS **两处都被注释掉了**（有迹），CFOUR 是无迹。交叉验证前必须先统一。
- **`get_hfw1` 的缓存只做了一半**：`somf/x2c_grad.py:216-242` 每次 `x_response=True` 调用都在
  `get_Asqi1` 里重算 `sqrtm(S0)`、`inv(Ssq0)`、`A`、`eigh(A)` —— 全是零阶量，与算符/原子/分量无关；
  `x2c_intermediate` 一个都没缓存。外加两个纯 Python 双重循环（nc=308 时 1,517,824 + 379,456 次）。
  所以「建一次复用」在**密度**轴上成立、在 **(原子, 分量)** 轴上不成立，而 40 原子 EFG 正好活在后者。
- **`prop/int1e.py:123` 硬编码的 `.5/c`** 和三个旧的 odd 算符都不匹配，HFC 照它移植会差一个 **c**。
- **AO 密度转置约定树内不一致**：`mcscf/zcasbase.py:216` 建 `C·d·C†`，
  `fci/addons.py:108 trans_rdm1_ao` 建 `C·d^T·C†`。
- **`SpinOrbitalX2CMPHelper.h4c` 是旋量矩阵**（`x2cmp.py:537`）而同一对象的 `get_hcore` 返回
  自旋轨道（`:1005-1007`），且 `nao_2c() == 2·nao_nr()` 精确相等 —— 形状对得上、什么都不报，
  这就是 §5.1 那个 −0.43 a.u. 的机制。
- **§5.2 的「`contr_coeff` 是错的」被推翻**（实测五个基组下 bit-identical），但一致是巧合不是构造。
- **pyscf 的 `ISOTOPE_QUAD_MOMENT` 符号被剥掉**：Sm-147 给 `+0.263`，随附源表
  `quadrupole_moment.txt:701` 是 `-0.26(3)`（符号丢失 + 不确定度数字拼到尾数）。
  Sm-149 是 `+0.078(8)`，**与 147 符号相反**。本轮不做换算所以够不着我们，但用者自己乘的时候要注意。

### 7.6 `with_x2c` 侧的目标形态（2026-09-01 定方向，性质线做完再动）

`somf/` 里的 X2C helper 已经清点过：spinor 侧有 pyscf `SpinorX2CHelper`（1e）、`snso`（1e +
Boettger）、`amf.SpinorX2CAMFHelper`（已 deprecated，≡ `x2cmp(x2cmp='x2camf')`）、
`x2cmp.SpinorX2CMPHelper`（`amf_type` 字符串分发约 20 个 flavor）、`x2cmmf`（从收敛 4c DHF）；
外加 GHF 孪生。**类的数量不是问题，`x2cmp.py:548-800` 那条 if/elif 链才是。**

定下的方向：**`x2cmp.SpinorX2CMPHelper` 是最终的 all-in-one `with_x2c`，其余都是它的 flavor。**
依据是实现本来就已经是一条流水线 —— 每个 flavor 都是「造 `(h4c, m4c)`，可选加一个 2c 修正
`so_2c`」，然后同一个 `x2c1e_hfw0_4cmat` 做 FW、同一个 `to_2c`、同一个 `get_hfw1(h4c1, x_response)`
做 picture change。SNSO 落在这条线上就是把 `h4c` SS 块里 `W − pVp` 乘 `boettger_damp`（`snso.py`
今天抽出来的函数，就是为这个留的）；MMF 就是 `h4c` = 收敛的 4c Fock；`'1e'` 已经有。

- 分发变成一张表：`FLAVORS[name](x2cobj, xmol) -> (h4c, m4c, so_2c|None)`；`get_hcore`/`get_hfw1`/
  `save_hcore`/`load_hcore` 各一份。实验性 flavor 用同一签名在 `somf/flavors_experimental.py` 注册。
- **GHF 是旋量的视图，不是第二套实现**：`SpinOrbital*` 只做 `spinor2sph(spinor hcore)`（x2cmp 的
  GHF 孪生现在已经是这么做的；`snso.py` 在 24da800 里也改成了这样，与直算版差 2e-11）。`get_hfw1`
  同理转一次。这同时消掉 7.5 里「`h4c` 是旋量而 `get_hcore` 是自旋轨道」那个形状陷阱的一半。
- `amf.SpinorX2CAMFHelper`、`x2cmmf` 的类、`snso` 的两个类到时候收掉；`spinor_hf.x2camf()/.x2cmp()`
  快捷方式保留，再加一个 `.x2c(flavor=…)`。
- 性质层从 helper 只要 `get_hcore` 和 `get_hfw1`（或 pyscf 的 `picture_change`）两样，统一前后都
  成立，所以当前分支不依赖这次重构；**动手前需要用者圈定 20 个 flavor 里哪些还活着**（猜：
  `x2camf, x2camf_sd, x2cmp, sfmp, 1e, sf1e` 活，其余实验）。
