# Stage 2：非相对论 MP2-F12/3C(FIX) 推导笔记

代码：`rf12/nr/mp2f12.py`（逐项公式），`rf12/nr/mp2f12_brute.py`（直接按算符定义，逐元素交叉检验）。
记号：物理学家记号 ⟨pq|A|rs⟩；i,j,k,l,m,n = 活性占据；o = 全部占据（含冻结芯）；a,b = OBS 虚轨道；
p,q,r = OBS 全部 MO；a′ = CABS；"1" = OBS ∪ CABS 的正交归一完备化空间（RI）；P,Q,A,C,I ∈ "1"。

## 1. Hylleraas 泛函与 ansatz
闭壳层一阶对函数 u_ij = Σ_ab T^{ij}_{ab}|ab⟩ + Σ_kl t^{ij}_{kl} Q12 f12 |kl⟩，
f12 = −exp(−γ r12)/γ（Slater 型 geminal，cusp 斜率 1），
Q12 = (1 − O1)(1 − O2) − V1V2（ansatz 3：同时投影掉占据和 OBS 虚轨道，使 F12 部分与常规双激发正交）。
SP ansatz（Ten-no）固定振幅：自旋轨道单态 1/2、三重态 1/4，空间轨道 t^{ij}_{ij} = 3/8，t^{ij}_{ji} = 1/8（i=j 时 1/2）。
对每个 pair 极小化 Hylleraas 泛函，常规振幅 T 用 MP2 值（对角 Fock，approximation 下 F12–常规耦合由 C 项给出）：
  E_ij = E_ij^{MP2} + 2 t·Ṽ_ij + t·B̃_ij·t，
  Ṽ^{kl}_{ij} = V^{kl}_{ij} − Σ_ab C^{kl}_{ab} ⟨ij|ab⟩ / (ε_a+ε_b−ε_i−ε_j)，
  B̃^{kl,mn}_{ij} = B^{kl}_{mn} − (ε_i+ε_j) X^{kl}_{mn} − Σ_ab C^{kl}_{ab} C^{mn}_{ab} / (ε_a+ε_b−ε_i−ε_j)。
自旋适配（i<j 给出因子 kd=2；三重态乘 3）：
  单态 E^s_ij = kd [ 2·½·½(Ṽ_ij,ij + Ṽ_ij,ji) + ¼·¼ Σ±B̃ ]，三重态同理取反对称组合（代码中 vs, bs, vt_, bt_）。

## 2. 积分核（全部来自 libcint，需 WITH_F12 编译）
| 记号 | 算符 | libcint |
|---|---|---|
| G | 1/r12 | int2e |
| F | f12 = −e^{−γr}/γ | −int2e_stg(γ)/γ |
| FG | f12/r12 | −int2e_yp(γ)/γ |
| F2 | f12² | int2e_stg(2γ)/γ² |
| U | (∂f12/∂r12)² = e^{−2γr} | int2e_stg(2γ) |
U 的来源：对局域 f(r12)，[[t1+t2, f], f] = −(∇1f)² − (∇2f)² = −2f′²，
所以 f t f = ½(f² t + t f²) + ½[f,[t,f]] = ½{f², t} + f′²（t = t1+t2）。

## 3. 中间量的定义（RI 全部插在 "1" 中）
P12 ≡ 1 − Q12 = O1 + O2 − O1O2 + V1V2。
- V^{kl}_{ij} = ⟨kl|f Q12 g|ij⟩ = FG − Σ_pq F_{kl,pq} G_{pq,ij} − Σ_{o,a′}(F_{kl,oa′}G_{oa′,ij} + F_{kl,a′o}G_{a′o,ij})
- X^{kl}_{mn} = ⟨kl|f Q12 f|mn⟩：同上，G → F
- C^{kl}_{ab} = ⟨kl|f Q12 (F̂1+F̂2)|ab⟩ = Σ_{a′} F_{kl,aa′} f_{ba′} + (1↔2)
  （F̂|a⟩ 在 "1" 中只有 CABS 分量能通过 Q12：Q12|a′b⟩ = |a′b⟩）
- B^{kl}_{mn} = ⟨kl|f Q12 (F̂1+F̂2) Q12 f|mn⟩，approximation C：
  B = ⟨f F̂12 f⟩_C − ⟨f P12 F̂12 f⟩ − ⟨f F̂12 P12 f⟩ + ⟨f P12 F̂12 P12 f⟩，
  ⟨f F̂12 f⟩_C = U + ½⟨{f², (F̂+K̂)_1 + (F̂+K̂)_2}⟩ − ⟨f (K̂1+K̂2) f⟩
  （用 F̂ = t + v_N + Ĵ − K̂、局域算符与 f 对易、§2 的对易子恒等式；交换算符 K̂ 非局域，用 "1" 中的 RI。）

## 4. B 的逐项展开（代码中 B1–B8）
记 sym(T)_{klmn} = T_{klmn} + T_{lknm}，最后 B ← ½(B + Bᵀ)（kl ↔ mn）：
- B1 U
- B2 + sym Σ_I F2_{kl,mI}(f+k)_{In}
- B3 − sym Σ_{PCA} F_{kl,PC} k_{CA} F_{mn,PA}
- B4 − sym Σ_{o,CA} F_{kl,oC} f_{CA} F_{mn,oA}
- B5 + sym Σ_{a′,o,o′} F_{kl,a′o} f_{oo′} F_{mn,a′o′}
- B6 − sym Σ_{b,r,p} F_{kl,br} f_{rp} F_{mn,bp}
- B7 − 2 sym Σ_{a′,o,I} F_{kl,a′I} f_{oI} F_{mn,a′o}
- B8 − 2 sym Σ_{b,r,a′} F_{kl,br} f_{ra′} F_{mn,ba′}
B4–B8 是把 P12 = O1+O2−O1O2+V1V2 代入 §3 的三个投影项、并用 f_{PQ}（"1" 中的 Fock 矩阵，含 OBS–CABS 耦合 f_{oa′}、f_{ra′}）展开的结果。
**检查**：`mp2f12_brute.py` 不使用 B2–B8，而是把 P12 作为 "1"⊗"1" 上的稠密投影算符直接施加于 ⟨kl|f|xy⟩，
得到的 V、X、C、B 与逐项公式之差 ≤ 1.4×10⁻¹⁴（He、Ne fc，aug-cc-pVDZ/OptRI）。

## 5. CABS 与 CABS singles
- RI：OBS+CABS 基函数正则正交化（重叠本征值 < 10⁻⁸ 丢弃）；CABS = OBS MO 与 RI 重叠矩阵的零空间（奇异值 < 10⁻⁶），Valeev 2004 的做法。
- "1" 中的 Fock：f = h + 2J − K（占据密度来自 OBS 的 HF），k = K。
- CABS singles：分别对角化 f_oo 与 f_{(v+a′)(v+a′)}，E_S = 2 Σ |f̃_{iA}|² / (ε̃_i − ε̃_A)。

## 6. 与 Psi4 1.11 的约定差别（已核对源码 psi4/src/psi4/f12/*.cc，v1.11）
- Psi4 的 geminal 是 −exp(−β r12) 的 6-Gaussian 拟合（系数固定、指数按 β² 缩放，**没有 1/β 因子**），固定振幅仍为 3/8、1/8；
  因此只在 β = 1 时满足 cusp。我们用精确 Slater 且带 1/γ；在 γ = β = 1 时两者定义相同，差别只剩拟合误差。
- Psi4 的 Uf 是 libint 的 delcgtg2 = (∇1 f)·(∇1 f) = f′²，与 §2 的 U 相同。
- 冻结芯：冻结轨道从 pair 中去掉，但仍在 RI 投影的 o 中（两者相同）。

## 7. libcint 的 F12 积分
pip 版 pyscf（2.14）附带的 libcint 编译时没有打开 WITH_F12。我们用同一版本 libcint v6.1.3、
与 pyscf 相同的 CMake 选项再加 `-DWITH_F12=1` 编译后替换 `pyscf/lib/deps/lib/libcint.so`；
替换前后普通 ERI 逐位相同（‖ERI‖ = 71.97117841575987，H2O/cc-pVTZ）。代码在缺少 F12 积分时会报错并指向本节。
