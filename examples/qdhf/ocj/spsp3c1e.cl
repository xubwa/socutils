#!/usr/bin/env clisp
(load "gen-code.cl")

(gen-cint "int3c1e_spsp.c"
  '("int3c1e_spsp"              (sigma dot p \, sigma dot p \,))
  '("int3c1e_spsp_r2_origk"     (sigma dot p \, sigma dot p \, r dot r))
  '("int3c1e_spsp_r4_origk"     (sigma dot p \, sigma dot p \, r dot r r dot r))
  '("int3c1e_spsp_r6_origk"     (sigma dot p \, sigma dot p \, r dot r r dot r r dot r))
  '("int3c1e_ipvip1_r2_origk"   (nabla \, nabla \, r dot r))
  '("int3c1e_ipvip1_r4_origk"   (nabla \, nabla \, r dot r r dot r))
  '("int3c1e_ipvip1_r6_origk"   (nabla \, nabla \, r dot r r dot r r dot r))
)
