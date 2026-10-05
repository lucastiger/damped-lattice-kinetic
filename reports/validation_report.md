# Validation report

- generated: 2026-10-04T10:30:58+00:00
- commit: `28b3a51`
- manifest: `manuscript/claims.yaml`
- mode: strict

**129 passed, 0 failed, 0 skipped of 129 claims.**

## All claims, by manuscript location

### Table 2

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `VAL-chat1` | `critical/c_hat1` | `0.8989297` | `0.8989297486` | |x - 0.89893| <= 3e-07 | PASS |
| `VAL-sigmahat1` | `critical/sigma_hat1` | `0.6501918` | `0.6501917819` | |x - 0.650192| <= 2e-07 | PASS |

### Table 2 / Table 5

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `VAL-cmax` | `c_max` | `0.900196` | `0.9001958592` | |x - 0.900196| <= 5e-06 | PASS |

### Sec. 8, Resolution

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `VAL-bc-sigma` | `bc_check/sigma` | `0.6165589432` | `0.6165589432` | |x - 0.616559| <= 1e-09 | PASS |

### Sec. 8

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `VAL-bc-left` | `bc_check/err_left` | `5.39e-09` | `5.387450486e-09` | within 0.6 decades of 5.39e-09 | PASS |
| `VAL-bc-right` | `bc_check/err_right` | `5.05e-09` | `5.054385244e-09` | within 0.6 decades of 5.05e-09 | PASS |

### Obs. 9.2

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `ID-sigma` | `c089/sigma` | `0.637300691` | `0.637300691` | |x - 0.637301| <= 1e-09 | PASS |
| `ID-sigmap` | `c089/sigma_prime` | `1.925132061` | `1.925132061` | |x - 1.92513| <= 1e-08 | PASS |
| `ID-kappa` | `c089/kappa` | `12.0959614798` | `12.09596148` | |x - 12.096| <= 1e-07 | PASS |
| `ID-orth` | `c089/phat_dot_cos` | `1e-11` | `1.153374271e-13` | x <= 1e-11 | PASS |
| `ID-pb` | `c089/power_balance_relerr` | `1e-12` | `1.121893696e-13` | x <= 1e-12 | PASS |
| `ID-edge` | `c089/dcphi_edge` | `2.498179` | `2.498179223` | |x - 2.49818| <= 2e-06 | PASS |
| `ID-edge-pred` | `c089/dcphi_edge_pred` | `2.498175` | `2.498175473` | |x - 2.49817| <= 2e-06 | PASS |
| `ID-sweep-abs` | `sweep/max_abs_err` | `1e-09` | `6.173479505e-10` | x <= 1e-09 | PASS |
| `ID-sweep` | `sweep/max_relerr` | `5e-10` | `1.75468602e-10` | x <= 5e-10 | PASS |

### Obs. 9.5

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `ID-resphat` | `c089/res_phat_rel` | `1e-12` | `7.09810347e-14` | x <= 1e-12 | PASS |
| `NV-sv1` | `sv/s1` | `2.66e-09` | `2.659952617e-09` | within 1.0 decades of 2.66e-09 | PASS |
| `NV-sv2` | `sv/s2` | `0.09887` | `0.09887184427` | |x - 0.09887| <= 0.002|0.09887| | PASS |
| `NV-sv3` | `sv/s3` | `0.1909` | `0.1909331948` | |x - 0.1909| <= 0.002|0.1909| | PASS |
| `NV-loc-p0` | `localization/p0` | `0.9898` | `0.989814293` | |x - 0.9898| <= 0.002 | PASS |
| `NV-loc-phat` | `localization/phat` | `0.9898` | `0.9897944073` | |x - 0.9898| <= 0.002 | PASS |
| `NV-refl-min` | `weight_test/reflection_ratio_min` | `0.993` | `0.9932246245` | |x - 0.993| <= 0.03 | PASS |
| `NV-refl-max` | `weight_test/reflection_ratio_max` | `1.047` | `1.046836164` | |x - 1.047| <= 0.03 | PASS |
| `NV-slope` | `weight_test/observed_slope` | `0.097` | `0.09676158401` | |x - 0.097| <= 0.002 | PASS |
| `NV-slope-sign` | `weight_test/slope_sign_matches_weight` | `False` | `False` | == False | PASS |

### Sec. 9.2 text

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `ID-fd` | `fd_discrepancy` | `0.00016` | `0.0001629523771` | within 0.5 decades of 0.00016 | PASS |

### Table 3

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `CV-1024` | `convergence/L200_N1024/relerr` | `0.000305` | `0.0003049501876` | within 1.0 decades of 0.000305 | PASS |
| `CV-2048` | `convergence/L200_N2048/relerr` | `1.26e-08` | `1.256729366e-08` | within 1.0 decades of 1.26e-08 | PASS |
| `CV-4096` | `convergence/L200_N4096/relerr` | `1e-12` | `4.185377905e-14` | x <= 1e-12 | PASS |
| `CV-3072` | `convergence/L300_N3072/relerr` | `1.26e-08` | `1.256731451e-08` | within 1.0 decades of 1.26e-08 | PASS |
| `CV-6144` | `convergence/L300_N6144/relerr` | `1e-11` | `9.325315684e-14` | x <= 1e-11 | PASS |
| `CV-150` | `convergence/L150_N2048/relerr` | `2.73e-12` | `2.701991863e-12` | within 1.0 decades of 2.73e-12 | PASS |
| `CV-res-1024` | `convergence/L200_N1024/res_M0p0` | `0.098` | `0.09755741204` | within 0.5 decades of 0.098 | PASS |
| `CV-res-2048` | `convergence/L200_N2048/res_M0p0` | `0.00029` | `0.0002934492623` | within 0.5 decades of 0.00029 | PASS |
| `CV-res-4096` | `convergence/L200_N4096/res_M0p0` | `2.5e-09` | `2.465436422e-09` | within 0.5 decades of 2.5e-09 | PASS |
| `CV-res-150` | `convergence/L150_N2048/res_M0p0` | `6.8e-06` | `6.829112762e-06` | within 0.5 decades of 6.8e-06 | PASS |

### Table 4

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `TH-0895-sp` | `rows/0.89500/sigma_prime` | `1.5331386` | `1.533138595` | |x - 1.53314| <= 2e-07 | PASS |
| `TH-0895-k` | `rows/0.89500/kappa` | `9.6329939` | `9.632993895` | |x - 9.63299| <= 2e-06 | PASS |
| `TH-0895-m` | `rows/0.89500/m` | `209.0` | `208.9990114` | |x - 209| <= 0.002|209| | PASS |
| `TH-0895-np` | `rows/0.89500/nu2_pred` | `-0.046091` | `-0.0460910979` | |x - -0.046091| <= 2e-06 | PASS |
| `TH-0895-nu` | `rows/0.89500/nu2` | `None` | `None` | is null | PASS |
| `TH-0898-sp` | `rows/0.89800/sigma_prime` | `0.7418062` | `0.741806238` | |x - 0.741806| <= 2e-07 | PASS |
| `TH-0898-k` | `rows/0.89800/kappa` | `4.6609061` | `4.660906055` | |x - 4.66091| <= 2e-06 | PASS |
| `TH-0898-m` | `rows/0.89800/m` | `236.5` | `236.5322732` | |x - 236.5| <= 0.002|236.5| | PASS |
| `TH-0898-np` | `rows/0.89800/nu2_pred` | `-0.019705` | `-0.01970515901` | |x - -0.019705| <= 2e-06 | PASS |
| `TH-0898-nu` | `rows/0.89800/nu2` | `-0.021961` | `-0.021960854` | |x - -0.021961| <= 3e-06 | PASS |
| `TH-0898-r` | `rows/0.89800/rho2` | `0.97584` | `0.9758413139` | |x - 0.97584| <= 2e-05 | PASS |
| `TH-08988-sp` | `rows/0.89880/sigma_prime` | `0.1477746` | `0.1477746308` | |x - 0.147775| <= 2e-07 | PASS |
| `TH-08988-k` | `rows/0.89880/kappa` | `0.9284954` | `0.9284953892` | |x - 0.928495| <= 2e-06 | PASS |
| `TH-08988-m` | `rows/0.89880/m` | `273.97` | `273.9733432` | |x - 273.97| <= 0.002|273.97| | PASS |
| `TH-08988-np` | `rows/0.89880/nu2_pred` | `-0.003389` | `-0.003388999011` | |x - -0.003389| <= 2e-06 | PASS |
| `TH-08988-nu` | `rows/0.89880/nu2` | `-0.003441` | `-0.003440634` | |x - -0.003441| <= 3e-06 | PASS |
| `TH-08988-r` | `rows/0.89880/rho2` | `0.99618` | `0.9961792869` | |x - 0.99618| <= 2e-05 | PASS |
| `TH-CRIT-sp` | `rows/0.89892/sigma_prime` | `0.0118952` | `0.01189519013` | |x - 0.0118952| <= 2e-07 | PASS |
| `TH-CRIT-k` | `rows/0.89892/kappa` | `0.0747397` | `0.07473968385` | |x - 0.0747397| <= 2e-06 | PASS |
| `TH-CRIT-m` | `rows/0.89892/m` | `283.38` | `283.383805` | |x - 283.38| <= 0.002|283.38| | PASS |
| `TH-CRIT-np` | `rows/0.89892/nu2_pred` | `-0.000264` | `-0.000263740138` | |x - -0.000264| <= 2e-06 | PASS |
| `TH-CRIT-nu` | `rows/0.89892/nu2` | `-0.000264` | `-0.000264048` | |x - -0.000264| <= 3e-06 | PASS |
| `TH-0899-sp` | `rows/0.89900/sigma_prime` | `-0.0900527` | `-0.09005272388` | |x - -0.0900527| <= 2e-07 | PASS |
| `TH-0899-k` | `rows/0.89900/kappa` | `-0.565818` | `-0.5658179516` | |x - -0.565818| <= 2e-06 | PASS |
| `TH-0899-m` | `rows/0.89900/m` | `290.42` | `290.4159272` | |x - 290.42| <= 0.002|290.42| | PASS |
| `TH-0899-np` | `rows/0.89900/nu2_pred` | `0.001948` | `0.001948302068` | |x - 0.001948| <= 2e-06 | PASS |
| `TH-0899-nu` | `rows/0.89900/nu2` | `0.001931` | `0.001931469` | |x - 0.001931| <= 3e-06 | PASS |
| `TH-0899-r` | `rows/0.89900/rho2` | `1.00215` | `1.002150773` | |x - 1.00215| <= 2e-05 | PASS |
| `TH-08992-nu` | `rows/0.89920/nu2` | `0.00774` | `0.007739949` | |x - 0.00774| <= 3e-06 | PASS |
| `TH-08995-nu` | `rows/0.89950/nu2` | `0.017591` | `0.017591345` | |x - 0.017591| <= 3e-06 | PASS |
| `TH-08998-nu` | `rows/0.89980/nu2` | `0.029743` | `0.029743461` | |x - 0.029743| <= 3e-06 | PASS |
| `TH-0900-nu` | `rows/0.90000/nu2` | `0.040624` | `0.040624036` | |x - 0.040624| <= 3e-06 | PASS |
| `TH-0900-m` | `rows/0.90000/m` | `-101.4` | `-101.3724231` | |x - -101.4| <= 0.005|-101.4| | PASS |

### Obs. 9.3(a)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `TH-zero-sp` | `zeros/sigma_prime_interp` | `0.8989293` | `0.8989293343` | |x - 0.898929| <= 3e-07 | PASS |
| `TH-zero-nu` | `zeros/nu2_interp` | `0.8989296` | `0.8989296214` | |x - 0.89893| <= 5e-07 | PASS |

### Obs. 9.3(b)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `TH-rel-0898` | `reduction_error/0.89800` | `0.103` | `0.1027143565` | |x - 0.103| <= 0.01 | PASS |
| `TH-rel-08988` | `reduction_error/0.89880` | `0.015` | `0.01500740533` | |x - 0.015| <= 0.003 | PASS |

### Obs. 9.3(e)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `TH-mscan` | `m_scan/all_positive` | `True` | `True` | == True | PASS |

### Obs. 9.3(d)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `TH-msign` | `m_sign_change_between_08999_09000` | `True` | `True` | == True | PASS |

### Obs. 9.4(a)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `FD-cmax-frac1` | `zero_fraction_cdot` | `0.336` | `0.3356303178` | |x - 0.336| <= 0.015 | PASS |
| `FD-cmax-frac2` | `zero_fraction_phat_one` | `0.342` | `0.341708106` | |x - 0.342| <= 0.015 | PASS |
| `FD-samestep` | `same_step` | `True` | `True` | == True | PASS |

### Obs. 9.4(b)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `FD-kappa` | `kappa_at_fold` | `-2.54` | `-2.533068446` | |x - -2.54| <= 0.02 | PASS |

### Obs. 9.4(c)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `FD-nu` | `nu_at_fold` | `0.0653` | `0.06577817197` | |x - 0.0653| <= 0.002 | PASS |
| `FD-count` | `n_unstable_exactly_one_all_steps` | `True` | `True` | == True | PASS |

### Obs. 9.4(d)

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `FD-invariant` | `invariant_max` | `2e-08` | `7.714181615e-09` | x <= 2e-08 | PASS |

### Table 5

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `FD-row23-c` | `rows/23/c` | `0.900196` | `0.9001959276` | |x - 0.900196| <= 2e-06 | PASS |
| `FD-row23-cdot` | `rows/23/cdot` | `1.38e-05` | `1.378686637e-05` | |x - 1.38e-05| <= 2e-06 | PASS |
| `FD-row23-one` | `rows/23/phat_one` | `-0.005925` | `-0.005925188541` | |x - -0.005925| <= 0.0002 | PASS |
| `FD-row24-one` | `rows/24/phat_one` | `0.011415` | `0.01141472362` | |x - 0.011415| <= 0.0002 | PASS |
| `FD-row24-cdot` | `rows/24/cdot` | `-2.73e-05` | `-2.729066935e-05` | |x - -2.73e-05| <= 2e-06 | PASS |

### Table 6

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `NV-d0-p0` | `decay/0/p0` | `0.6742` | `0.6741719465` | within 0.3 decades of 0.6742 | PASS |
| `NV-d0-ph` | `decay/0/phat` | `0.4784` | `0.4783716189` | within 0.3 decades of 0.4784 | PASS |
| `NV-d10-p0` | `decay/10/p0` | `0.1223` | `0.1222867194` | within 0.3 decades of 0.1223 | PASS |
| `NV-d10-ph` | `decay/10/phat` | `0.1484` | `0.1483716766` | within 0.3 decades of 0.1484 | PASS |
| `NV-d20-p0` | `decay/20/p0` | `0.005328` | `0.005328243066` | within 0.3 decades of 0.005328 | PASS |
| `NV-d20-ph` | `decay/20/phat` | `0.006655` | `0.00665483211` | within 0.3 decades of 0.006655 | PASS |
| `NV-d40-p0` | `decay/40/p0` | `0.01192` | `0.01192328552` | within 0.3 decades of 0.01192 | PASS |
| `NV-d40-ph` | `decay/40/phat` | `0.01189` | `0.01188802494` | within 0.3 decades of 0.01189 | PASS |
| `NV-d80-p0` | `decay/80/p0` | `0.0001187` | `0.0001186504397` | within 0.3 decades of 0.0001187 | PASS |
| `NV-d80-ph` | `decay/80/phat` | `0.0001239` | `0.00012388333` | within 0.3 decades of 0.0001239 | PASS |
| `NV-d120-p0` | `decay/120/p0` | `2.655e-07` | `2.654733474e-07` | within 0.4 decades of 2.655e-07 | PASS |
| `NV-d120-ph` | `decay/120/phat` | `1.186e-07` | `1.186201432e-07` | within 0.4 decades of 1.186e-07 | PASS |
| `NV-d199-p0` | `decay/199/p0` | `1.063e-09` | `1.062790995e-09` | within 0.5 decades of 1.063e-09 | PASS |
| `NV-d199-ph` | `decay/199/phat` | `1.086e-09` | `1.086123606e-09` | within 0.5 decades of 1.086e-09 | PASS |

### Obs. 9.6

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `SP-zero` | `c089/nu_zero_abs` | `1e-07` | `7.254020419e-09` | x <= 1e-07 | PASS |
| `SP-mgamma` | `c089/nu_minus_gamma` | `-0.1` | `-0.1000000073` | |x - -0.1| <= 1e-07 | PASS |
| `SP-pair` | `c089/pairing_max_err` | `1e-07` | `7.253571506e-15` | x <= 1e-07 | PASS |
| `SP-nreal` | `c089/n_real_nontrivial` | `0` | `0` | |x - 0| <= 0 | PASS |
| `SP-circle` | `c089/circle_radius` | `0.945369` | `0.9453691666` | |x - 0.945369| <= 1e-06 | PASS |
| `CS-random` | `random_K/relerr` | `1e-12` | `1.920685833e-14` | x <= 1e-12 | PASS |

### Table 7

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `GN-fk-f` | `rows/fk/f` | `0.48196` | `0.4819599977` | |x - 0.48196| <= 2e-06 | PASS |
| `GN-fk-fp` | `rows/fk/fprime` | `2.174156` | `2.174155967` | |x - 2.17416| <= 3e-06 | PASS |
| `GN-fk-k` | `rows/fk/kappa` | `1.146659` | `1.146659361` | |x - 1.14666| <= 3e-06 | PASS |
| `GN-fk-e` | `rows/fk/relerr` | `1e-08` | `1.795513172e-10` | x <= 1e-08 | PASS |
| `GN-m05-f` | `rows/mu2m05/f` | `0.540326` | `0.5403255333` | |x - 0.540326| <= 2e-06 | PASS |
| `GN-m05-fp` | `rows/mu2m05/fprime` | `2.676268` | `2.67626817` | |x - 2.67627| <= 3e-06 | PASS |
| `GN-m05-k` | `rows/mu2m05/kappa` | `1.069733` | `1.069732749` | |x - 1.06973| <= 3e-06 | PASS |
| `GN-m05-e` | `rows/mu2m05/relerr` | `3.93e-06` | `3.928122901e-06` | within 0.5 decades of 3.93e-06 | PASS |
| `GN-nnn-f` | `rows/nnn/f` | `0.166333` | `0.1663325617` | |x - 0.166333| <= 2e-06 | PASS |
| `GN-nnn-fp` | `rows/nnn/fprime` | `0.601746` | `0.6017456529` | |x - 0.601746| <= 3e-06 | PASS |
| `GN-nnn-k` | `rows/nnn/kappa` | `0.480627` | `0.4806266005` | |x - 0.480627| <= 3e-06 | PASS |
| `GN-nnn-e` | `rows/nnn/relerr` | `1e-10` | `1.466817733e-14` | x <= 1e-10 | PASS |
| `GN-mix-f` | `rows/mu2nnn/f` | `0.191512` | `0.1915124488` | |x - 0.191512| <= 2e-06 | PASS |
| `GN-mix-fp` | `rows/mu2nnn/fprime` | `1.009012` | `1.009012031` | |x - 1.00901| <= 3e-06 | PASS |
| `GN-mix-k` | `rows/mu2nnn/kappa` | `0.735407` | `0.7354070282` | |x - 0.735407| <= 3e-06 | PASS |
| `GN-mix-e` | `rows/mu2nnn/relerr` | `1e-09` | `2.133135623e-11` | x <= 1e-09 | PASS |
| `GN-soft-f` | `rows/soft/f` | `0.379908` | `0.3799079951` | |x - 0.379908| <= 2e-06 | PASS |
| `GN-soft-fp` | `rows/soft/fprime` | `1.487688` | `1.487687634` | |x - 1.48769| <= 3e-06 | PASS |
| `GN-soft-k` | `rows/soft/kappa` | `1.177935` | `1.177935311` | |x - 1.17793| <= 3e-06 | PASS |
| `GN-soft-e` | `rows/soft/relerr` | `1e-10` | `1.502370703e-13` | x <= 1e-10 | PASS |

### Obs. 9.7

| id | key | expected | obtained | test | status |
|---|---|---|---|---|---|
| `GN-refine` | `refinement/mu2m05_N4096_relerr` | `1e-10` | `2.389947723e-12` | x <= 1e-10 | PASS |
| `GN-refine-fp` | `refinement/mu2m05_N4096_fprime` | `2.6762666` | `2.676266649` | |x - 2.67627| <= 3e-06 | PASS |
