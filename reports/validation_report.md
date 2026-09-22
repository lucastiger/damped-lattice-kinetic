# Validation report

`manuscript/claims.yaml` resolved against `data/*.json`. Every number quoted in the manuscript is listed here with the key path that produced it and the tolerance it had to meet.

## UNRESOLVED

None. Every claim that was checked passed at its stated tolerance.

## Summary

| | |
|---|---|
| claims in manifest | 125 |
| passed | 125 |
| failed | 0 |
| skipped (quick data) | 0 |
| accounted for | 125 |
| strict mode | on |
| manifest | `manuscript/claims.yaml` |
| manifest drift | none — manuscript copy matches the handoff original |
| git sha (HEAD) | `dc307e5` |
| generated | 2026-09-22T01:25:34+00:00 |
| pipeline wall time (from data metadata) | 37.6 min |

### Data files

| file | exists | quick | git sha | matches HEAD | mtime (UTC) | wall time | peak RSS |
|---|---|---|---|---|---|---|---|
| `data/01_kinetic.json` | yes | no | `485af29` | **no** | 2026-09-22T00:52:55+00:00 | 637 s | 877 MB |
| `data/02_identity.json` | yes | no | `485af29` | **no** | 2026-09-22T01:04:45+00:00 | 710 s | 2009 MB |
| `data/03_threshold.json` | yes | no | `485af29` | **no** | 2026-09-22T01:09:27+00:00 | 282 s | 1134 MB |
| `data/04_fold.json` | yes | no | `485af29` | **no** | 2026-09-22T01:17:37+00:00 | 489 s | 439 MB |
| `data/05_nullvec.json` | yes | no | `485af29` | **no** | 2026-09-22T01:18:19+00:00 | 42 s | 861 MB |
| `data/06_spectrum.json` | yes | no | `485af29` | **no** | 2026-09-22T01:18:36+00:00 | 16 s | 342 MB |
| `data/07_conformal.json` | yes | no | `485af29` | **no** | 2026-09-22T01:19:10+00:00 | 34 s | 347 MB |
| `data/08_general.json` | yes | no | `485af29` | **no** | 2026-09-22T01:19:56+00:00 | 45 s | 864 MB |

## All claims, by manuscript location

### Table 2  (2 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| VAL-chat1 | `01_kinetic.json:critical/c_hat1` | 0.8989297 | 0.8989297486 | abs 3e-07 | PASS |
| VAL-sigmahat1 | `01_kinetic.json:critical/sigma_hat1` | 0.6501918 | 0.6501917819 | abs 2e-07 | PASS |

### Table 2 / Table 5  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| VAL-cmax | `04_fold.json:c_max` | 0.900196 | 0.9001958592 | abs 5e-06 | PASS |

### Sec. 8, Resolution  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| VAL-bc-sigma | `01_kinetic.json:bc_check/sigma` | 0.6165589432 | 0.6165589432 | abs 1e-09 | PASS |

### Sec. 8  (2 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| VAL-bc-left | `01_kinetic.json:bc_check/err_left` | ~5.39e-09 | 5.387450486e-09 | 0.6 dec | PASS |
| VAL-bc-right | `01_kinetic.json:bc_check/err_right` | ~5.05e-09 | 5.054385244e-09 | 0.6 dec | PASS |

### Obs. 9.2  (9 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| ID-sigma | `02_identity.json:c089/sigma` | 0.637300691 | 0.637300691 | abs 1e-09 | PASS |
| ID-sigmap | `02_identity.json:c089/sigma_prime` | 1.925132061 | 1.925132061 | abs 1e-08 | PASS |
| ID-kappa | `02_identity.json:c089/kappa` | 12.09596148 | 12.09596148 | abs 1e-07 | PASS |
| ID-orth | `02_identity.json:c089/phat_dot_cos` | <= 1e-11 | 1.153374271e-13 | 1e-11 | PASS |
| ID-pb | `02_identity.json:c089/power_balance_relerr` | <= 1e-12 | 1.121893696e-13 | 1e-12 | PASS |
| ID-edge | `02_identity.json:c089/dcphi_edge` | 2.498179 | 2.498179223 | abs 2e-06 | PASS |
| ID-edge-pred | `02_identity.json:c089/dcphi_edge_pred` | 2.498175 | 2.498175473 | abs 2e-06 | PASS |
| ID-sweep-abs | `02_identity.json:sweep/max_abs_err` | <= 1e-09 | 6.173479505e-10 | 1e-09 | PASS |
| ID-sweep | `02_identity.json:sweep/max_relerr` | <= 5e-10 | 1.75468602e-10 | 5e-10 | PASS |

### Obs. 9.5  (6 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| ID-resphat | `02_identity.json:c089/res_phat_rel` | <= 1e-12 | 7.09810347e-14 | 1e-12 | PASS |
| NV-sv1 | `05_nullvec.json:sv/s1` | ~2.66e-09 | 2.659944483e-09 | 1 dec | PASS |
| NV-sv2 | `05_nullvec.json:sv/s2` | 0.09887 | 0.09887184427 | rel 0.002 | PASS |
| NV-sv3 | `05_nullvec.json:sv/s3` | 0.1909 | 0.1909331948 | rel 0.002 | PASS |
| NV-loc-p0 | `05_nullvec.json:localization/p0` | 0.9898 | 0.989814293 | abs 0.002 | PASS |
| NV-loc-phat | `05_nullvec.json:localization/phat` | 0.9898 | 0.9897944073 | abs 0.002 | PASS |

### Sec. 9.2 text  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| ID-fd | `02_identity.json:fd_discrepancy` | ~0.00016 | 0.0001629523771 | 0.5 dec | PASS |

### Table 3  (10 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| CV-1024 | `02_identity.json:convergence/L200_N1024/relerr` | ~0.000305 | 0.0003049501876 | 1 dec | PASS |
| CV-2048 | `02_identity.json:convergence/L200_N2048/relerr` | ~1.26e-08 | 1.256729366e-08 | 1 dec | PASS |
| CV-4096 | `02_identity.json:convergence/L200_N4096/relerr` | <= 1e-12 | 4.185377905e-14 | 1e-12 | PASS |
| CV-3072 | `02_identity.json:convergence/L300_N3072/relerr` | ~1.26e-08 | 1.256731451e-08 | 1 dec | PASS |
| CV-6144 | `02_identity.json:convergence/L300_N6144/relerr` | <= 1e-11 | 9.325315684e-14 | 1e-11 | PASS |
| CV-150 | `02_identity.json:convergence/L150_N2048/relerr` | ~2.73e-12 | 2.701991863e-12 | 1 dec | PASS |
| CV-res-1024 | `02_identity.json:convergence/L200_N1024/res_M0p0` | ~0.098 | 0.09755741204 | 0.5 dec | PASS |
| CV-res-2048 | `02_identity.json:convergence/L200_N2048/res_M0p0` | ~0.00029 | 0.0002934492623 | 0.5 dec | PASS |
| CV-res-4096 | `02_identity.json:convergence/L200_N4096/res_M0p0` | ~2.5e-09 | 2.465436422e-09 | 0.5 dec | PASS |
| CV-res-150 | `02_identity.json:convergence/L150_N2048/res_M0p0` | ~6.8e-06 | 6.829112762e-06 | 0.5 dec | PASS |

### Table 4  (33 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| TH-0895-sp | `03_threshold.json:rows/0.89500/sigma_prime` | 1.5331386 | 1.533138595 | abs 2e-07 | PASS |
| TH-0895-k | `03_threshold.json:rows/0.89500/kappa` | 9.6329939 | 9.632993895 | abs 2e-06 | PASS |
| TH-0895-m | `03_threshold.json:rows/0.89500/m` | 209 | 208.9990114 | rel 0.002 | PASS |
| TH-0895-np | `03_threshold.json:rows/0.89500/nu2_pred` | -0.046091 | -0.0460910979 | abs 2e-06 | PASS |
| TH-0895-nu | `03_threshold.json:rows/0.89500/nu2` | null | null | exact | PASS |
| TH-0898-sp | `03_threshold.json:rows/0.89800/sigma_prime` | 0.7418062 | 0.741806238 | abs 2e-07 | PASS |
| TH-0898-k | `03_threshold.json:rows/0.89800/kappa` | 4.6609061 | 4.660906055 | abs 2e-06 | PASS |
| TH-0898-m | `03_threshold.json:rows/0.89800/m` | 236.5 | 236.5322732 | rel 0.002 | PASS |
| TH-0898-np | `03_threshold.json:rows/0.89800/nu2_pred` | -0.019705 | -0.01970515901 | abs 2e-06 | PASS |
| TH-0898-nu | `03_threshold.json:rows/0.89800/nu2` | -0.021961 | -0.021960854 | abs 3e-06 | PASS |
| TH-0898-r | `03_threshold.json:rows/0.89800/rho2` | 0.97584 | 0.9758413139 | abs 2e-05 | PASS |
| TH-08988-sp | `03_threshold.json:rows/0.89880/sigma_prime` | 0.1477746 | 0.1477746308 | abs 2e-07 | PASS |
| TH-08988-k | `03_threshold.json:rows/0.89880/kappa` | 0.9284954 | 0.9284953892 | abs 2e-06 | PASS |
| TH-08988-m | `03_threshold.json:rows/0.89880/m` | 273.97 | 273.9733432 | rel 0.002 | PASS |
| TH-08988-np | `03_threshold.json:rows/0.89880/nu2_pred` | -0.003389 | -0.003388999011 | abs 2e-06 | PASS |
| TH-08988-nu | `03_threshold.json:rows/0.89880/nu2` | -0.003441 | -0.003440634 | abs 3e-06 | PASS |
| TH-08988-r | `03_threshold.json:rows/0.89880/rho2` | 0.99618 | 0.9961792869 | abs 2e-05 | PASS |
| TH-CRIT-sp | `03_threshold.json:rows/0.89892/sigma_prime` | 0.0118952 | 0.01189519013 | abs 2e-07 | PASS |
| TH-CRIT-k | `03_threshold.json:rows/0.89892/kappa` | 0.0747397 | 0.07473968385 | abs 2e-06 | PASS |
| TH-CRIT-m | `03_threshold.json:rows/0.89892/m` | 283.38 | 283.383805 | rel 0.002 | PASS |
| TH-CRIT-np | `03_threshold.json:rows/0.89892/nu2_pred` | -0.000264 | -0.000263740138 | abs 2e-06 | PASS |
| TH-CRIT-nu | `03_threshold.json:rows/0.89892/nu2` | -0.000264 | -0.000264048 | abs 3e-06 | PASS |
| TH-0899-sp | `03_threshold.json:rows/0.89900/sigma_prime` | -0.0900527 | -0.09005272388 | abs 2e-07 | PASS |
| TH-0899-k | `03_threshold.json:rows/0.89900/kappa` | -0.565818 | -0.5658179516 | abs 2e-06 | PASS |
| TH-0899-m | `03_threshold.json:rows/0.89900/m` | 290.42 | 290.4159272 | rel 0.002 | PASS |
| TH-0899-np | `03_threshold.json:rows/0.89900/nu2_pred` | 0.001948 | 0.001948302068 | abs 2e-06 | PASS |
| TH-0899-nu | `03_threshold.json:rows/0.89900/nu2` | 0.001931 | 0.001931469 | abs 3e-06 | PASS |
| TH-0899-r | `03_threshold.json:rows/0.89900/rho2` | 1.00215 | 1.002150773 | abs 2e-05 | PASS |
| TH-08992-nu | `03_threshold.json:rows/0.89920/nu2` | 0.00774 | 0.007739949 | abs 3e-06 | PASS |
| TH-08995-nu | `03_threshold.json:rows/0.89950/nu2` | 0.017591 | 0.017591345 | abs 3e-06 | PASS |
| TH-08998-nu | `03_threshold.json:rows/0.89980/nu2` | 0.029743 | 0.029743461 | abs 3e-06 | PASS |
| TH-0900-nu | `03_threshold.json:rows/0.90000/nu2` | 0.040624 | 0.040624036 | abs 3e-06 | PASS |
| TH-0900-m | `03_threshold.json:rows/0.90000/m` | -101.4 | -101.3724231 | rel 0.005 | PASS |

### Obs. 9.3(a)  (2 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| TH-zero-sp | `03_threshold.json:zeros/sigma_prime_interp` | 0.8989293 | 0.8989293343 | abs 3e-07 | PASS |
| TH-zero-nu | `03_threshold.json:zeros/nu2_interp` | 0.8989296 | 0.8989296214 | abs 5e-07 | PASS |

### Obs. 9.3(b)  (2 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| TH-rel-0898 | `03_threshold.json:reduction_error/0.89800` | 0.103 | 0.1027143565 | abs 0.01 | PASS |
| TH-rel-08988 | `03_threshold.json:reduction_error/0.89880` | 0.015 | 0.01500740533 | abs 0.003 | PASS |

### Obs. 9.3(e)  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| TH-mscan | `03_threshold.json:m_scan/all_positive` | true | true | exact | PASS |

### Obs. 9.3(d)  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| TH-msign | `03_threshold.json:m_sign_change_between_08999_09000` | true | true | exact | PASS |

### Obs. 9.4(a)  (3 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| FD-cmax-frac1 | `04_fold.json:zero_fraction_cdot` | 0.336 | 0.3356303178 | abs 0.015 | PASS |
| FD-cmax-frac2 | `04_fold.json:zero_fraction_phat_one` | 0.342 | 0.341708106 | abs 0.015 | PASS |
| FD-samestep | `04_fold.json:same_step` | true | true | exact | PASS |

### Obs. 9.4(b)  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| FD-kappa | `04_fold.json:kappa_at_fold` | -2.54 | -2.533068446 | abs 0.02 | PASS |

### Obs. 9.4(c)  (2 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| FD-nu | `04_fold.json:nu_at_fold` | 0.0653 | 0.06577817197 | abs 0.002 | PASS |
| FD-count | `04_fold.json:n_unstable_exactly_one_all_steps` | true | true | exact | PASS |

### Obs. 9.4(d)  (1 claim)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| FD-invariant | `04_fold.json:invariant_max` | <= 2e-08 | 7.714181615e-09 | 2e-08 | PASS |

### Table 5  (5 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| FD-row23-c | `04_fold.json:rows/23/c` | 0.900196 | 0.9001959276 | abs 2e-06 | PASS |
| FD-row23-cdot | `04_fold.json:rows/23/cdot` | 1.38e-05 | 1.378686637e-05 | abs 2e-06 | PASS |
| FD-row23-one | `04_fold.json:rows/23/phat_one` | -0.005925 | -0.005925188541 | abs 0.0002 | PASS |
| FD-row24-one | `04_fold.json:rows/24/phat_one` | 0.011415 | 0.01141472362 | abs 0.0002 | PASS |
| FD-row24-cdot | `04_fold.json:rows/24/cdot` | -2.73e-05 | -2.729066935e-05 | abs 2e-06 | PASS |

### Table 6  (14 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| NV-d0-p0 | `05_nullvec.json:decay/0/p0` | ~0.6742 | 0.6741719465 | 0.3 dec | PASS |
| NV-d0-ph | `05_nullvec.json:decay/0/phat` | ~0.4784 | 0.4783716189 | 0.3 dec | PASS |
| NV-d10-p0 | `05_nullvec.json:decay/10/p0` | ~0.1223 | 0.1222867194 | 0.3 dec | PASS |
| NV-d10-ph | `05_nullvec.json:decay/10/phat` | ~0.1484 | 0.1483716766 | 0.3 dec | PASS |
| NV-d20-p0 | `05_nullvec.json:decay/20/p0` | ~0.005328 | 0.005328243066 | 0.3 dec | PASS |
| NV-d20-ph | `05_nullvec.json:decay/20/phat` | ~0.006655 | 0.00665483211 | 0.3 dec | PASS |
| NV-d40-p0 | `05_nullvec.json:decay/40/p0` | ~0.01192 | 0.01192328552 | 0.3 dec | PASS |
| NV-d40-ph | `05_nullvec.json:decay/40/phat` | ~0.01189 | 0.01188802494 | 0.3 dec | PASS |
| NV-d80-p0 | `05_nullvec.json:decay/80/p0` | ~0.0001187 | 0.0001186504397 | 0.3 dec | PASS |
| NV-d80-ph | `05_nullvec.json:decay/80/phat` | ~0.0001239 | 0.00012388333 | 0.3 dec | PASS |
| NV-d120-p0 | `05_nullvec.json:decay/120/p0` | ~2.655e-07 | 2.654733637e-07 | 0.4 dec | PASS |
| NV-d120-ph | `05_nullvec.json:decay/120/phat` | ~1.186e-07 | 1.186201432e-07 | 0.4 dec | PASS |
| NV-d199-p0 | `05_nullvec.json:decay/199/p0` | ~1.063e-09 | 1.062804942e-09 | 0.5 dec | PASS |
| NV-d199-ph | `05_nullvec.json:decay/199/phat` | ~1.086e-09 | 1.086123606e-09 | 0.5 dec | PASS |

### Obs. 9.6  (6 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| SP-zero | `06_spectrum.json:c089/nu_zero_abs` | <= 1e-07 | 7.254025658e-09 | 1e-07 | PASS |
| SP-mgamma | `06_spectrum.json:c089/nu_minus_gamma` | -0.1 | -0.1000000073 | abs 1e-07 | PASS |
| SP-pair | `06_spectrum.json:c089/pairing_max_err` | <= 1e-07 | 8.105875559e-15 | 1e-07 | PASS |
| SP-nreal | `06_spectrum.json:c089/n_real_nontrivial` | 0 | 0 | abs 0 | PASS |
| SP-circle | `06_spectrum.json:c089/circle_radius` | 0.945369 | 0.9453691666 | abs 1e-06 | PASS |
| CS-random | `07_conformal.json:random_K/relerr` | <= 1e-12 | 1.920685833e-14 | 1e-12 | PASS |

### Table 7  (20 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| GN-fk-f | `08_general.json:rows/fk/f` | 0.48196 | 0.4819599977 | abs 2e-06 | PASS |
| GN-fk-fp | `08_general.json:rows/fk/fprime` | 2.174156 | 2.174155967 | abs 3e-06 | PASS |
| GN-fk-k | `08_general.json:rows/fk/kappa` | 1.146659 | 1.146659361 | abs 3e-06 | PASS |
| GN-fk-e | `08_general.json:rows/fk/relerr` | <= 1e-08 | 1.795478316e-10 | 1e-08 | PASS |
| GN-m05-f | `08_general.json:rows/mu2m05/f` | 0.540326 | 0.5403255333 | abs 2e-06 | PASS |
| GN-m05-fp | `08_general.json:rows/mu2m05/fprime` | 2.676268 | 2.67626817 | abs 3e-06 | PASS |
| GN-m05-k | `08_general.json:rows/mu2m05/kappa` | 1.069733 | 1.069732749 | abs 3e-06 | PASS |
| GN-m05-e | `08_general.json:rows/mu2m05/relerr` | ~3.93e-06 | 3.928122885e-06 | 0.5 dec | PASS |
| GN-nnn-f | `08_general.json:rows/nnn/f` | 0.166333 | 0.1663325617 | abs 2e-06 | PASS |
| GN-nnn-fp | `08_general.json:rows/nnn/fprime` | 0.601746 | 0.6017456529 | abs 3e-06 | PASS |
| GN-nnn-k | `08_general.json:rows/nnn/kappa` | 0.480627 | 0.4806266005 | abs 3e-06 | PASS |
| GN-nnn-e | `08_general.json:rows/nnn/relerr` | <= 1e-10 | 6.929847558e-15 | 1e-10 | PASS |
| GN-mix-f | `08_general.json:rows/mu2nnn/f` | 0.191512 | 0.1915124488 | abs 2e-06 | PASS |
| GN-mix-fp | `08_general.json:rows/mu2nnn/fprime` | 1.009012 | 1.009012031 | abs 3e-06 | PASS |
| GN-mix-k | `08_general.json:rows/mu2nnn/kappa` | 0.735407 | 0.7354070282 | abs 3e-06 | PASS |
| GN-mix-e | `08_general.json:rows/mu2nnn/relerr` | <= 1e-09 | 2.132456271e-11 | 1e-09 | PASS |
| GN-soft-f | `08_general.json:rows/soft/f` | 0.379908 | 0.3799079951 | abs 2e-06 | PASS |
| GN-soft-fp | `08_general.json:rows/soft/fprime` | 1.487688 | 1.487687634 | abs 3e-06 | PASS |
| GN-soft-k | `08_general.json:rows/soft/kappa` | 1.177935 | 1.177935311 | abs 3e-06 | PASS |
| GN-soft-e | `08_general.json:rows/soft/relerr` | <= 1e-10 | 1.504255735e-13 | 1e-10 | PASS |

### Obs. 9.7  (2 claims)

| id | file / key | expected | obtained | tolerance | status |
|---|---|---|---|---|---|
| GN-refine | `08_general.json:refinement/mu2m05_N4096_relerr` | <= 1e-10 | 2.415486684e-12 | 1e-10 | PASS |
| GN-refine-fp | `08_general.json:refinement/mu2m05_N4096_fprime` | 2.6762666 | 2.676266649 | abs 3e-06 | PASS |

