# scan2 fixture

`input_scan2_nb2.TGLFEP` is a deliberately tiny threshold scan (`PROCESS_IN=5`,
`SCAN_N=2`, `N_BASIS=2`, `IRS=2`) used by `test/unit_scan_finalize.jl`. It is the
standalone-file analogue of FUSE's `ActorTJLFEP` smoke test (SCAN_N=2 / N_BASIS=2
ITER scan): running it through the gacode file path exercises the multi-radius
scan orchestration, `finalize_gacode_scan`, and the α critical-gradient
post-processing in TJLFEP's own CI.

It is applied to `examples/DIIID_202017C42_500ms_v3.1/input.gacode` (NR=101), so
`ir_exp_from_scan(101, 2, 2) = [2, 101]` — the last scan point lands on the
separatrix `ir=NR` (rho~1), where the TGLF Hermite matrix can be singular and the
solve must degrade gracefully to a finite SFmin instead of erroring.

Derived from `examples/DIIID_202017C42_500ms_v3.1/input_scan20_nb6.TGLFEP` with
`SCAN_N` 20→2 and `N_BASIS` 6→2 for speed.
