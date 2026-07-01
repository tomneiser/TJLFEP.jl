# widthscan fixture

`input_widthscan_nb2.TGLFEP` is a spectrum input (`PROCESS_IN=3`, `N_BASIS=2`,
`SCAN_N=1`, `IRS=40`) with `WIDTH_IN_FLAG=.false.` and a deliberately narrow width
grid (`WIDTH_MIN=1.50`, `WIDTH_MAX=1.60`, 11 points at the fixed 0.01 step). It is
used by `test/unit_widthscan.jl` to exercise `TJLFEP_ky_widthscan` — the
auto-width branch of the `PROCESS_IN=3` spectrum path (`_mainsub_spectrum`), which
the spectrum regression never hits because it pins a fixed width
(`WIDTH_IN_FLAG=.true.`).

The test calls `TJLFEP_ky_widthscan` directly rather than the full `PROCESS_IN=3`
driver: the 3-mode TM spectrum that follows the scan is already covered by
`runtests_regression_spectrum.jl` and is far too slow to duplicate, whereas the
narrow scan + `find_max` + `out.ky_widthscan` buffer run in a few seconds.

Applied to `examples/DIIID_202017C42_500ms_v3.1/input.gacode` (NR=101); derived
from `input_spectrum.TGLFEP` with `N_BASIS` added (32→2), `IRS` 40, and
`WIDTH_IN_FLAG` flipped to false with a narrow `WIDTH_MIN/WIDTH_MAX` band.
