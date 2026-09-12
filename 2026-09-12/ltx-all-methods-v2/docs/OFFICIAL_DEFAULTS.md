# Official-default cadence mapped to LTX

This is a declared LTX mapping, not an official LTX implementation. Wan defaults use 6 warmup steps, fixed refresh interval 6 and a 28/40 cached-layer fraction. We map that fraction to 20/28 LTX layers using the frozen LTX profiling rank. The last step is not forced full. No speed or quality tuning is applied.

| Case | Native seconds | Mapped seconds | Speedup | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| holdout_01 | 95.950 | 50.275 | 1.909 | 0.91845 | 0.07003 |
| holdout_02 | 96.470 | 51.454 | 1.875 | 0.86858 | 0.07672 |
| holdout_03 | 93.571 | 48.129 | 1.944 | 0.92825 | 0.08966 |

Three-case ratio of total online times: 1.908x. Mean mapped latency: 49.953 seconds.

This supplemental diagnostic is separate from the main speed-selected tier and is not added to the default 11-method VBench plan. Official paper WAN 2.01x is not an LTX timing claim.
