| run | total params | active params | final val loss | vs A | jump at switch | steps to recover | dead experts (final / peak) | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A  dense, keeps training | 837,888 | 837,888 | 1.0440 | - | - | - | - | 26,317 |
| B  MoE, drop-upcycled | 2,558,208 | 854,272 | 1.0040 | -0.0401 | +0.5567 | 200 | 0 / 7 | 8,953 |
| D  MoE, cloned slices | 2,558,208 | 854,272 | 1.0054 | -0.0387 | +0.2158 | 200 | 3 / 7 | 8,605 |
| C  MoE from scratch | 2,558,208 | 854,272 | 0.9729 | -0.0712 | - | - | 0 / 6 | 8,769 |
