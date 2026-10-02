| run | total params | active params | final val loss | vs A | jump at switch | steps to recover | dead experts (final / peak) | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A  dense, keeps training | 10,823,424 | 10,823,424 | 0.5511 | - | - | - | - | 157,657 |
| B  MoE, drop-upcycled | 33,900,288 | 10,897,152 | 0.5551 | +0.0040 | +1.8189 | 1200 | 0 / 42 | 40,117 |
| D  MoE, cloned slices | 33,900,288 | 10,897,152 | 0.5548 | +0.0037 | +0.5274 | 1100 | 0 / 69 | 40,372 |
| C  MoE from scratch | 33,900,288 | 10,897,152 | 0.5281 | -0.0230 | - | - | 0 / 1 | 38,479 |
