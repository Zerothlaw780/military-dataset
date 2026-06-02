## Train / Val / Test leakage report

- **Dataset**: `/Users/aliaydin/military-dataset/merged_dataset`
- **Splits checked**: train, val, test
- **Near-duplicate threshold**: Hamming distance ≤ 8 (pHash)

### Image counts per split

| split | images |
|---|---:|
| train | 2743 |
| val | 783 |
| test | 394 |
| **total** | 3920 |

### 1. Exact duplicates (SHA-1 byte hash)

- **Duplicate groups (same bytes, any split)**: 0 (0 files in groups)
- **Cross-split exact duplicate groups (leakage)**: **0**

No exact duplicate images were found across different splits.

### 2. Near-duplicates (perceptual hash)

- **Cross-split near-duplicate clusters**: **14**
- **Files involved**: 32

#### Clusters spanning multiple splits

**Cluster 1** — 5 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_115_png.rf.49be685db5f3abdda5792648d2adeb87_002260.jpg` | `80c0743d3bb59ed5` |
| train | `ab_171_png.rf.2258e12e8733defd02f2fb2fba2e7927_002583.jpg` | `80c074372b9d9e77` |
| train | `ab_bandicam-2023-10-16-16-58-05-099_jpg.rf.ee26fbf5512142023aff85dccc2c0f69_002654.jpg` | `80c0745f3a9d8fd5` |
| val | `ab_124_png.rf.170af4399079ad2b6b2068db3d2f6d9c_002452.jpg` | `c0c0701d3f3f9ad5` |
| val | `ab_161_png.rf.53f92461f6b0d7c934802a3bb330fb00_000816.jpg` | `80c874b52b379ed9` |

**Cluster 2** — 3 images, splits: test, train

| split | filename | phash |
|---|---|---|
| test | `ab_bandicam-2023-10-17-20-03-31-571_jpg.rf.40ed8e492e873cc045f748fd4450307d_002726.jpg` | `84c261353a3f8ecf` |
| train | `ab_IMG_20231019_173444_657_0036-1_png.rf.3f28163a0e5379d1a3497c011e3db755_001055.jpg` | `85ca60353a9f8e5d` |
| train | `ab_bandicam-2023-10-17-20-03-31-571_jpg.rf.27b4c853a96b48c149d399cd8b573e67_002837.jpg` | `84c061353a3f9e5f` |

**Cluster 3** — 2 images, splits: test, train

| split | filename | phash |
|---|---|---|
| test | `ab_15_png.rf.570fa8ad7d049aa6d5f2597886e3619d_002714.jpg` | `844878b72e9dc877` |
| train | `ab_117_png.rf.5b7ba1ea034056be4238299e743bc98d_002409.jpg` | `c44839a72e9dce35` |

**Cluster 4** — 2 images, splits: test, train

| split | filename | phash |
|---|---|---|
| test | `ab_71_png.rf.61bcb0c31feb44337e119bbea8bc0cd4_002078.jpg` | `9b9c6463269a9b99` |
| train | `ab_139_png.rf.d52679e602ecdd1f29439531469dbfc7_002724.jpg` | `9b9e64616792929b` |

**Cluster 5** — 2 images, splits: test, train

| split | filename | phash |
|---|---|---|
| test | `ab_IMG_20231019_173813_779_0038-0_png.rf.edf24512573b60bc3cb1f750d478b965_001485.jpg` | `c1c63e3861c3cf9c` |
| train | `ab_IMG_20231019_174948_303_0027-0_png.rf.6f82f37e29b3d8cae4e44c69bf9869d8_002351.jpg` | `c0873e3c6163c79e` |

**Cluster 6** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_130_png.rf.492eca1e5df37efe564f26c72ae113a8_002427.jpg` | `c5b53a48ef62c09e` |
| val | `ab_bandicam-2023-10-16-17-19-13-838_jpg.rf.d0c61ff79b19e2e8333ae814f6c58589_001330.jpg` | `d5b52a486f6290dd` |

**Cluster 7** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_35_png.rf.52c923d76a579aa91ecc3bcdc5837f7f_001957.jpg` | `cc3233cd6639c999` |
| val | `ab_162_png.rf.082abe4aab76d320d676ad4f3664ebaf_001088.jpg` | `ce3331cc66339999` |

**Cluster 8** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_3_png.rf.5ac8aef290803ce77addae8e5072f97c_000243.jpg` | `c4b73b486ee2cb88` |
| val | `ab_4_png.rf.46f959cfdab12d8f70ee7d2f9d9e5664_000667.jpg` | `c4b73a486ec2db86` |

**Cluster 9** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_44_png.rf.a7127e75e6780444ca0232b55f80c9d8_002884.jpg` | `c0953f68eac095bd` |
| val | `ab_bandicam-2023-10-16-15-40-08-110_jpg.rf.15c099c092b2da55025be0c019b2dd9f_001790.jpg` | `d1953d6a6ac2859d` |

**Cluster 10** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_bandicam-2023-10-16-15-33-18-599_jpg.rf.29feeb09793ebd641d6057f53e01a316_000716.jpg` | `86b6795971c58633` |
| val | `ab_bandicam-2023-10-16-15-33-10-781_jpg.rf.2e623c0509979c61aab4e9d9e7bb925b_002229.jpg` | `86b6795971c4c633` |

**Cluster 11** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_bandicam-2023-10-16-15-33-18-599_jpg.rf.a3bb48f1ad2ad26b023854cc707ab60b_000605.jpg` | `d3e2681d3f99c066` |
| val | `ab_bandicam-2023-10-16-15-33-12-567_jpg.rf.6955a2702155284d177756b126da8ffc_001934.jpg` | `d3e26c1d2791c366` |

**Cluster 12** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_bandicam-2023-10-16-15-40-08-110_jpg.rf.209078b706a7000fcf3f9154e892590a_001854.jpg` | `8080697f3ff5d0d8` |
| val | `ab_bandicam-2023-10-16-15-40-08-110_jpg.rf.38f0a995bac280cdf3ea7f757ad3bfea_002566.jpg` | `84c0693f3fb7d0c8` |

**Cluster 13** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_bandicam-2023-10-16-16-57-38-251_jpg.rf.c0d5b683f7a691f39e66f6112e1b2241_002587.jpg` | `d49537686a6acaca` |
| val | `ab_159_png.rf.dc3443bd5e507e23122ff03b2aa9a56d_002893.jpg` | `d4b5334a6e6aca62` |

**Cluster 14** — 2 images, splits: train, val

| split | filename | phash |
|---|---|---|
| train | `ab_bandicam-2023-10-17-17-38-05-743_jpg.rf.5b708ba2c30cfee1af98d5699fe0f4a2_002367.jpg` | `94686b953b3fc688` |
| val | `ab_bandicam-2023-10-17-17-38-03-183_jpg.rf.2d729616306a370a9061018b2a801ff7_000929.jpg` | `d46869973e3fc608` |

### 3. Summary

- **14** near-duplicate cluster(s) link visually similar images across splits.
- Review the groups above; consider removing duplicates from val/test or re-splitting so related frames stay in one split.
