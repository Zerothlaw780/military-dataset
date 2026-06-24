## Final merged dataset statistics

- **Dataset root**: `/Users/aliaydin/military-dataset/merged_dataset`
- **Classes**: ['tank', 'armored_vehicle', 'civilian_vehicle']

### train
- **Images**: 3166
- **Empty label files**: 0

| class | instances | images_with_class |
|---|---:|---:|
| tank | 1570 | 1328 |
| armored_vehicle | 1530 | 1259 |
| civilian_vehicle | 6977 | 700 |

### val
- **Images**: 904
- **Empty label files**: 0

| class | instances | images_with_class |
|---|---:|---:|
| tank | 445 | 380 |
| armored_vehicle | 459 | 363 |
| civilian_vehicle | 1978 | 200 |

### test
- **Images**: 454
- **Empty label files**: 0

| class | instances | images_with_class |
|---|---:|---:|
| tank | 210 | 188 |
| armored_vehicle | 209 | 180 |
| civilian_vehicle | 1184 | 100 |

### TOTAL (all splits)
- **Images**: 4524

| class | instances |
|---|---:|
| tank | 2225 |
| armored_vehicle | 2198 |
| civilian_vehicle | 10139 |

### Pipeline metadata
- **split**: 70/20/10 stratified by dominant class
- **visdrone_civilian_target**: ~1000 images, 10000-12000 instances
- **merged_instances**: {1: 2198, 0: 2225, 2: 10139}
