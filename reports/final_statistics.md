## Final merged dataset statistics

- **Dataset root**: `/Users/aliaydin/military-dataset/merged_dataset`
- **Classes**: ['tank', 'armored_vehicle', 'civilian_vehicle']

### train
- **Images**: 2743
- **Empty label files**: 0

| class | instances | images_with_class |
|---|---:|---:|
| tank | 1570 | 1328 |
| armored_vehicle | 1530 | 1259 |
| civilian_vehicle | 5641 | 277 |

### val
- **Images**: 783
- **Empty label files**: 0

| class | instances | images_with_class |
|---|---:|---:|
| tank | 445 | 380 |
| armored_vehicle | 459 | 363 |
| civilian_vehicle | 1530 | 79 |

### test
- **Images**: 394
- **Empty label files**: 0

| class | instances | images_with_class |
|---|---:|---:|
| tank | 210 | 188 |
| armored_vehicle | 209 | 180 |
| civilian_vehicle | 829 | 40 |

### TOTAL (all splits)
- **Images**: 3920

| class | instances |
|---|---:|
| tank | 2225 |
| armored_vehicle | 2198 |
| civilian_vehicle | 8000 |

### Pipeline metadata
- **split**: 70/20/10 stratified by dominant class
- **visdrone_civilian_target**: 6000-8000 instances
- **merged_instances**: {1: 2198, 0: 2225, 2: 8000}
