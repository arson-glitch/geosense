# EDA summary – GeoSense

- **Data:** 49,997 raw images, 124 countries, all 1536x662 JPEG (7.2 GB).
  After cleaning: 47,990 images, 56 countries (>= 100 images each).
- **Imbalance:** USA = 12,014 images (25%); smallest = 106.
  Always guessing USA gives 25% accuracy -> report macro-F1, use class weights / weighted sampling.
- **Shape:** 2.3:1 panoramas. Centre-crop to square drops 57% of the width; recommend squash (or 224x448).
- **Interface overlay:** score box, buttons and minimap sit in fixed positions -> fixed mask is valid; navigation arrows move -> cannot be masked.
- **Colour:** differs by country but overlaps strongly; useful baseline feature, not sufficient alone.
- **Shortcut check (validation set, chance macro-F1 = 1/56 = 0.018):** file size alone is at chance (no technical fingerprint); colour is ~2x chance (weak real signal).

| features           |   accuracy |   macro-F1 |
|:-------------------|-----------:|-----------:|
| majority class     |      0.251 |      0.007 |
| file size only     |      0.006 |      0.007 |
| colour stats only  |      0.033 |      0.03  |
| file size + colour |      0.041 |      0.038 |

- **Limitation:** duplicate detection only catches identical screenshots; same place / other heading can still cross splits.
