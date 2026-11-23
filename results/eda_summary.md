# EDA summary – GeoSense

- **Data:** 49,997 raw images, 124 countries, JPEG (7.2 GB); 49,942 are 1536x662, 55 are 1030x703.
  After cleaning: 47,807 images, 56 countries (>= 100 images each).
- **Imbalance:** USA = 11,973 images (25%); smallest = 106.
  Always guessing USA gives 25% accuracy -> report macro-F1, use class weights / weighted sampling.
- **Shape:** after the UI crop images are 1.79:1; the team rule (short side + square crop) keeps 56% of the width per crop. Suggest testing squash vs crop on val before the big runs.
- **Continents:** Europe 17,538, North America 14,253, Asia 8,136, South America 3,946, Oceania 2,254, Africa 1,680. Africa has 5 countries (~3.5%).
- **UI crop:** 5% -> 82% removes buttons, score box and mini-map; applied in pixels (77 / 276 px) so it also works on the 1030x703 images. Navigation arrows stay.
- **Colour:** differs by country but overlaps strongly; useful baseline feature, not sufficient alone.
- **Shortcut check (validation set, chance macro-F1 = 1/56 = 0.018):** file size alone is at chance (no technical fingerprint); colour is ~2x chance (weak real signal).

| features           |   accuracy |   macro-F1 |
|:-------------------|-----------:|-----------:|
| majority class     |      0.25  |      0.007 |
| file size only     |      0.013 |      0.011 |
| colour stats only  |      0.051 |      0.036 |
| file size + colour |      0.06  |      0.042 |

- **Limitation:** duplicate detection only catches identical screenshots; same place / other heading can still cross splits.
