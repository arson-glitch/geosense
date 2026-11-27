# Responsible AI notes (Maya) — validation set, StreetCLIP linear probe (T = 1.12)

All numbers: val split (7,171 images, 56 countries). Nothing here uses the test set.

## 1. US bias
- The US is 25.0% of val images. Always answering "United States" already scores 25% top-1 (majority baseline).
- The StreetCLIP probe predicts the US for 25.4% of images, so it does **not** over-predict it;
  ResNet-50 predicts it 24.3% of the time. Only 10% of the probe's mistakes land on the US.
- Caveat: the model still *learns from* a US-heavy training set (24% of data), so US photos get the most
  training signal. Report macro-F1 (0.776) next to top-1 (0.885): macro-F1 weights every country equally.

## 2. Weakest countries (recall, val)
Ukraine 0.29 (n=17), Latvia 0.33 (18), Hungary 0.36 (25), Belgium 0.36 (33), Slovakia 0.44 (16),
Bolivia 0.44 (18), Czechia 0.50 (38), Colombia 0.50 (38).
Pattern: small countries with look-alike neighbours (Europe) and few images. By continent, recall is
Europe 0.82 and South America 0.82, versus 0.93–0.95 for North America, Asia and Africa. Per-country table:
`results/val_per_country.csv`. Small n means these recalls are noisy (one image ≈ 3–6 points).

## 3. Africa is under-represented
- Only 5 African countries pass the ≥100-image filter (South Africa, Botswana, Kenya, Nigeria, Ghana):
  254 val images, 3.5% of the data.
- Recall on these 5 is high (0.93, mean error 394 km) — but that is the **easy** version of the task:
  the model only has to pick among 5 African answers. The other ~49 African countries are not in the
  label set at all, so any photo from them is forced onto one of the 56 kept countries.
- Do not claim "works well in Africa". Claim "works on 5 African countries".

## 4. Countries never seen in training
- 124 countries in the raw data, 56 kept; 1,996 images (≈4%) from the 68 dropped countries were removed.
- The model cannot answer "other" at country level. Its confidence is still informative:
  mean top-1 confidence is 0.92 when right and 0.60 when wrong, so low confidence is a usable
  "I am not sure / maybe unseen country" signal. The app's top-5 + "Other" line should keep that mass visible
  and never show 100%.

## 5. Privacy and licence
- Output is country level only; no coordinates, addresses or people are predicted or stored.
- Street View frames can still show faces, number plates and houses: we never redistribute images,
  and the repo contains only file names and country labels.
- Images: Street View, academic non-commercial use only. StreetCLIP: CC-BY-NC. The business case cannot
  assume commercial deployment without new licences.
- Misuse risk: a stronger geolocator could help locate someone from a private photo. Country level
  limits the harm, but it should be stated in the report.

## 6. Explanations are not proof
Grad-CAM / attention rollout (`geosense/explain.py`) show where the model looked, not why it was right.
Check a sample for shortcuts (e.g. the car's own roof or the Google watermark) before trusting a map.
