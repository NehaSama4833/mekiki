# mekiki — 目利き

**目利き (mekiki)** means "a discerning eye, the ability to judge quality". After deployment you usually have no labels. Can you still estimate your model's accuracy when the inputs shift? This is a small study of confidence-based estimators (average confidence, difference of confidences, ATC) on real digit images under 5 kinds of image corruption plus label shift. The methods come from the literature; this is a hands-on check, not a new method.

## Setup
Digits (8x8): 900 train, 300 source-validation, 597 target images, 3 random splits, 3 models (logistic regression, MLP, random forest). Estimators use only source validation data and the model's outputs on unlabeled target images. Shifts: Gaussian noise, brightness, pixel dropout, blur, rotation (several severities each) and label shift (90% digits 0-4).

## Results (`results/report.md`): mean absolute error of the accuracy estimate, in accuracy points
| shift | true accuracy range | assume no shift | avg confidence | ATC (max conf) |
|---|---|---|---|---|
| no shift | 96-97% | 0.6 | 8.7 | 1.2 |
| gaussian noise | 13-97% | 34.1 | 39.7 | 28.6 |
| brightness | 36-96% | 8.1 | 15.2 | 8.5 |
| pixel dropout | 45-95% | 21.6 | 18.3 | 9.2 |
| **blur** | 10-96% | 46.9 | 41.1 | **36.9** |
| **rotation** | 10-87% | 49.6 | 39.7 | **32.5** |
| label shift | 94-99% | 1.4 | 8.3 | 1.2 |
| all pooled | | 26.2 | 27.9 | 18.8 |

- Confidence-based estimates help on dropout, brightness and label shift, but **fail badly on blur and rotation**: the models stay confident while accuracy collapses. The worst misses estimate about 100% when the truth is 10-13% (MLP and logistic regression on strong blur).
- Gaussian noise is also poorly estimated (28.6 points even for ATC), so these estimators are not reliable monitors by themselves.
- Per model, ATC's error was 17.2 (logistic regression), 24.9 (MLP) and 14.5 (random forest) points; for the MLP the estimates barely rank conditions (Spearman 0.29), for the random forest they rank them well (0.95) while the absolute error is still large.
- On "no shift", average confidence is 8.7 points off because the models are over- or under-confident; the ATC threshold fixes this by calibrating on source validation data.

## Caveats
Tiny images and small models; shifts are synthetic and applied to the whole target set; 3 splits only; I did not implement domain-classifier or ensemble-disagreement estimators; estimator errors vary much more by model than the method names suggest, so do not read the pooled row as a ranking of methods.

## Run
```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]' scipy
.venv/bin/python -m pytest -q            # 5 tests
.venv/bin/python -m mekiki.experiment    # ~1 minute
```
