# mekiki results

Digits (8x8): 900 train, 300 source-validation, 597 target images; 3 random splits; 3 models; shifts applied to the target images only. Estimators use only source validation data plus unlabeled target predictions.

## Mean absolute error of the accuracy estimate (accuracy points), by shift family (all models, severities, repeats pooled)

| shift family | true accuracy range | source accuracy (assume no shift) | average confidence | difference of confidences | ATC (max confidence) | ATC (negative entropy) |
|---|---|---|---|---|---|---|
| no shift | 96%-97% | 0.6 | 8.7 | 0.7 | 1.2 | 1.7 |
| gaussian noise | 13%-97% | 34.1 | 39.7 | 32.2 | 28.6 | 33.0 |
| brightness | 36%-96% | 8.1 | 15.2 | 7.5 | 8.5 | 13.1 |
| pixel dropout | 45%-95% | 21.6 | 18.3 | 10.9 | 9.2 | 12.2 |
| blur | 10%-96% | 46.9 | 41.1 | 35.4 | 36.9 | 40.1 |
| rotation | 10%-87% | 49.6 | 39.7 | 36.4 | 32.5 | 37.0 |
| label shift (90% digits 0-4) | 94%-99% | 1.4 | 8.3 | 1.5 | 1.2 | 1.4 |
| **all shifts pooled** | | **26.2** | **27.9** | **20.8** | **18.8** | **22.2** |

## Per model, mean absolute error over all conditions (accuracy points)

| model | source accuracy (assume no shift) | average confidence | difference of confidences | ATC (max confidence) | ATC (negative entropy) |
|---|---|---|---|---|---|
| logistic regression | 26.9 | 19.1 | 19.8 | 17.2 | 17.2 |
| MLP | 38.5 | 29.7 | 29.7 | 24.9 | 26.6 |
| random forest | 13.1 | 34.8 | 12.9 | 14.5 | 22.9 |

## Do the estimates at least rank conditions correctly? (Spearman between estimated and true accuracy, per model)

| model | average confidence | difference of confidences | ATC (max confidence) | ATC (negative entropy) |
|---|---|---|---|---|
| logistic regression | 0.59 | 0.61 | 0.61 | 0.53 |
| MLP | 0.27 | 0.27 | 0.29 | 0.14 |
| random forest | 0.95 | 0.95 | 0.95 | 0.94 |

## Worst single misses (largest |estimate - truth|)

| estimator | model | shift | true acc | estimated |
|---|---|---|---|---|
| ATC (negative entropy) | MLP | blur 1.5 | 10% | 100% |
| ATC (max confidence) | MLP | blur 1.5 | 10% | 100% |
| average confidence | MLP | blur 1.5 | 10% | 99% |
| difference of confidences | MLP | blur 1.5 | 10% | 99% |
| ATC (negative entropy) | MLP | blur 1.5 | 13% | 100% |
| ATC (negative entropy) | logistic regression | blur 1.5 | 11% | 98% |