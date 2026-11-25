# Diagnosis on the first 27 stations (24 November 2026)

Mean absolute error in °C on held-out stations within 2023 (5 station groups, GroupKFold).
Table built from the first 27 stations, square-average land cover features, positions as listed.

| Model | MAE |
|---|---|
| Raw ERA5 | 1.349 |
| Average hourly bias | 1.315 |
| Lapse-rate height correction | 1.360 |
| LightGBM, all features | 1.518 |
| LightGBM, weather and time only | 1.304 |
| LightGBM, small set (weather, time, height gap, sea shares, upwind sea, built share) | 1.568 |
| LightGBM, all features, strong regularisation | 1.452 |

LightGBM settings: num_leaves 15, learning_rate 0.03, min_child_samples 50, 400 trees (strong
regularisation: num_leaves 7, 300 trees, min_child_samples 2000, colsample_bytree 0.5).
Conclusion at the time: with 27 stations, fixed place features act as station labels, so the
station set was expanded.
