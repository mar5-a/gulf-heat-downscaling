# Plan for the 2025 test

Written on 25 November 2026, after the code changes below and before any 2025 score was
computed. The 2025 results are reported as they come out, whichever way they go. After the run,
only bug fixes are allowed, and any fix will be listed in the README.

## Why a new test year

The first test (trained on 2023, scored on 2024) is kept in `reports/test_2024/`. After it, I
studied the 2024 results and found two problems: Bahrain airport got much worse, and the street
map followed the height contours of the Dhahran dome. Fixing them and scoring again on 2024 would
turn 2024 into a tuning year. So 2024 becomes training data and 2025 is the test.

## Data

- Training: 2023 and 2024. Test: 2025.
- Same 80 stations and the same checks as before (station years with under 60% of hours dropped;
  positions on water moved to land; four stations dropped for bad positions).
- 67 stations have enough 2025 readings: 541,905 station-hours. All 67 have training rows.
- Only stations with training rows are scored, so "seen" means seen. (In the first test, 6 of the
  78 scored stations had no 2023 rows, so their "seen" scores were really unseen.)

## What changed in the design (frozen)

1. Absolute terrain height is no longer a feature. The first design used it as a stand-in for
   "inland" and drew the dome's contours on the map.
2. Every LightGBM model in the design carries a monotone constraint: being higher than the ERA5
   cell can only lower the prediction, at every hour. Across 2023 and 2024 stations, ERA5's error
   falls by about 0.4 °C per 100 m of height above the cell, by day and by night alike.
3. A new feature, `era5_sea_share`: how much of the site's ERA5 value comes from grid points
   ERA5 treats as sea (cell height exactly 0 m), blended with the same bilinear weights as the
   values. It is 1 at Bahrain airport. A first version, built from land cover water within each
   ERA5 cell, correlated 0.97 with the existing `sea_frac_25km`, so it was replaced.
4. Feature sets: hourly; plus ERA5 cell (height above the cell, sea share); plus static sea; plus
   land cover.
5. The map model is chosen by station-grouped cross-validation on 2023 and 2024, among models
   that cannot make higher ground warmer (baselines and the constrained LightGBM models).

The first design's map model (`lightgbm_first_design`: hourly features, absolute height and
height above the cell, no constraint) is retrained and tuned the same way as a reference.

## Questions and how the answers will be read

1. **Does the correction still help at stations it never saw?** Map model against raw ERA5,
   unseen-station mean absolute error in 2025, for all stations and the Gulf box.
2. **Did the fixes cost accuracy?** Map model against `lightgbm_first_design`, same scores. The
   height fix is mainly for the map, which stations cannot check, so a small loss here is
   acceptable. I will report the difference either way.
3. **Bahrain.** Unseen-station error at Bahrain airport in 2025 for raw ERA5, the first design
   and the map model. The fix counts as working if the map model is no worse than raw ERA5 there.
4. **The map.** On land cells at least 3 km from the sea, mean correction above 25 m minus mean
   correction below 10 m, at 2 pm local on 19 June 2024, for both designs
   (`reports/tables/map_by_height.csv`). The fix counts as working if that gap is much smaller
   than the first design's. Other features that vary with height can still leave a gap, so it
   need not be zero. This check uses no 2025 data.
