# WNBA RAPM + 4-Factor RAPM

Reproducible ridge-regression RAPM and four-factor decomposition for the WNBA, built from possession-level stint data.

## Quickstart

```bash
git clone https://github.com/shankapotomus/wnba-rapm.git
cd wnba-rapm
pip install numpy pandas scipy scikit-learn jupyter
jupyter notebook rapm_reproducible.ipynb
```

Run all cells. That's it — data is included.

## What's in the notebook

### Standard RAPM
Ridge regression on possession-level stints. Each possession is a row; players on offense get +1, players on defense get +1 in separate regressions predicting points scored/allowed. Coefficients are scaled to per-100-possession impact.

| Output column | Meaning |
|---|---|
| `orapm` | Offensive RAPM |
| `drapm` | Defensive RAPM (positive = good defense) |
| `net_rapm` | orapm + drapm |

### 4-Factor RAPM
Decomposes each player's RAPM into eight on-court rate components using OLS regression against demeaned four-factor rates from `stints_rich`.

| Factor | Side | Meaning |
|---|---|---|
| `ots` | Offense | True shooting rate vs league avg |
| `otov` | Offense | Turnover rate vs league avg |
| `oreb` | Offense | Offensive rebound rate vs league avg |
| `otrans` | Offense | Transition possession rate vs league avg |
| `dts` | Defense | Opponent TS% held below league avg |
| `dtov` | Defense | Opponent turnover rate forced above league avg |
| `dreb` | Defense | Opponent offensive rebound rate held below league avg |
| `dtrans` | Defense | Opponent transition rate held below league avg |

For all factors, **positive = good for the player**.

The OLS betas tell you how much each factor is worth in RAPM points (e.g. a player 3% above league avg in TS% on offense ≈ +3.5 RAPM points from that factor alone).

## Configuration

All tunable parameters are in one cell near the top of the notebook:

```python
DATA_DIR     = Path("wnba_data")   # path to data folder
RAPM_SEASONS = [2023, 2024, 2025]  # seasons to pool (more = more stable)
LAMBDA       = 2000                # ridge penalty (higher = more shrinkage)
MIN_POSS     = 200                 # minimum possessions to appear in output
```

## Data

| Folder / File | Contents |
|---|---|
| `wnba_data/stints/` | Possession-level stints 2017–2026 (used for RAPM) |
| `wnba_data/stints_rich/` | Stints with per-possession four-factor flags 2021–2026 |
| `wnba_data/player_names.csv` | Player ID → name lookup |
| `wnba_data/rapm_*.csv` | Pre-computed RAPM outputs by season |
| `wnba_data/rapm_8factor_*.csv` | Pre-computed 4F RAPM outputs |

Raw PBP JSON files (~400 MB) are excluded from the repo.

## Output

The notebook saves results to `wnba_data/rapm_and_4f_output.csv` — one row per qualifying player with RAPM, reconstructed RAPM, residual, and all eight factor rates.
