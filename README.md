# Invasive Species Spread Simulator: Monk Parakeet (*Myiopsitta monachus*)

A grid-based cellular automata simulation modeling the spatial spread of the invasive Monk Parakeet across Florida, seeded with real occurrence data from the Global Biodiversity Information Facility (GBIF).

## Overview

The Monk Parakeet is one of the most successful parrot invaders worldwide. This simulator recreates and analyzes the species' colonization of Florida from 1969 to 2025 using a computational ecology approach.

The model incorporates:

- **Real GBIF occurrence records** (45 verified records across 18 Florida counties) as initial colonization points
- **Habitat suitability modeling** based on urban density and proximity to infrastructure
- **Logistic population growth** with carrying capacity per grid cell (100 individuals/cell)
- **Stochastic dispersal** with local 8-neighbor diffusion and long-distance jump events
- **Seasonal reproduction dynamics** (spring breeding, 4-8 eggs/clutch)

## Simulation Parameters

| Parameter | Value |
|---|---|
| Time span | 1969 - 2025 |
| Grid resolution | 0.1° (~11 km per cell) |
| Carrying capacity | 100 individuals/cell |
| Intrinsic growth rate | 0.25/year |
| Local dispersal probability | 0.15 |
| Long-distance jump probability | 0.02 |
| Growth model | Logistic with habitat-weighted K |

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python monk_parakeet_invasion_sim.py
```

The script generates two output figures:

1. **fig1_spread_snapshots.png** - Six-panel map showing population spread at years 1970, 1980, 1990, 2000, 2010, and 2025
2. **fig2_analysis.png** - Four-panel analysis figure:
   - Habitat suitability index map
   - Population growth and range expansion over time
   - Year of first colonization per cell
   - Temporal distribution of GBIF records

## Project Structure

```
├── monk_parakeet_invasion_sim.py   # Main simulation script
├── requirements.txt                # Python dependencies
└── README.md
```

## Data Source

GBIF.org (2025) GBIF Occurrence Download
- Species: *Myiopsitta monachus* (Boddaert, 1783)
- Filtered: Florida, USA | Coordinate-verified records
- License: CC BY 4.0

## References

- Avery, M.L., et al. (2012). Biology of invasive monk parakeets in south Florida. *Wilson Journal of Ornithology* 124:581-588.
- Van Bael, S. & Pruett-Jones, S. (1996). Exponential population growth of Monk Parakeets in the United States. *Wilson Bulletin* 108:584-588.
- Forcey, G., et al. (2009). Spatial modeling monk parakeet distributions on electric utility structures in Broward and Dade Counties, Florida.
- Moulton, M.P. & Pimm, S.L. (1986). Species introductions to Hawaii. *Ecology of Biological Invasions of North America and Hawaii*, pp. 231-249.
