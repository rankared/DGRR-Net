# DGRR-Net

DGRR-Net is the public reproducibility package for dual-scale terrain-aware
residual refinement of SRTM elevation using terrain derivatives and NAIP
optical information on a 1 m target grid.

The GitHub repository contains executable methodology. The companion Zenodo
deposit contains the frozen checkpoint and supporting evidence used for the
manuscript.

## Proposed model

The proposed model is `E2cDualScaleResidualUNet`.

- Input channels: 11
- Parameters: 3,807,489
- Base channels: 32
- Normalization: GroupNorm
- Activation: SiLU
- Decoder upsampling: bilinear
- Dropout: 0.05
- Output: normalized residual elevation correction

The eleven model inputs are:

1. globally normalized bias-corrected SRTM
2. patch-local normalized SRTM
3. dz/dx
4. dz/dy
5. Laplacian
6. NAIP red
7. NAIP green
8. NAIP blue
9. NAIP NIR
10. NDWI-like index
11. NDVI-like index

The manuscript global-only ablation is `E2aPlainResidualUNet`.

## Protocol-A split

The frozen supporting-data deposit contains 213 accepted patches:

- 107 training
- 27 validation
- 25 test
- 54 unused spatial-buffer patches

Patch size is 256 x 256 pixels.

Stride is 128 pixels.

Minimum valid ratio is 0.95.

The split is spatially x-blocked within each ROI with buffer columns.

## Repository structure

DGRR-Net/
- configs/
- src/
- scripts/
- README.md
- LICENSE
- THIRD_PARTY_DATA.md
- requirements.txt
- environment_snapshot.json
- source_provenance.json
- SOURCE_SHA256SUMS.txt

## Installation

Python 3.10 or newer is recommended.

Install dependencies:

    python -m pip install -r requirements.txt

Run commands from the repository root.

## Verify the released checkpoint without raw patches

    python -m scripts.verify_release --support-dir /path/to/DGRR-Net_supporting_data

This verifies:

- checkpoint SHA-256
- Protocol-A split counts
- E2c architecture parameter count
- strict checkpoint loading
- selected epoch
- 11-channel forward pass

## Reconstructing study inputs

Raw SRTM, USGS 3DEP, and NAIP rasters are not redistributed.

The public preprocessing implementation contains the frozen Phase-5 multi-ROI
workflow.

Aligned source stacks use the following band order:

1. SRTM
2. USGS 3DEP
3. NAIP red
4. NAIP green
5. NAIP blue
6. NAIP NIR

Patch preparation can be run with:

    python -m scripts.prepare_data         --stack-dir /path/to/aligned_stacks         --output-dir /path/to/patches         --inventory-csv /path/to/patch_inventory.csv         --split-csv /path/to/protocol_a_split.csv

## Verify using reconstructed real patches

    python -m scripts.verify_release         --support-dir /path/to/DGRR-Net_supporting_data         --patch-root /path/to/patches

## Evaluate the released checkpoint

    python -m scripts.evaluate_model         --support-dir /path/to/DGRR-Net_supporting_data         --patch-root /path/to/patches         --split test

The released E2c checkpoint has:

- selected epoch: 63
- SHA-256:
  e85cf834cbe50bb356a4021980f9fe7f642b3966af44471e42a2cc57acc985a8

The release audit reproduced the frozen 25-patch elevation benchmark.

## Reproduce statistical tests

The manuscript paired statistical analysis can be reproduced without the
original raster data:

    python -m scripts.reproduce_statistics         --support-dir /path/to/DGRR-Net_supporting_data

The released per-patch table reproduces all 27 planned E2c comparisons,
including:

- two-sided Wilcoxon signed-rank tests
- Holm-adjusted p-values
- paired rank-biserial effect sizes
- standardized paired effect sizes
- ROI-stratified bootstrap confidence intervals

## Hydrological evaluation

The public implementation includes:

- Priority-Flood depression conditioning
- deterministic D8 routing
- flow accumulation
- sink metrics
- tolerance-based stream F1

Hydrological evaluation is patch-internal. Flow entering from outside each
patch is not represented, so these results should not be interpreted as
full-catchment hydrological simulation.

## Vertical calibration scope

The benchmark uses one constant SRTM vertical correction estimated within
each complete study ROI before spatial partitioning.

The reported results should therefore be interpreted as refinement of an
ROI-calibrated coarse DEM.

Deployment to an unseen region requires an external or
reference-independent vertical-calibration strategy.

## Benchmark evidence

The companion supporting-data deposit contains:

- frozen E2c checkpoint
- Protocol-A split
- study-area metadata
- training normalization statistics
- per-patch test metrics
- paired statistical results
- primary manuscript benchmark
- compact extended comparison

SGNet-DEM is not assigned a frozen-test numerical result because the audited
Phase-9D record documents completed training but no authorized frozen test
evaluation.

## Raw source data

See `THIRD_PARTY_DATA.md`.

Raw SRTM, 3DEP, and NAIP products are not redistributed.

## Source provenance

`source_provenance.json` records the frozen notebook/source lineage from
which the public implementation was recovered.

## License

Repository software is released under the MIT License.

The companion Zenodo supporting-data deposit has a separate CC BY 4.0
license notice.

Third-party source products retain their provider terms.

## Citation

Article citation: pending final bibliographic record.

GitHub repository: https://github.com/rankared/DGRR-Net

Zenodo DOI: 10.5281/zenodo.22942515

DOI URL: https://doi.org/10.5281/zenodo.22942515

The Zenodo DOI is reserved for the companion supporting-data record and
will resolve after that record is published.
