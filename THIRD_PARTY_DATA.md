# Third-party source data

DGRR-Net was developed using publicly available geospatial source products.

The raw source rasters are not redistributed in this repository or in the
companion supporting-data deposit.

## Source products

The frozen preprocessing workflow uses:

- SRTM: `USGS/SRTMGL1_003`
- USGS 3DEP 1 m DEM: `USGS/3DEP/1m`
- NAIP imagery: `USDA/NAIP/DOQQ`

The Phase-5 workflow selects the most recent available NAIP year for each
study ROI and constructs a yearly NAIP mosaic before export.

## Study regions

The four study regions are:

- Houston Buffalo Bayou
- San Antonio River
- Austin Waller Creek
- Dallas Trinity River

Exact bounding boxes and UTM EPSG identifiers are provided in the companion
supporting-data file `data/roi_manifest.csv`.

## Vertical calibration

For each ROI, one constant SRTM vertical correction is estimated from the
complete co-located SRTM and 3DEP surfaces before patch partitioning.

Accordingly, the reported benchmark should be interpreted as refinement of
an ROI-calibrated coarse DEM rather than as fully reference-independent
deployment.

## Redistribution

No raw SRTM, 3DEP, or NAIP raster is included in this public release.
