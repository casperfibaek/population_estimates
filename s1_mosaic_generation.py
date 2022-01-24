from buteo.earth_observation.s1_mosaic import mosaic_sentinel1

folder = "path_to_db_processed_s1_imagery" # use buteo.earth_observation.s1_preprocess

mosaic_sentinel1(
    folder + "S1/august/",
    folder + "S1/august/dst/",
    folder + "S1/august/tmp/",
    interest_area=folder + "vector/mwanza_extent.gpkg",
    target_projection=32736,
    kernel_size=3,
    overlap=0.00,
    step_size=1.0,
    quantile=0.5,
    max_images=0,
    weighted=True,
    overwrite=False,
    use_tiles=False,
    high_memory=True,
    polarization="VV",
    prefix="",
    postfix="",
)

mosaic_sentinel1(
    folder + "S1/august/",
    folder + "S1/august/dst/",
    folder + "S1/august/tmp/",
    interest_area=folder + "vector/mwanza_extent.gpkg",
    target_projection=32736,
    kernel_size=3,
    overlap=0.00,
    step_size=1.0,
    quantile=0.5,
    max_images=0,
    weighted=True,
    overwrite=False,
    use_tiles=False,
    high_memory=True,
    polarization="VH",
    prefix="",
    postfix="",
)
