from buteo.earth_observation.s2_mosaic import mosaic_tile, join_s2_tiles
from buteo.earth_observation.s2_utils import (
    get_tile_files_from_safe_zip,
    unzip_files_to_folder,
)

# Folder base
folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/kumasi/"

tmp = folder + "tmp/"  # temporary files, should be deleted afterwards
raw = folder + "raw/"  # location of with S2 images in zipped .safe folders
dst = folder + "dst/"  # destination folder of the cloud free tile

tiles = [
    "30NYM",
    "30NZM",
    "30NYN",
    "30NXN",
]

for tile in tiles:

    unzipped = unzip_files_to_folder(
        get_tile_files_from_safe_zip(raw, tile),
        tmp,
    )

    mosaic_tile(
        tmp,
        tile,
        dst,
        min_improvement=0.1,
        quality_threshold=110,
        time_penalty=90,
        max_time_delta=1500.0,
        max_images=10,
        harmonise=True,
        max_harmony=100,
        ideal_date="20210815",
        process_bands=[
            {"size": "10m", "band": "B02"},
            {"size": "10m", "band": "B03"},
            {"size": "10m", "band": "B04"},
            {"size": "20m", "band": "B05"},
            {"size": "20m", "band": "B06"},
            {"size": "20m", "band": "B07"},
            {"size": "20m", "band": "B8A"},
            {"size": "10m", "band": "B08"},
            {"size": "20m", "band": "B11"},
            {"size": "20m", "band": "B12"},
        ],
    )

# destination of the sentinel-2 mosaic if multiple input tiles are given
# projection is the epsg code (epsg.io) of the output.
mos = folder + "mos/"

join_s2_tiles(dst, mos, tmp, harmonisation=True, projection_to_match=32630)
