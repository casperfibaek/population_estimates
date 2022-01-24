from glob import glob
from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster


folder = "path_to_unaligned_raster_folder"
aligned = folder + "aligned/"

target = folder + "august_B12_20m.jp2"

align_rasters(
    clip_raster(
        glob(folder + "*10m.*"),
        clip_geom=target,
        postfix="",
        all_touch=False,
        adjust_bbox=False,
        dst_nodata=0,
    ),
    aligned,
    postfix="",
    dst_nodata=False,
    ram="80%",
    bounding_box=target,
)

align_rasters(
    clip_raster(
        glob(folder + "*20m.*"),
        clip_geom=target,
        postfix="",
        all_touch=False,
        adjust_bbox=False,
    ),
    aligned,
    postfix="",
    ram="80%",
    dst_nodata=False,
    bounding_box=target,
)
