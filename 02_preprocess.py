import numpy as np

from buteo.machine_learning.ml_utils import preprocess_optical, preprocess_sar


def preprocess(
    folder,
    outdir,
    low=0,
    high=1,
    optical_top=10000,
):
    b02 = folder + "B02_10m.npy"
    b03 = folder + "B03_10m.npy"
    b04 = folder + "B04_10m.npy"
    b08 = folder + "B08_10m.npy"

    b05 = folder + "B05_20m.npy"
    b06 = folder + "B06_20m.npy"
    b07 = folder + "B07_20m.npy"
    b11 = folder + "B11_20m.npy"
    b12 = folder + "B12_20m.npy"

    vv = folder + "VV_10m.npy"
    vh = folder + "VH_10m.npy"

    target = "volume"

    label_area = folder + f"label_{target}_10m.npy"

    area = np.load(label_area)
    shuffle_mask = np.random.permutation(area.shape[0])

    np.save(outdir + f"label_{target}", area[shuffle_mask])

    rgbn = preprocess_optical(
        np.stack(
            [
                np.load(b02),
                np.load(b03),
                np.load(b04),
                np.load(b08),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(outdir + "RGBN.npy", rgbn[shuffle_mask])

    reswir = preprocess_optical(
        np.stack(
            [
                np.load(b05),
                np.load(b06),
                np.load(b07),
                np.load(b11),
                np.load(b12),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(outdir + "RESWIR.npy", reswir[shuffle_mask])

    sar = preprocess_sar(
        np.stack(
            [
                np.load(vv),
                np.load(vh),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
    )

    np.save(outdir + "SAR.npy", sar[shuffle_mask])

if __name__ == "__main__":

    base_folder = "path_to_patches"
    folder = base_folder + "raw/"
    outdir = base_folder

    preprocess(folder, outdir)
