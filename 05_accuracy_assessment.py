import os
import numpy as np
from glob import glob
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.resample import resample_raster
from buteo.raster.resample import resample_raster
from buteo.raster.clip import clip_raster


def create_ground_truth(buildings, b4, output, size=0.2):
    cm40 = rasterize_vector(buildings, size, extent=b4)
    resampled = resample_raster(cm40, b4, resample_alg="average", dtype="float32")
    return array_to_raster(
        raster_to_array(resampled) * 100, reference=b4, out_path=output
    )


def print_result(number, name, pred, truth, zfill=6):
    pred = pred.flatten()
    truth = truth.flatten()

    _mae = str(round(mean_absolute_error(truth, pred), 3)).zfill(zfill)
    _rmse = str(round(mean_squared_error(truth, pred, squared=False), 3)).zfill(zfill)
    _exvar = str(round(explained_variance_score(truth, pred), 3)).zfill(zfill)
    tpe = str(np.round((pred.sum() / truth.sum()) * 100, 3)).zfill(zfill)

    print(
        f"{number} - {name}. MAE: {_mae} - RMSE: {_rmse} - Exp.var: {_exvar} - TPE: {tpe}"
    )


def print_result_bin(name, _truth, _pred, zfill=6):
    bacc = str(round(balanced_accuracy_score(_truth, _pred), 3)).zfill(zfill)
    acc = str(round(accuracy_score(_truth, _pred), 3)).zfill(zfill)
    f1 = str(round(f1_score(_truth, _pred), 3)).zfill(zfill)
    precision = str(round(precision_score(_truth, _pred, zero_division=True), 3)).zfill(
        zfill
    )
    recall = str(round(recall_score(_truth, _pred, zero_division=True), 3)).zfill(zfill)
    tpe = str(np.round((_pred.sum() / _truth.sum()) * 100, 3)).zfill(zfill)
    print(
        f"{name} Binary || Bal. Accuracy: {bacc} - Accuracy: {acc} - F1: {f1} - Precision: {precision} - Recall: {recall} - TPE: {tpe}"
    )


def main(folder):
    truth_arr = []
    pred_arr = []
    osm_arr = []
    google_90_arr = []
    google_85_arr = []
    google_80_arr = []
    google_00_arr = []
    ghls2_arr = []
    ecw_arr = []

    for area in sorted(glob(folder + "ground_truth_*.tif")):
        name = os.path.splitext(os.path.basename(area))[0]
        number = name.split("_")[-1]

        # if number != "4":
        #     continue

        mask = np.zeros_like(raster_to_array(area))

        truth = np.ma.array(raster_to_array(area), mask=mask)
        truth = truth.filled(0)
        truth_arr.append(truth.flatten())

        pred = np.ma.array(raster_to_array(f"{folder}pred_v23_{number}.tif"), mask=mask)
        pred = pred.filled(0)
        pred_arr.append(pred.flatten())

        osm = np.ma.array(raster_to_array(f"{folder}OSM_{number}.tif"), mask=mask)
        osm = osm.filled(0)
        osm_arr.append(osm.flatten())

        google_90 = np.ma.array(
            raster_to_array(f"{folder}Google90_{number}.tif"), mask=mask
        )
        google_90 = google_90.filled(0)
        google_90_arr.append(google_90.flatten())

        google_85 = np.ma.array(
            raster_to_array(f"{folder}Google85_{number}.tif"), mask=mask
        )
        google_85 = google_85.filled(0)
        google_85_arr.append(google_85.flatten())

        google_80 = np.ma.array(
            raster_to_array(f"{folder}Google80_{number}.tif"), mask=mask
        )
        google_80 = google_80.filled(0)
        google_80_arr.append(google_80.flatten())

        google_00 = np.ma.array(
            raster_to_array(f"{folder}Google00_{number}.tif"), mask=mask
        )
        google_00 = google_00.filled(0)
        google_00_arr.append(google_00.flatten())

        ghls2 = np.ma.array(raster_to_array(f"{folder}ghls2_{number}.tif"), mask=mask)
        ghls2 = ghls2.filled(0)
        ghls2_arr.append(ghls2.flatten())

        ecw = np.ma.array(raster_to_array(f"{folder}ecw_{number}.tif"), mask=mask)
        ecw = ecw.filled(0)
        ecw_arr.append(ecw.flatten())

    truth_c = np.concatenate(truth_arr)

    print_result("all", "Pred.     ", np.concatenate(pred_arr), truth_c)
    print_result("all", "Google 50 ", np.concatenate(google_00_arr), truth_c)
    print_result("all", "Google 80 ", np.concatenate(google_80_arr), truth_c)
    print_result("all", "Google 85 ", np.concatenate(google_85_arr), truth_c)
    print_result("all", "Google 90 ", np.concatenate(google_90_arr), truth_c)
    print_result("all", "OSM       ", np.concatenate(osm_arr), truth_c)

    limit = 0.5
    bin_truth = np.array(truth_c > 0, dtype="uint8")
    bin_pred = np.array(np.concatenate(pred_arr) >= limit, dtype="uint8")
    bin_osm = np.array(np.concatenate(osm_arr) >= limit, dtype="uint8")
    bin_google90 = np.array(np.concatenate(google_90_arr) >= limit, dtype="uint8")
    bin_google85 = np.array(np.concatenate(google_85_arr) >= limit, dtype="uint8")
    bin_google80 = np.array(np.concatenate(google_80_arr) >= limit, dtype="uint8")
    bin_google00 = np.array(np.concatenate(google_00_arr) >= limit, dtype="uint8")
    bin_ghls2 = np.array(np.concatenate(ghls2_arr) > 0, dtype="uint8")
    bin_ghls2_50 = np.array(np.concatenate(ghls2_arr) > 50, dtype="uint8")
    bin_ecw = np.array(np.concatenate(ecw_arr) > 0, dtype="uint8")

    print_result_bin("predition ", bin_truth, bin_pred)
    print_result_bin("Google 50 ", bin_truth, bin_google00)
    print_result_bin("Google 80 ", bin_truth, bin_google80)
    print_result_bin("Google 85 ", bin_truth, bin_google85)
    print_result_bin("Google 90 ", bin_truth, bin_google90)
    print_result_bin("OSM       ", bin_truth, bin_osm)
    print_result_bin("ECW       ", bin_truth, bin_ecw)
    print_result_bin("GHLS2 > 0 ", bin_truth, bin_ghls2)
    print_result_bin("GHLS2 > 50", bin_truth, bin_ghls2_50)


if __name__ == "__main__":
    folder = "main_folder"
    folder_buildings = folder + "buildings/"
    folder_buildings_raster = folder + "buildings_raster/"
    folder_labels = folder + "area_compare/"


    gt_vector = False
    gt_raster = False
    resample = False
    egypt_version = "23"

    # Create ground truth from vector
    if gt_vector:
        for tile in glob(folder_labels + "ground_truth_*.tif"):
            print("Processing:", tile)
            name = os.path.splitext(os.path.basename(tile))[0]
            nameid = name.split("_")[2]

            create_ground_truth(
                folder_buildings + "GOB_buildings.gpkg",
                tile,
                folder_labels + f"Google00_{nameid}.tif",
            )
            create_ground_truth(
                folder_buildings + "GOB_buildings_80p.gpkg",
                tile,
                folder_labels + f"Google80_{nameid}.tif",
            )
            create_ground_truth(
                folder_buildings + "GOB_buildings_85p.gpkg",
                tile,
                folder_labels + f"Google85_{nameid}.tif",
            )
            create_ground_truth(
                folder_buildings + "GOB_buildings_90p.gpkg",
                tile,
                folder_labels + f"Google90_{nameid}.tif",
            )
            create_ground_truth(
                folder_buildings + "OSM_buildings.gpkg",
                tile,
                folder_labels + f"OSM_{nameid}.tif",
            )

            print("Created ground truth for:", name)


    # Create ground truth from raster
    if gt_raster:
        for tile in glob(folder_labels + "ground_truth_*.tif"):
            print("Processing:", tile)
            name = os.path.splitext(os.path.basename(tile))[0]
            nameid = name.split("_")[2]

            arr = raster_to_array(
                clip_raster(
                    folder_buildings_raster + f"egypt_v{egypt_version}.tif",
                    clip_geom=tile,
                    adjust_bbox=False,
                    all_touch=False,
                )
            )

            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(0)

            array_to_raster(
                arr,
                reference=tile,
                out_path=folder_labels + f"pred_v{egypt_version}_{nameid}.tif",
            )

            arr = raster_to_array(
                clip_raster(
                    folder_buildings_raster + "ESA_WorldCover_10m_2020_egypt_builtup.tif",
                    clip_geom=tile,
                    adjust_bbox=False,
                    all_touch=False,
                )
            )

            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(0)

            array_to_raster(
                arr, reference=tile, out_path=folder_labels + f"ecw_{nameid}.tif"
            )

            arr = raster_to_array(
                clip_raster(
                    folder_buildings_raster + "GHS-S2N_egypt.tif",
                    clip_geom=tile,
                    adjust_bbox=False,
                    all_touch=False,
                )
            )

            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(0)

            array_to_raster(
                arr, reference=tile, out_path=folder_labels + f"ghls2_{nameid}.tif"
            )

            print("Created ground truth for:", name)

    if resample:
        resample_raster(
            glob(folder_labels + f"pred_v{egypt_version}*.tif"),
            target_size=100,
            out_path=folder_labels + "resampled/",
            resample_alg="average",
            prefix="",
            postfix="",
        )


    main(folder_labels + "resampled/")
    main(folder_labels)
