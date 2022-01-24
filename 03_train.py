import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping

from buteo.machine_learning.ml_utils import tpe, SaveBestModel
from buteo.machine_learning.augmentation import image_augmentation
from buteo.utils import timing


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")


folder = "main_folder"
folder_country = folder + "Egypt/"
folder_gaza = folder + "Gaza/"
folder_israel = folder + "Israel/"
outdir = folder + "models/"

model_name = "name_of_model"
donor_model_path = outdir + "name_of_donor_model"

add_noise = True

labels = np.concatenate([
    np.load(folder_country + "label_area.npy"),
    np.load(folder_gaza + "label_area.npy"),
    np.load(folder_israel + "label_area.npy"),
])

shufflemask = np.random.permutation(labels.shape[0])
labels = labels[shufflemask]

rgbn = np.concatenate([
    np.load(folder_country + "RGBN.npy"),
    np.load(folder_gaza + "RGBN.npy"),
    np.load(folder_israel + "RGBN.npy"),
])[shufflemask]

sar = np.concatenate([
    np.load(folder_country + "SAR.npy"),
    np.load(folder_gaza + "SAR.npy"),
    np.load(folder_israel + "SAR.npy"),
])[shufflemask]

reswir = np.concatenate([
    np.load(folder_country + "RESWIR.npy"),
    np.load(folder_gaza + "RESWIR.npy"),
    np.load(folder_israel + "RESWIR.npy"),
])[shufflemask]

test_size = 10000

x_train = [
    rgbn[:test_size],
    sar[:test_size],
    reswir[:test_size],
]

x_val = [
    rgbn[-test_size:],
    sar[-test_size:],
    reswir[-test_size:],
]

y_train = [
    labels[:test_size],
]

y_val = [
    labels[-test_size:],
]

#OLD VERSION

x_test = [
   np.load(folder_country + "RGBN_test.npy"),
   np.load(folder_country + "SAR_test.npy"),
   np.load(folder_country + "RESWIR_test.npy"),
]

y_test = [
   np.load(folder_country + "label_area_test.npy"),
]


if add_noise:
    x_train, y_train = image_augmentation(x_train, y_train, options={
        "scale": 0.02, # 0.035
        "band": 0.01, # 0.01
        "contrast": 0.01, # 0.01
        "pixel": 0.01, # 0.01
        "drop_threshold": 0.00,
        "clamp": True,
        "clamp_max": 1,
        "clamp_min": 0,
    })

lr = 0.0001
min_delta = 0.005

with tf.device("/device:GPU:0"):
    epochs = [25, 25, 25, 25, 25]
    bs = [8, 16, 32, 64, 128]

    # version 1 Without momentum from the Ghana model
    donor_model = tf.keras.models.load_model(donor_model_path, custom_objects={"tpe": tpe})
    model = tf.keras.models.clone_model(donor_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=optimizer,
        loss="mse", # or sumbias
        metrics=["mse", "mae", tpe], 
    )

    model.set_weights(donor_model.get_weights())

    # version 2 With momentum from the donor model
    # model = tf.keras.models.load_model(donor_model_path, custom_objects={"tpe": tpe})


    val_loss, _mse, _mae, _tpe = model.evaluate(x=x_test, y=y_test, batch_size=512) # Evaluate the donor model

    start = time.time()
    monitor = "val_loss" # or "val_tpe"

    # This ensures that the weights of the best performing model is saved at the end
    save_best_model = SaveBestModel(save_best_metric=monitor)

    # Reduces the amount of total epochs by early stopping a new fit if it is not better than the previous fit.
    best_val_loss = val_loss
   
    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            shuffle=True,
            epochs=use_epoch,
            initial_epoch=initial_epoch,
            verbose=1,
            batch_size=use_bs,
            use_multiprocessing=True,
            workers=0,
            callbacks=[
                EarlyStopping(            #it gives the model 3 epochs to improve results based on val_loss value, if it doesnt improve-drops too much, the model running
                    monitor=monitor,   #is stopped. If this continues, it would be overfitting (refer to notes)
                    patience=5,
                    min_delta=min_delta,
                    mode="min", # loss is suppose to minimize
                    baseline=best_val_loss, # Fit has to improve upon baseline
                    restore_best_weights=True, # If stopped early, restore best weights.
                ),
                save_best_model,
            ],
        )

        # Saves the val loss to the best_val_loss for early stopping between fits.
        model.set_weights(save_best_model.best_weights)
        val_loss, _mse, _mae, _tpe = model.evaluate(x=x_val, y=y_val, batch_size=512) #it evaluates the accuracy of the model we just created here
        best_val_loss = val_loss
        model.save(f"{outdir}{model_name.lower()}_{str(use_epoch)}")

    print("Saving...")

    val_loss, _mse, _mae, _tpe = model.evaluate(x=x_test, y=y_test, batch_size=512) #it evaluates the accuracy of the model we just created here
    model.save(f"{outdir}{model_name.lower()}")

    timing(start)
