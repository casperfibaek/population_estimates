import tensorflow as tf
from model_layers import inception_block, expansion_block, reduction_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
)
from buteo.machine_learning.ml_utils import tpe


def model_trio_down_class(
    shape_higher_01,
    shape_higher_02,
    shape_lower,
    donor_model_path,
    output_classes,
    activation="relu",
    output_activation="softmax",
    kernel_initializer="glorot_normal",
    sizes=[40, 48, 56],
    inception_blocks=3,
    name="denmark",
):
    model_input_higher_01 = Input(
        shape=shape_higher_01, name=f"{name}_trio_input_higher_01"
    )
    model_higher_01 = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_model_higher_01_trio",
    )(model_input_higher_01)

    for idx in range(inception_blocks):
        model_higher_01 = inception_block(
            model_higher_01,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_01_ib_00_{idx}",
        )

    model_input_higher_02 = Input(
        shape=shape_higher_02, name=f"{name}_trio_input_higher_02"
    )
    model_higher_02 = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_model_higher_02_trio",
    )(model_input_higher_02)

    for idx in range(inception_blocks):
        model_higher_02 = inception_block(
            model_higher_02,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_02_ib_00_{idx}",
        )

    model = Concatenate(name=f"{name}_trio_skip1_merge_concat")(
        [model_higher_01, model_higher_02]
    )

    model_skip_outer = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_01_ib_01",
    )

    model = reduction_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_01_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_01_ib_02_{idx}",
        )

    model_input_model_lower = Input(
        shape=shape_lower, name=f"{name}_trio_input_model_lower"
    )

    model_lower = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_model_lower_trio",
    )(model_input_model_lower)

    for idx in range(inception_blocks):
        model_lower = inception_block(
            model_lower,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_lower_ib_00_{idx}",
        )

    model_skip_inner = Concatenate(name=f"{name}_trio_skip2_merge_concat")(
        [model, model_lower]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model_skip_inner,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_00_{idx}",
        )

    model = reduction_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_01_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[2],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_eb_00",
    )

    model = Concatenate(name=f"{name}_trio_skip_inner_concat")(
        [model, model_skip_inner]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_02_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_eb_01",
    )

    model = Concatenate(name=f"{name}_trio_skip_outer_concat")(
        [model, model_skip_outer]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_tail_trio_ib_00_{idx}",
        )

    model = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_trio_ib_01",
    )

    model = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_conv_00",
    )(model)

    output = Conv2D(
        output_classes,
        kernel_size=3,
        padding="same",
        activation=tf.keras.activations.softmax,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_output",
        dtype="float32",
    )(model)

    model = Model(
        inputs=[
            model_input_higher_01,
            model_input_higher_02,
            model_input_model_lower,
        ],
        outputs=output,
    )

    # # transfer weights
    donor_model = tf.keras.models.load_model(
        donor_model_path, custom_objects={"tpe": tpe}
    )

    # version 2
    # 37 is top of tail
    # 12 is after first inception block of the tail
    for idx, _ in enumerate(donor_model.layers[:-12]):
        model.layers[idx].set_weights(donor_model.layers[idx].get_weights())
        # model.layers[idx].trainable = False

    return model
