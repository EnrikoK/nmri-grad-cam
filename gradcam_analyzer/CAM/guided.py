
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import keras


def make_gradcam_heatmap_keras(img_array, model, last_conv_layer_name, pred_index=None):

    model.layers[-1].activation = None

    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    guided_grads = tf.maximum(grads, 0)
    weights = tf.reduce_mean(guided_grads, axis=(0,1))
    cam = tf.reduce_sum(weights*conv_outputs, axis=-1).numpy()

    cam = np.maximum(cam,0)
    cam = cam / cam.max()

    # Convert the heatmap to a PIL Image
    heatmap = Image.fromarray((cam[0] * 255).astype(np.uint8))  # Scale the values to [0, 255]
    heatmap = heatmap.resize((800,600))
    heatmap.save("Grad-CAM_"+last_conv_layer_name+".png")
    return heatmap
