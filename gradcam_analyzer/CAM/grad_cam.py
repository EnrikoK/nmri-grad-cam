
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import keras


def make_gradcam_heatmap_keras(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM implementation based on the Keras documentation.
    https://keras.io/examples/vision/grad_cam/

    :param img_array: image as a numpy array
    :param model: the trained model
    :param last_conv_layer_name: the name of the last convolutional layer
    :param pred_index: the index of the class for which the Grad-CAM is computed (optional)
    :return: Tkinter-compatible PhotoImage object (for displaying in Tkinter)
    """
    dummy_input = np.random.rand(1, 205, 300, 1).astype("float32")
    _ = model.predict(dummy_input)
    model(dummy_input)


    model.layers[-1].activation = None

    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([img_array])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])  # Get the top predicted class
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()


    # Convert the heatmap to a PIL Image
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))  # Scale the values to [0, 255]
    heatmap = heatmap.resize((800,600))
    heatmap.save("Grad-CAM_"+last_conv_layer_name+".png")
    return heatmap

