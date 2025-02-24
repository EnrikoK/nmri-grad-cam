import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


def make_gradcam_plus_plus_heatmap_keras(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM++ heatmap for a given image and model.
    https://github.com/samson6460/tf_keras_gradcamplusplus/blob/master/gradcam.py

    Parameters:
      img_array (numpy.ndarray): Preprocessed image array.
      model (tf.keras.Model): Trained Keras model.
      last_conv_layer_name (str): Name of the last convolutional layer.
      pred_index (int, optional): Index of target class. If None, uses the top predicted class.

    Returns:
      heatmap (PIL.Image.Image): Heatmap as a PIL image (resized to 800x600 for display).

    This implementation follows the Grad-CAM++ algorithm, which computes a weighted combination
    of the activations from the last conv layer using weights derived from first, second, and third
    order gradients. These weights allow the method to better localize the regions that have a
    positive influence on the target class prediction.

    References:
      - Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks
        (Chattopadhay et al., 2018) cite arxiv:1710.11063
      - Keras Grad-CAM example cite keras_gradcam_example
    """

    # Create a model that outputs the activations of the last conv layer and the model predictions.
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = grad_model([img_array])

                category_id = np.argmax(predictions[0])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    # Convert the heatmap to an 8-bit PIL image.
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    # Resize the heatmap for visualization (optional, here set to 800x600).
    heatmap = heatmap.resize((800, 600))
    return heatmap