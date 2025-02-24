# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from PIL import Image


def compute_heatmap(img_array, model, last_conv_layer_name, pred_index,eps=1e-8):

        """
        Guided Grad-CAM implementations based of https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/?utm_source=chatgpt.com
        :param img_array:
        :param model:
        :param last_conv_layer_name:
        :param pred_index:
        :param eps:
        :return:
        """

        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model

        gradModel = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output,
                     model.output])

        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(img_array, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, pred_index]
            # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch

        # Commented out but dont know why, it work better
        #convOutputs = convOutputs[0]
        #guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1).numpy()

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        # Convert the heatmap to a PIL Image
        heatmap = Image.fromarray((cam[0] * 255).astype(np.uint8))  # Scale the values to [0, 255]
        heatmap = heatmap.resize((800, 600))
        #heatmap.save("Grad-CAM_" + last_conv_layer_name + ".png")
        return heatmap


        #(w, h) = (image.shape[2], image.shape[1])
        #heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        #numer = heatmap - np.min(heatmap)
        ##denom = (heatmap.max() - heatmap.min()) + eps
        #heatmap = numer / denom
        #heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        #return heatmap