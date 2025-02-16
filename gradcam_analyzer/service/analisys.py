import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from ..CAM.grad_cam import  make_gradcam_heatmap_keras

class AnalysisService():

    cam_options = {
        "Grad-CAM": make_gradcam_heatmap_keras
    }

    def __init__(self, models_folder: str):
        self.models_folder = models_folder

    def get_models(self):
        if os.path.isdir(self.models_folder):
            models = filter(
                lambda x: x.endswith(".h5") or x.endswith(".keras"),
                os.listdir(self.models_folder)
            )
            return list(models)
        else:
            return None

    def preprocess_pil_image(self,pil_img):
        """
        Preprocess a PIL image for model inference.

        :param pil_img: A PIL Image object
        :return: Preprocessed image as a NumPy array
        """
        pil_img = pil_img.resize((300, 205)).convert("L")  # "L" mode for grayscale
        img_tensor = img_to_array(pil_img)  # Shape: (height, width, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Shape: (1, 205, 300, 1)
        img_tensor /= 255.0
        return img_tensor


    def load_model_from_location(self, model_name):
        return load_model(os.path.join(self.models_folder, model_name))
    
    def get_analysis(self, model_name, selected_cam, image, prediction_classes):
        """

        :param model_name: Name of the selected model in the specified models folder. File name is used as model name
        :param selected_cam: CAM option
        :param image: The image that needs to be analyzed
        :param prediction_classes: Array of class labels that the model uses for predictions in logical order
        [class1, class2, class3, ...]
        :return: dict classNumber=(score,cam-image)
        """
        model = self.load_model_from_location(model_name)

        image = self.preprocess_pil_image(image)


        predictions = model.predict(image)
        results = {}

        for index, class_name in enumerate(prediction_classes):

            score = predictions[0, index]

            cam_image = AnalysisService.cam_options[selected_cam](
                img_array=image,
                model=model,
                last_conv_layer_name="conv2d",#model.layers[0].name
                pred_index=index
            )

            results[class_name] = (score, cam_image)

        return results