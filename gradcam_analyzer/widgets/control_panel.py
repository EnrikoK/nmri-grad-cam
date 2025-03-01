import tkinter as tk
from functools import partial
from tkinter import ttk, PhotoImage
from PIL import Image, ImageTk

from ..service import AnalysisService
from ..widgets import  ImageLoader

class ControlPanel(tk.Frame):


    def refresh_image(self, image:PhotoImage) -> None :
        self.image_loader.update_image(image)


    def add_model_prediction_scores(self, scores: dict[str,float] = None):
        if scores is None:
            return
        idx = 0
        for key, val in scores.items():
            label = ttk.Label(self.model_analysis_area, text=f"{key}: {val}", font=("Arial", 12))
            label.grid(row=0, column=idx, padx=10, pady=5)
            idx+=1


    def analyze_image(self):
        result = self.analysis_service.get_analysis(
            self.model_selector_button.get(),
            self.cam_selector_button.get(),
            self.image_loader.image_data,
            ["Fatty acids", "Indol", "Steroiods"],
            self.conv_layer_selector_button.get()
        )
        idx = 0
        for key, (score,cam_image) in result.items():
            label = ttk.Label(self.model_analysis_area, text=f"{key}: {round(score,5)}", font=("Arial", 12))
            label.grid(row=0, column=idx, padx=10, pady=5)
            idx+=1
        self.analysis_result = result


    def __init__(self, parent, image_loader: ImageLoader):

        super().__init__(parent,width=500, height=200, borderwidth=2)
        self.analysis_result = None
        # Dependencies
        self.image_loader = image_loader
        self.analysis_service = AnalysisService("../models")

        #Widgets
        # Selecting the model
        self.model_selector_button = ttk.Combobox(self, values=self.analysis_service.get_models())
        self.model_selector_button.set(self.analysis_service.get_models()[0])
        # Selecting the CAM options
        options = self.analysis_service.get_cam_options()
        self.cam_selector_button = ttk.Combobox(self, values=options)
        self.cam_selector_button.set(options[0])
        # Selecting the analysis layer
        self.conv_layer_selector_button = ttk.Combobox(self, values=["conv2d","conv2d_1", "conv2d_2", "conv2d_3"])
        self.conv_layer_selector_button.set("conv2d_3")

        # Area for populating the model predictions
        self.model_analysis_area = tk.LabelFrame(self, text="Model predictions")

        # Analysis respect to target class selector
        self.analysis_layers = ttk.Frame(self)
        self.analysis_layers.grid(row=0, column=0)
        self.class_layers = {
            "Original Image": tk.BooleanVar(value=True),
            "Fatty acids": tk.BooleanVar(value=False),
            "Indol": tk.BooleanVar(value=False),
            "Steroiods": tk.BooleanVar(value=False)
        }

        self.selected_layer = tk.StringVar(value="Original Image")  # StringVar to hold the selected layer, default image
        # Create a radio button for each layer
        for i, layer in enumerate(self.class_layers):
            rad = ttk.Radiobutton(
                self.analysis_layers,
                text=layer,
                value=self.class_layers[layer],  # Value of the radio button
                variable=self.selected_layer,  # This variable will hold the selected value
                command=partial(self.show_layer, layer)  # Call show_layer when selected
            )
            rad.grid(row=0, column=i, padx=10, pady=5)

        self.refresh_button = tk.Button(self, text="Analyze", command=self.analyze_image)


        #Pack all things into frame
        self.analysis_layers.pack(pady=5)
        self.model_selector_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.cam_selector_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.conv_layer_selector_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.model_analysis_area.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)
        self.refresh_button.pack(side=tk.RIGHT, padx=10, pady=10)



    def create_masked_layer(self, mask, alpha=0.5):
        """ Generate a red mask based on the CAM image returned and merge it with the original image"""
        base = self.image_loader.image_data.convert("RGBA")
        base=base.resize((800,600))
        red_mask = Image.merge("RGBA", (
            mask,  # Red channel = grayscale mask
            Image.new("L", mask.size, 0),  # Green channel = 0
            Image.new("L", mask.size, 0), # Blue channel = 0
            mask
        ))
        red_mask = red_mask.resize(base.size)
        return Image.alpha_composite(base,red_mask)

    def show_layer(self, layer):
        if layer == "Original Image":
            self.image_loader.reset_image()
        else:
            image= self.analysis_result[layer][1]
            #image = ImageTk.PhotoImage(self.create_masked_layer(image))
            self.image_loader.update_image(self.create_masked_layer(image))

