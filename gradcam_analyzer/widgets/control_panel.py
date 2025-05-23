import tkinter as tk
from functools import partial
from tkinter import ttk, PhotoImage
from PIL import Image, ImageTk

from ..service import AnalysisService
from ..widgets import  ImageLoader

class ControlPanel(tk.Frame):




    def __init__(self, parent, image_loader: ImageLoader):

        super().__init__(parent,width=500, height=200, borderwidth=2)
        self.analysis_result = None
        # Dependencies
        self.image_loader = image_loader
        self.analysis_service = AnalysisService("../models")

        #Widgets
        # Frame for model/CAM/layer selectors
        self.selection_frame = ttk.LabelFrame(self, text="Analysis Settings")
        self.selection_frame.pack(padx=10, pady=10, fill="x")

        # Model selector
        ttk.Label(self.selection_frame, text="Model:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.model_selector_button = ttk.Combobox(self.selection_frame, values=self.analysis_service.get_models())
        self.model_selector_button.set(self.analysis_service.get_models()[0])
        self.model_selector_button.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # CAM selector
        ttk.Label(self.selection_frame, text="CAM Method:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        options = self.analysis_service.get_cam_options()
        self.cam_selector_button = ttk.Combobox(self.selection_frame, values=options)
        self.cam_selector_button.set(options[0])
        self.cam_selector_button.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Conv layer selector
        ttk.Label(self.selection_frame, text="Conv Layer:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.conv_layer_selector_button = ttk.Combobox(self.selection_frame,
                                                       values=["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"])
        self.conv_layer_selector_button.set("conv2d_3")
        self.conv_layer_selector_button.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        self.selection_frame.columnconfigure(1, weight=1)

        # === Class layer radio buttons ===
        self.analysis_layers = ttk.Frame(self)
        self.analysis_layers.pack(pady=5)

        self.class_layers = {
            "Original Image": tk.BooleanVar(value=True),
            "Fatty acids": tk.BooleanVar(value=False),
            "Indol": tk.BooleanVar(value=False),
            "Steroiods": tk.BooleanVar(value=False)
        }

        self.selected_layer = tk.StringVar(value="Original Image")

        for i, layer in enumerate(self.class_layers):
            rad = ttk.Radiobutton(
                self.analysis_layers,
                text=layer,
                value=layer,
                variable=self.selected_layer,
                command=partial(self.show_layer, layer)
            )
            rad.grid(row=0, column=i, padx=10, pady=5)

        # === Model prediction area ===
        self.model_analysis_area = tk.LabelFrame(self, text="Model Predictions")
        self.model_analysis_area.pack(side=tk.LEFT, padx=10, pady=10, fill="both", expand=True)

        # === Analyze button ===
        self.refresh_button = tk.Button(self, text="Analyze", command=self.analyze_image)
        self.refresh_button.pack(side=tk.RIGHT, padx=10, pady=10)

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
            label = ttk.Label(self.model_analysis_area, text=f"{key}: {round(score)}", font=("Arial", 12))
            label.grid(row=0, column=idx, padx=10, pady=5)
            idx+=1
        self.analysis_result = result

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

