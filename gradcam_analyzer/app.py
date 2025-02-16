import tkinter as tk

from gradcam_analyzer.widgets.control_panel import ControlPanel
from widgets import ImageLoader

class GRADCAMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GRAD-CAM analyzer for nmri images")

        self.image_loader = ImageLoader(self)
        self.control_panel = ControlPanel(self, self.image_loader)

        self.image_loader.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.control_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

if __name__ == "__main__":
    root = GRADCAMApp()
    root.mainloop()