import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import Image, ImageTk

class ImageLoader(tk.Frame):
    def __init__(self, parent, width=800, height=600):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.image_label = tk.Label(self)
        self.image_label.pack(expand=True)
        self.pack()

        self.load_button = tk.Button(self, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.save_button = tk.Button(self, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=10)


    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_data = Image.open(file_path)
            self.display_image = self.image_data.resize((self.width, self.height))

            photo = ImageTk.PhotoImage(self.display_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo


    def update_image(self, image):
        self.display_image = image
        image = ImageTk.PhotoImage(image)

        self.image_label.config(image=image)
        self.image_label.image = image


    def reset_image(self):
        display_image = self.image_data.resize((self.width, self.height))
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            self.display_image.save(file_path)