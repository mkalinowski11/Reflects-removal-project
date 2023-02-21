#window that displays images
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from math import ceil

class Reconstruction_Image_Display(tk.Toplevel):
    def __init__(self, master, number_of_images):
        super().__init__(master)
        self.number_of_images = number_of_images
        self.title("Reconstruction history")
        self.image_frame = tk.Frame(self)
        self.model_labels = [tk.Label(self.image_frame) for _ in range(len(self.master.reconstruction_history_model_types))]
        self.image_labels = [tk.Label(self.image_frame) for j in range(self.number_of_images)]
        self.__configure_image_labels()
        self.image_frame.pack()
        self.images = []
        self.display_images()
    
    def display_images(self):
        global images
        for idx, image in enumerate(self.master.reconstruction_history):
            if not idx %2:
                self.images.append(ImageTk.PhotoImage(Image.fromarray(
                    cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB))))
        for idx, image in enumerate(self.master.reconstruction_history):
            if idx%2:
                self.images.append(ImageTk.PhotoImage(Image.fromarray(
                    cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB))))
        for idx, image in enumerate(self.images):
            self.image_labels[idx].configure(image=image)
        for idx, label in enumerate(self.model_labels):
            label.configure(text=', '.join(self.master.reconstruction_history_model_types[idx]))
    
    def __configure_image_labels(self):
        row = 1
        column = 1
        self.__configure_model_labels()
        for image_label in self.image_labels:
            image_label.grid(row=row, column=column)
            column += 1
            if column > self.number_of_images / 2:
                column = 1
                row += 1

    def __configure_model_labels(self):
        for idx, label in enumerate(self.model_labels):
            label.grid(row=0, column = (idx+1))