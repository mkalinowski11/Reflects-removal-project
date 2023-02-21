import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import messagebox

class ImageManagager(tk.Toplevel):
    def __init__(self, master, image, image_window):
        super().__init__(master)
        self.frame = tk.Frame(self)
        self.title('Image resizer')
        self.button1 = tk.Button(self.frame, text='⇐', command = lambda : self.changePosition('left')).grid(row= 2,column= 1)
        self.button2 = tk.Button(self.frame, text='⇒', command = lambda : self.changePosition('right')).grid(row= 2,column= 3)
        self.button3 = tk.Button(self.frame, text='⇑', command = lambda : self.changePosition('up')).grid(row= 1,column= 2)
        self.button4 = tk.Button(self.frame, text='⇓', command = lambda : self.changePosition('down')).grid(row= 3,column= 2)
        self.button5 = tk.Button(self.frame, text='cut image', command = self.returnReducedImage).grid(row= 4,column= 2)
        self.frame.pack()
        self.startPoint = [0, 0]
        self.endPoint = [256, 256]
        self.image = image
        self.image_label = tk.Label(self)
        self.image_label.pack()
        self.updateImage()
        self.image_window_on_main = image_window

    def changePosition(self, action:str):
        if action == 'up':
            if self.startPoint[1] >= 10:
                self.startPoint[1] -= 10
                self.endPoint[1] -= 10

        elif action ==  'down':
            if self.endPoint[1] <= self.image.shape[0] - 10:
                self.startPoint[1] += 10
                self.endPoint[1] += 10

        elif action == 'right':
            if self.endPoint[0] <= self.image.shape[1] - 10:
                self.startPoint[0] += 10
                self.endPoint[0] += 10

        elif action == 'left':
            if self.startPoint[0] >= 10:
                self.startPoint[0] -= 10
                self.endPoint[0] -= 10

        self.updateImage()

    def updateImage(self):
        global imageToPaint
        imageToPaint = ImageTk.PhotoImage(
            Image.fromarray(
                cv2.rectangle(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB),
                              self.startPoint,
                              self.endPoint, color=(255, 0, 0),
                              thickness=3)
            )
        )
        #put new image on screen
        self.image_label.configure(image = imageToPaint)

    def returnReducedImage(self):
        self.master.set_resized_image([*self.startPoint, *self.endPoint], self.image, self.image_window_on_main)
        self.destroy()