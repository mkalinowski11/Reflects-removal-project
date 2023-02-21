import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from reflect_generator_method import *

class Reflect_Generator(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.frame = tk.Frame(self)
        self.title('Reflect generator')
        self.button_choose_image = tk.Button(self.frame , text='Prepare reflects', command = lambda : self.prepare_image()).grid(row=1,column=1)
        self.number_of_reflects_to_generate = tk.IntVar()
        self.scale_number_of_image_generate = tk.Scale(self.frame , from_=0, to=250, orient=tk.HORIZONTAL)
        self.scale_number_of_image_generate.grid(row=1,column=2)
        self.size_of_reflects = tk.Scale(self.frame , from_=5, to=25, orient=tk.HORIZONTAL)
        self.size_of_reflects.grid(row=4, column=2)
        self.use_borders = tk.BooleanVar()
        #labels
        self.label_size_of_reflects = tk.Label(self.frame, text='Number of reflects').grid(row = 3, column = 2)
        self.label_number_of_reflects_to_generate = tk.Label(self.frame, text='Size of reflects').grid(row = 5, column = 2)
        self.not_use_borders_button = tk.Radiobutton(self.frame , text='Do not use borders', variable=self.use_borders, value=False).grid(row=1, column=3)
        self.use_borders_button = tk.Radiobutton(self.frame , text='Use borders', variable=self.use_borders, value=True).grid(row=4, column=3)
        self.use_mean_filter = tk.BooleanVar()
        self.use_mean_filter_button = tk.Radiobutton(self.frame , text='Not use mean filter', variable=self.use_mean_filter, value=False).grid(row=1, column=4)
        self.use_mean_filter_button = tk.Radiobutton(self.frame , text='Use mean filter', variable=self.use_mean_filter, value=True).grid(row=4, column=4)
        self.save_button = tk.Button(self.frame , text='Save image', command=lambda: self.__image_save()).grid(row=1, column=5)
        self.upload_to_main_window = tk.Button(self.frame , text='Upload to main', command=lambda: self.__upload_image_to_recontruct()).grid(row=4, column=5)
        self.image = None
        self.reflected_image = None
        self.upload_button_image = tk.Button(self.frame , text='Select image', command = lambda : self.load_image()).grid(row=4,column=1)
        self.frame.pack()
        self.image_label = tk.Label(self, image=None)
        self.image_label.pack()
        self.button_used = False
    def prepare_image(self):
        if self.image is None:
            messagebox.showinfo(title='Error', message='Select image to add reflects')
        else:
            try:
                number_of_reflects = self.scale_number_of_image_generate.get()
                self.reflected_image = Rainbow_Dash_Algorithm.prepare_image_with_reflects(self.image, use_mean_filter=self.use_mean_filter.get(),create_borders=self.use_borders.get(),
                                                                                          max_reflects = number_of_reflects, max_radius=self.size_of_reflects.get())
                self.__display_image(selected_image='prepared')
            except Exception as exc:
                print(exc)

    def load_image(self):
        filename = filedialog.askopenfilename(initialdir='C:\\Users\\Majkel\\Desktop\\baza zdjec\\',
                                              title='select a file',
                                              filetypes=(
                                                  ('jpeg files', '*.jpg'),
                                                  ('png files', "*.png")
                                              ))
        try:
            self.image = cv2.imread(filename)
            self.__display_image()
        except Exception as exc:
            messagebox.showinfo(title='Error', message=f'Choose image')

    def __display_image(self, selected_image='start_image'):
        global imageToPaint
        if selected_image == 'prepared':
            image = self.reflected_image
        else:
            image = self.image
        imageToPaint = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        if self.button_used:
            self.image_label.destroy()
        self.image_label.configure(image = imageToPaint)

    def __upload_image_to_recontruct(self):
        if self.reflected_image is not None:
            self.master.upload_augmented_image(self.reflected_image)
            self.destroy()
        else:
            messagebox.showinfo(title='Information', message='No image to send')

    def __image_save(self):
        if self.reflected_image is not None:
            filename = filedialog.asksaveasfilename(filetypes=[('jpg','*.jpg') , ('png','*.png')],  defaultextension='.jpg')
            try:
                cv2.imwrite(filename, self.reflected_image)
                messagebox.showinfo('Info', message='Succesfully saved image')
            except Exception:
                messagebox.showinfo('Error', message='Unable to save file')
        else:
            messagebox.showinfo(title='Information', message='No image to save')
