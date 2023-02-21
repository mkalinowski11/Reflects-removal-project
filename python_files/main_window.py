import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import PIL
from PIL import Image, ImageTk
from image_manager import ImageManagager
from model_manager import ModelManager, InvalidDataException, InvalidShapeException, InvalidModelPath
from Measures import Measures
import cv2
import numpy as np
from relfect_maker import *
from collections import deque
from display_window import Reconstruction_Image_Display
from tensorflow.image import resize

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry('1000x420')
        self.title('Image reconstruction app')
        self.resizable = (False, False)
        self.image_to_reconstruct = None
        self.reconstructed_image = None
        self.reference_image = None
        self.button_frame = tk.Frame(self)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button1 = tk.Button(self.button_frame, text='Image to reconstruct',
                                 command=lambda: self.uploadImage('image_to_reconstruct')).grid(row=0, column=1, sticky='ew')
        self.button2 = tk.Button(self.button_frame, text='Image reconstruction',
                                 command=lambda: self.reconstruct_image()).grid(row=1, column=1, sticky='ew')
        self.button3 = tk.Button(self.button_frame, text='Reference image',
                                 command=lambda: self.uploadImage('reference_image')).grid(row=2, column=1, sticky='ew')
        self.button4 = tk.Button(self.button_frame, text='Prepare image',
                                 command=lambda: self.run_reflect_maker()).grid(row=3, column=1, sticky='ew')
        
        #quality measures button
        self.button5 = tk.Button(self.button_frame, text='Quality measures',
                               command=self.make_measures).grid(row=4, column=1, sticky='ew')
        self.button6 = tk.Button(self.button_frame, text='Change model',
                               command=lambda:self.change_model()).grid(row=5, column=1, sticky='ew')
        self.button7 = tk.Button(self.button_frame, text='Reconstruction history',
                               command=lambda:self.__open_reconstruction_history()).grid(row=6, column=1, sticky='ew')
        #info button
        self.button8 = tk.Button(self.button_frame, text='Info',
                                 command=lambda: self.show_info()).grid(row=7, column=1, sticky='ew')
        self.button_frame.pack(side=tk.RIGHT)
        #metrics frame
        self.metrics_frame = tk.Frame(self)
        self.model_name = tk.Label(self.metrics_frame)
        self.model_name .grid(row=4, column=4)
        self.measure_1 = tk.Label(self.metrics_frame, text='PSNR = x')
        self.measure_1 .grid(row=5, column=4)
        self.measure_2 = tk.Label(self.metrics_frame, text='SSIM = x')
        self.measure_2.grid(row=6, column=4)
        self.measure_3 = tk.Label(self.metrics_frame, text='NAE = x')
        self.measure_3.grid(row=7, column=4)
        self.measure_4 = tk.Label(self.metrics_frame, text='SC = x' )
        self.measure_4.grid(row=8, column=4)
        self.metrics_frame.pack(side=tk.BOTTOM)
        #image frame
        self.image_frame = tk.Frame(self)
        #image labels
        self.image_to_reconstruct_description = tk.Label(self.image_frame, text='')
        self.reconstructed_image_description = tk.Label(self.image_frame, text='')
        self.reference_image_description = tk.Label(self.image_frame, text='')
        self.image_to_reconstruct_description.grid(row=0, column=1)
        self.reconstructed_image_description.grid(row=0, column=2)
        self.reference_image_description.grid(row=0, column=3)
        #image display
        self.image_to_reconstruction_label = tk.Label(self.image_frame, image=None)
        self.image_to_reconstruction_label.grid(row=1, column = 1)
        self.reconstructed_image_label = tk.Label(self.image_frame, image=None)
        self.reconstructed_image_label.grid(row=1, column = 2)
        self.reference_image_label = tk.Label(self.image_frame, image=None)
        self.reference_image_label.grid(row=1, column = 3)
        self.image_frame.pack(side=tk.LEFT)
        self.updateImages()
        self.model_type = None
        self.metrics = Measures()
        self.metrics_list = None
        self.model = ModelManager(self, '../model/unet_mae_150_base1.h5')
        self.reconstruction_history = deque()
        self.reconstruction_history_model_types = deque()
        #mainloop of app
        self.mainloop()

    def uploadImage(self, image):
        try:
            #open filedialog
            filename = filedialog.askopenfilename(initialdir='../data/',
                                                   title='select a file',
                                                   filetypes=(
                                                        ('jpeg files', '*.jpg'),
                                                        ('png files', "*.png")
                                                             ))
            #change images on main page app
            if image == 'image_to_reconstruct':
                self.image_to_reconstruct = self.fitImage(
                cv2.imread(filename)
                )
            elif image == 'reference_image':
                self.reference_image = self.fitImage(
                    cv2.imread(filename),image_window=0
                )
            #call function
            self.updateImages()
        except Exception as exc:
            messagebox.showinfo('Error', message=f'choose correct image {image} exception message {exc}')

    def fitImage(self, image, image_window=1):
        if image.shape == (256, 256, 3) :
            return image
        #func is going to return photoManager
        elif image.shape[0] > 256 and image.shape[1] > 256 and image.shape[2] == 3:
            ImageManagager(self, image, image_window)
        else :
            image = resize(image, (256, 256))
            image = np.array(image, dtype='object').astype('uint8')
            return image

    def set_resized_image(self, cords, image, image_window=1):
        resized_image = image[cords[1]:cords[3], cords[0]:cords[2], :]
        if image_window:
            self.image_to_reconstruct = resized_image
        else:
            self.reference_image = resized_image
        self.updateImages()

    def make_measures(self):
        if self.reference_image is not None:
            reference_image = self.reference_image.astype('float32')
            #reference image has to be converted to float32 type
            mertrics = self.metrics.make_metrics(reference_image, self.reconstructed_image)
            self.metrics_list = mertrics
            self.update_metrics()
        else:
            messagebox.showinfo(title='Error', message='Reference image is None')

    def returnImage(self, image: str):
        if image == 'upload':
            return self.image_to_reconstruct
        elif image == 'reconstructed':
            return self.reconstructed_image
        else:
            return self.reference_image

    def reconstruct_image(self):
        try:
            tmp_image = self.image_to_reconstruct
            tmp_image = (tmp_image).astype('float32')
            tmp_image = tmp_image / 255.
            tmp_image = tmp_image[None, :,:,:]
            reconstructed_image = self.model.make_prediction(tmp_image)
            reconstructed_image = reconstructed_image*255
            self.reconstructed_image = (np.array(reconstructed_image).astype('float32'))[0]
            self.__add_images_to_history()
            self.updateImages()
        except InvalidDataException as exc:
            messagebox.showinfo(title='Error', message=f'No input image {exc}')
        except InvalidShapeException as exc:
            messagebox.showinfo(title='Error', message=f'No input image {exc}')
        except Exception as exc:
            messagebox.showinfo(title='Error', message=f'Unable to reconstruct image, reason {exc}')

    def __prepare_empty_image(self):
        return ImageTk.PhotoImage(
                   Image.fromarray(
                            cv2.rectangle(np.full((256,256,3), 1, dtype='uint8'), (0,0), (255,255), (0,0,255), 4)
                )
            )

    def updateImages(self):
        #updates images on GUI
        global tmpImage1
        if self.image_to_reconstruct is not None:
            tmpImage1 = ImageTk.PhotoImage(Image.fromarray(
                        cv2.cvtColor(self.image_to_reconstruct, cv2.COLOR_BGR2RGB)
                                                    )
            )
            self.update_images_descriptions('image_to_reconstruct')
        #filling empty region with square
        else:
            tmpImage1 = self.__prepare_empty_image()
            self.update_images_descriptions('image_to_reconstruct')

        if self.reconstructed_image is not None:
            global tmpImage2
            tmpImage2 = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor((self.reconstructed_image).astype(np.uint8), cv2.COLOR_BGR2RGB))
            )
        else:
            tmpImage2 = self.__prepare_empty_image()
            
            self.update_images_descriptions('reconstructed_image')
        if self.reference_image is not None:
            global tmpImage3
            tmpImage3 = ImageTk.PhotoImage(Image.fromarray(
                cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
                )
            )
        else:
            tmpImage3 = self.__prepare_empty_image()
            self.update_images_descriptions('reference_image')
        self.image_to_reconstruction_label.configure(image = tmpImage1)
        self.reconstructed_image_label.configure(image = tmpImage2)
        self.reference_image_label.configure(image = tmpImage3)

    def return_fitted_image(self, cords,image):
        if image is not None:
            self.image_to_reconstruct = image[cords[1]:cords[3], cords[0]:cords[2],:]
        self.updateImages()
    
    def upload_augmented_image(self, image):
        if image is not None:
            print(image.shape)
            self.image_to_reconstruct = self.fitImage(image,image_window=1)
            self.updateImages()

    def update_metrics(self):
        if self.metrics_list is not None:
            self.measure_1.configure(text=f'PSNR = {self.metrics_list[0]:.3f}')
            self.measure_2.configure(text=f'SSIM = {self.metrics_list[1]:.3f}')
            self.measure_3.configure(text=f'NAE = {self.metrics_list[2]:.3f}')
            self.measure_4.configure(text=f'SC = {self.metrics_list[3]:.3f}')

    def run_reflect_maker(self):
        try:
            Reflect_Generator(self)
        except Exception:
            messagebox.showinfo('Error', message='Unable to find module')
    
    def update_images_descriptions(self, image_to_label):
        if image_to_label == 'image_to_reconstruct':
            self.image_to_reconstruct_description.configure(text='Image to reconstruction')
        elif image_to_label == 'reconstructed_image':
            self.reconstructed_image_description.configure(text='Reconstructed image')
        else:
            self.reference_image_description.configure(text='Reference image')
    
    def change_model(self):
        path = filedialog.askopenfilename(initialdir='../model/',
                                                   title='select a file',
                                                   filetypes=(
                                                        ('h5 files', '*.h5'),))
        try:
            self.model.reload_model_architecture(path)
        except InvalidModelPath as exc:
            messagebox.showinfo(title='Error', message=f'{exc}')
        except Exception as exc:
            messagebox.showinfo(title='Error', message=f'Unable to load model : {exc}')

    def show_info(self):
        messagebox.showinfo(
            'Information', message=\
            '''This application is created for visualization of reconstruction, options are listed below:\n
            Image to reconstruct -> allows user to choose image to be reconstructed, format 256x256x3 is needed. If selected image's shape is bigger,
            resizer is opened to allow to cut image with correct size. Application allows *.jpg and *.png formats\n
            Image reconstruction -> allows user to reconstruct selected image\n
            Reference image -> allows user to select reference image\n
            Prepare image -> allows user to prepare custom image with random reflects for reconstruction. \n
            Quality measures -> if reconstructed image and reference image are displayed, displays quality measures between images\n
            Load model -> enables user to change model architecture for reconstruction\n
            Reconstruction history -> displays last 4 images reconstruction'''
        )

    def __add_images_to_history(self):
        if self.image_to_reconstruct is not None and self.reconstructed_image is not None:
            self.__dequeue_checker()
            self.reconstruction_history.append(self.image_to_reconstruct)
            self.reconstruction_history.append(self.reconstructed_image)
            self.reconstruction_history_model_types.append(self.model_type)
    
    def __dequeue_checker(self):
        #removes 2 images from deque - to reconstruction and prediction
        if len(self.reconstruction_history) == 8:
            for _ in range(2):
                self.reconstruction_history.popleft()
        #removes model label from deque
        if len(self.reconstruction_history_model_types) == 4:
            self.reconstruction_history_model_types.popleft()
    
    def __open_reconstruction_history(self):
        if not self.reconstruction_history:
            messagebox.showinfo(title='Information', message='No reconstructions yet')
        else:
            Reconstruction_Image_Display(self, len(self.reconstruction_history))

    def update_model_label(self, model_type):
        if len(model_type) == 4:
            self.model_name.configure(text = f'Model type : {model_type[0]}, loss : {model_type[1]}, epochs : {model_type[2]}, trained on : {model_type[3]}')
        else:
            self.model_name.configure(text = ''.join(model_type))
        self.model_type = model_type

if __name__ == '__main__':
    MainWindow()