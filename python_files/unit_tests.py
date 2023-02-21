#unit tests for app
from tkinter import Tk
import unittest
from reflect_generator_method import Rainbow_Dash_Algorithm, NoImageExcepiton, ImageShapeException, InvalidImage
from Measures import Measures, NoImageExcepiton as no_img, ImageShapeException as invalid_shape, InvalidImage as invalid_data
from model_manager import ModelManager, InvalidDataException as model_invalid_data, InvalidShapeException as model_invalid_shape, InvalidModelPath as invalid_path
import numpy as np
from main_window import MainWindow

class Class_For_Test():
    def update_model_label(self, obj):
        pass

class Reflect_generator_test(unittest.TestCase):

    def test_image_border_create(self):
        image = np.zeros((256,256,3))
        created_image = Rainbow_Dash_Algorithm.prepare_image_with_reflects(image)
        self.assertTrue(image.shape == created_image.shape)

    def test_create_img_exception(self):
        image = None
        self.assertRaises(NoImageExcepiton, Rainbow_Dash_Algorithm.prepare_image_with_reflects, image)
        image = [1,2,3]
        self.assertRaises(InvalidImage, Rainbow_Dash_Algorithm.prepare_image_with_reflects, image)

    def test_image_subtraction(self):
        image_1 = np.ones((256,256,3))
        image_1 = image_1 * 255
        image_2 = np.ones((256,256,3))
        image_subtracted = image_1 - image_2
        self.assertEqual(Rainbow_Dash_Algorithm.image_subtraction(image_1, image_2).all(), image_subtracted.all())

    def test_image_subtraction_errors(self):
        image_1 = None
        image_2 = np.ones((256,256,3))
        self.assertRaises( NoImageExcepiton ,Rainbow_Dash_Algorithm.image_subtraction,image_1, image_2)
        image_1 = np.ones((256, 256, 3))
        image_2 = None
        self.assertRaises(NoImageExcepiton, Rainbow_Dash_Algorithm.image_subtraction, image_1, image_2)
        image_1 = np.ones((256, 256, 3))
        image_2 = np.ones((254, 254, 3))
        self.assertRaises(ImageShapeException, Rainbow_Dash_Algorithm.image_subtraction, image_1, image_2)
        image_1 = [2,3,4,5]
        self.assertRaises(InvalidImage, Rainbow_Dash_Algorithm.image_subtraction, image_1, image_2)

class Model_Manager_Test(unittest.TestCase):
    def test_model_manager_invalid_path_test(self):
        self.assertRaises(invalid_path, ModelManager, Class_For_Test() , '../model/bad_path.h5')
    
    def test_model_none_image_input(self):
        model = ModelManager(master = Class_For_Test(), path = '../model/unet_ssim_250_base3.h5')
        image1 = None
        self.assertRaises(model_invalid_data, model.make_prediction, image1)
    
    def test_model_bad_image_input(self):
        model = ModelManager(master = Class_For_Test(), path = '../model/unet_ssim_250_base3.h5')
        image = [1,2,3,4,5,10]
        self.assertRaises(model_invalid_data, model.make_prediction, image)

    def test_model_invalid_shape_input(self):
        model = ModelManager(master = Class_For_Test(), path = '../model/unet_ssim_250_base3.h5')
        image = np.ones((256,256,3))
        self.assertRaises(model_invalid_shape, model.make_prediction, image)


    def test_model_manager_prediction_test(self):
        model = ModelManager(master = Class_For_Test(), path = '../model/unet_ssim_250_base3.h5')
        image = np.ones((256,256,3))
        image = image[None, :,:,:]
        reconstructed_image = model.make_prediction(image)
        self.assertEqual(image.shape, reconstructed_image.shape)
    
    def test_reload_model(self):
        model = ModelManager(master = Class_For_Test(), path = '../model/unet_ssim_250_base3.h5')
        model.reload_model_architecture(path = '../model/unet_ssim_250_base3.h5')
    
    def test_reload_model_err(self):
        model = ModelManager(master = Class_For_Test(), path = '../model/unet_ssim_250_base3.h5')
        self.assertRaises(Exception, model.reload_model_architecture, 'some/bad/pathmodel.h5')

class Measures_TestCase(unittest.TestCase):

    def test_measures(self):
        image1 = np.ones((45,45,3)).astype('float32')
        image1 = image1 / 255.
        image2 = np.ones((45, 45, 3)).astype('float32')
        image1 = image1 / 255.
        measurements_class = Measures()
        ssim = measurements_class.SSIM(image1, image2)
        psnr = measurements_class.PSNR(image1, image2)
        nae = measurements_class.NAE(image1, image2)
        sc = measurements_class.SC(image1, image2)
        test_measurements = (psnr, ssim, nae, sc)
        class_meas = measurements_class.make_metrics(image1, image2)
        self.assertTrue(test_measurements, class_meas)
    
    def test_measures_no_img_error(self):
        image1 = None
        image2 = np.ones((24,24,3))
        measurements_class = Measures()
        self.assertRaises(no_img, measurements_class.make_metrics, image1, image2)
        image1 = np.ones((24, 24, 3))
        image2 = None
        self.assertRaises(no_img, measurements_class.make_metrics, image1, image2)
    
    def test_measures_not_equal_image_shape_error(self):
        image1 = np.ones((23,23,3))
        image2 = np.ones((24,24,3))
        measurements_class = Measures()
        self.assertRaises(invalid_shape, measurements_class.make_metrics, image1, image2)
    
    def test_measures_invalid_data_error(self):
        image1 = [23,33,49]
        image2 = np.array([23,33,49])
        measurements_class = Measures()
        self.assertRaises(invalid_data, measurements_class.make_metrics, image1, image2)

if __name__ == '__main__':
    unittest.main()