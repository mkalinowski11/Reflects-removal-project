import os
import tensorflow as tf


class InvalidDataException(Exception):
    def __init__(self, msg = "Invalid input data"):
        super().__init__(msg)

class InvalidShapeException(Exception):
    def __init__(self, msg = "Invalid data shape"):
        super().__init__(msg)

class InvalidModelPath(Exception):
    def __init__(self, msg = "Invalid model directory"):
        super().__init__(msg)

class ModelManager():
    def __init__(self,master,path):
        self.model = None
        self.master = master
        self.__load_model(path)
    
    def __load_model(self, path):
        path = os.path.join('', path)
        if not os.path.exists(path):
            raise InvalidModelPath()
        else:
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                self.__get_model_name(path)
            except Exception as exception:
                self.model = None
                raise Exception(f'Unable to load model {exception}')
    
    def make_prediction(self, image_to_reconstruct):
        self.__image_error_handler(image_to_reconstruct)
        if self.model is not None:
            reconstructed_image = self.model.predict(image_to_reconstruct)
            return reconstructed_image
        else:
            raise Exception('Model is None')

    def reload_model_architecture(self, path):
        try:
            self.__load_model(path)
        except Exception:
            raise Exception("Unable to load model")
    
    def __image_error_handler(self, image):
        if image is None:
            raise InvalidDataException(msg="Input image is None")
        try:
            image_shape = image.shape
        except Exception:
            raise InvalidDataException()
        if len(image_shape) != 4:
            raise InvalidShapeException(msg=f'Invalid data input shape {image_shape} expected 4 dims')
    
    def __get_model_name(self, path):
        if path is not None:
            _, model_type = os.path.split(path)
            model_type = model_type.replace('.h5', '')
            model_type = model_type.split('_')
            self.master.update_model_label(model_type)
        

