import cv2
import numpy as np
from detect_digital import preprocess_image, detect_display_contour, get_transformed_image
from segment_number import segment_number
import joblib
from skimage.feature import hog


image_path = 'inference_images/pic5_7.jpg'



def inference():
    
    edged_image, color_image = preprocess_image(image_path)
    display_contour, display_image = detect_display_contour(edged_image, color_image)
    svm_model = joblib.load('svm_model.pkl')

    if display_contour is not None:
        transformed_image, box = get_transformed_image(color_image, display_contour)
        transformed_image = cv2.resize(transformed_image,(300,100))
        transformed_image = transformed_image[5:-13, 6:-23]
        segmented_images = segment_number(transformed_image)
        number_list = ''
        for segment in segmented_images:
            features = hog(segment, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
            prediction = svm_model.predict(features)
            number_list += str(prediction[0])
        read_number = int(number_list)/1000
        print(f"Number read: {read_number} ")



        
if __name__ == '__main__':

    inference()
    # save_labelled_images()
