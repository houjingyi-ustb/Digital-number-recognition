import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_digital import preprocess_image, detect_display_contour, get_transformed_image
from segment_number import segment_number
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.feature import hog


read_dir = 'org_images/'
unlabeled_dir = 'unlabeled_images/'
labeled_dir = 'labeled_images/'
label_name = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def save_segmented_images():
    if not os.path.exists(unlabeled_dir):
        os.makedirs(unlabeled_dir)
    

    for image_name in os.listdir(read_dir):
        image_path = read_dir + image_name
        if not image_name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
            continue
        print(image_path)
        edged_image, color_image = preprocess_image(image_path)
        display_contour, display_image = detect_display_contour(edged_image, color_image)

        if display_contour is not None:
            transformed_image, box = get_transformed_image(color_image, display_contour)
            transformed_image = cv2.resize(transformed_image,(300,100))
            transformed_image = transformed_image[5:-13, 6:-23]
            segmented_images = segment_number(transformed_image)
            for i, segment in enumerate(segmented_images):
                save_name = image_name.split('.')[0] + f'_{i}.jpg'
                cv2.imwrite(unlabeled_dir + save_name, segment)


            # show_image(edged_image, display_image, transformed_image, box)
def save_labelled_images():
    if not os.path.exists(labeled_dir):
        os.makedirs(labeled_dir)
    
    images = []
    image_names = []
    features = []
    for image_name in os.listdir(unlabeled_dir):
        if not image_name.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
            continue
        image_path = unlabeled_dir + image_name
        print(image_path)
        image = cv2.imread(image_path, cv2.THRESH_BINARY)
        images.append(image.flatten())
        image_names.append(image_name.split('.')[0])
        hog_feature = hog(image.reshape(100, 60), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_feature)
    images = np.array(images)
    features = np.array(features)

    # Perform K-means clustering

    # pca = PCA(n_components=50)
    # features = pca.fit_transform(images)

    kmeans = KMeans(n_clusters=10, random_state=42).fit(features)

    clusters = kmeans.labels_

    for i, cluster in enumerate(clusters):
        image_name = image_names[i]
        cluster_name = label_name[cluster]
        save_name = f'{cluster_name}_' + image_name + f'.jpg'
        cv2.imwrite(labeled_dir + save_name, images[i].reshape(100, 60))



        
if __name__ == '__main__':

    # save_segmented_images()
    save_labelled_images()

    # image_path = 'img0.jpg'

    # edged_image, color_image = preprocess_image(image_path)
    # display_contour, display_image = detect_display_contour(edged_image, color_image)

    # if display_contour is not None:
    #     transformed_image, box = get_transformed_image(color_image, display_contour)
    #     transformed_image = cv2.resize(transformed_image,(300,100))
    #     transformed_image = transformed_image[5:-20, 6:-30]
    #     segmented_images = segment_number(transformed_image)


    #     # show_image(edged_image, display_image, transformed_image, box)
    # else:
    #     print("No display contour detected.")