import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import joblib

label_dict = {'a': 1, 'b': 5, 'c': 2, 'd': 3, 'e': 7, 'f': 6, 'g': 9, 'h': 8, 'i': 3, 'j': 4, 'k':0}
data_folder = 'labeled_images/'

def load_dataset(data_folder):
    images = []
    labels = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.jpg'):
            label = label_dict[filename.split('_')[0]]  # 从文件名中提取类别标签
            image_path = os.path.join(data_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            images.append(binary_image)
            labels.append(label)
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

if __name__ == '__main__':
    images, labels = load_dataset(data_folder)
    hog_features = extract_hog_features(images)

    # 训练SVM分类器
    clf = svm.SVC(kernel='linear')
    clf.fit(hog_features, labels)

    # 保存训练好的模型
    joblib.dump(clf, 'svm_model.pkl')
