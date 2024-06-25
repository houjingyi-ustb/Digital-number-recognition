import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize_image(image):
    
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化图像
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary, image

def horizontal_projection(binary_image):
    # 计算水平投影
    projection = np.sum(binary_image, axis=0)  # 黑色像素的数量
    return projection

def segment_by_projection(projection, threshold):
    # 找到投影的边界
    segments = []
    in_segment = False
    start = 0
    
    for i, value in enumerate(projection):
        if value > threshold and not in_segment:
            start = i
            in_segment = True
        elif value <= threshold and in_segment:
            end = i
            in_segment = False
            if end - start > 1:  # 跳过非常窄的分割区域
                segments.append((start, end))
    
    # 处理最后一个分割
    if in_segment:
        segments.append((start, len(projection)))
    
    return segments
def pad_to_fixed_size(binary_image, target_size=(100, 60)):
    # 获取图像的高度和宽度
    height, width = binary_image.shape
    
    # 计算需要添加的上下和左右边距
    pad_height = target_size[0] - height
    pad_width = target_size[1] - width
    
    # 如果图像已经大于或等于目标大小，不进行处理
    if pad_height < 0:  pad_height = 0 
    if pad_width < 0:  pad_width = 0 
    
    # 计算上下和左右边距
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    # 使用numpy.pad进行补零
    padded_image = np.pad(binary_image, ((top, bottom), (left, right)), 'constant', constant_values=0)
    
    return padded_image

def extract_segments(binary_image, segments):
    # 提取分割后的区域
    segmented_images = []
    for start, end in segments:
        segment = binary_image[:, start:end]
        segment = pad_to_fixed_size(segment)
        segmented_images.append(segment)
        
    
    return segmented_images

def segment_number(image):

    binary_image, original_image = binarize_image(image)
    projection = horizontal_projection(binary_image)

    # 设置投影阈值
    threshold = 2000

    segments = segment_by_projection(projection, threshold)
    segmented_images = extract_segments(binary_image, segments)
    # show_results(binary_image, projection, segmented_images)

    return segmented_images

def show_results(binary_image, projection, segmented_images):
    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binarized Image')

    plt.subplot(1, 2, 2)
    plt.plot(projection)
    plt.title('Horizontal Projection')

    plt.figure(figsize=(15, 5))
    for i, segment in enumerate(segmented_images):
        plt.subplot(1, len(segmented_images), i + 1)
        plt.imshow(segment, cmap='gray')
        plt.title(f'Segment {i + 1}')
        plt.axis('off')

    plt.show()
