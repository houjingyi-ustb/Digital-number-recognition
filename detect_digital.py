import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_number import segment_number

def preprocess_image(image_path):
    # 加载图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = image[100:-100, 150:-200]
    # 图像扩展为原来的2倍
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.erode(gray,np.ones((3,3),np.uint8),iterations = 1)

    
    # 二值化图像
    # _, biimage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # laplace锐化
    # blurred = cv2.Laplacian(gray, cv2.CV_8U)
    
    # 使用高斯模糊来减少噪音
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    # alpha = 1  # 控制对比度
    # beta = -10    # 控制亮度
    # # 线性变换
    # blurred = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    # laplacian_img = cv2.Laplacian(blurred, cv2.CV_8U)
    # 将边缘检测结果应用于原始彩色图像，得到最终的清晰化图像
    # blurred = cv2.bitwise_and(blurred, blurred, mask=laplacian_img)

    # 锐化图像
    # blurred = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # 使用Canny边缘检测
    edged = cv2.Canny(blurred, 50, 200)
    # edged = cv2.morphologyEx(edged,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)
    
    return edged, image

def detect_display_contour(edged_image, original_image):
    # 寻找轮廓
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filtered_contours = []
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     ratio = float(w) / h
    #     if 2 <= ratio <= 8:
    #         filtered_contours.append(contour)

    # contours = filtered_contours

    
    # 找到面积最大的轮廓
    if len(contours) == 0:
        return None, original_image
    
    display_contour = max(contours, key=cv2.contourArea)

    
    # 绘制轮廓

    cv2.drawContours(original_image,contours,-1,(0,0,255),3)
    # cv2.drawContours(original_image, [display_contour], -1, (0, 255, 0), 2)
    
    
    return display_contour, original_image

def get_transformed_image(image, contour):
    # 获取轮廓的四个顶点
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # 定义目标矩形的大小
    width = 300
    height = 100
    
    # 目标矩形的四个顶点
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(np.float32(box), dst)
    
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped, box

def show_image(edged_image, display_image, transformed_image, box):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(edged_image, cmap='gray')
    plt.title('Edge Detection')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Display Contour')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.title('Transformed Display')

    plt.show()
    
    print(f"Display contour detected at: {box}")
    
if __name__ == '__main__':

    # image_path = 'org_images/img0.jpg'
    image_path = 'org_images/pic5_8.jpg'

    edged_image, color_image = preprocess_image(image_path)
    display_contour, display_image = detect_display_contour(edged_image, color_image)

    if display_contour is not None:
        transformed_image, box = get_transformed_image(color_image, display_contour)
        transformed_image = cv2.resize(transformed_image,(300,100))
        transformed_image = transformed_image[5:-13, 6:-23]
        segmented_images = segment_number(transformed_image)


        show_image(edged_image, display_image, transformed_image, box)
    else:
        print("No display contour detected.")
        
        
