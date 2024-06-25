# 电子秤读数识别系统
（模式识别课程设计教学用）

## 环境配置

python 3.9 + opencv + sklearn + scikit-image + matplotlib

## 使用说明

- `collect_traindata.py`用于生成训练数据集，需要
  1. 先执行`save_segmented_images()`，生成检测出的数字图像，对图像进行手工处理，比如删除检测错误的图像;
  2. 再执行`save_labelled_images()`，生成标签数据集,对采用k-means方法初步标注的图像进行手工调整，对错误的类别进行修改。

    具体实现方法：可以新建一个python文件，import该文件，分别执行两个函数。或者直接在该文件的`if __name__ == '__main__':`中分别执行两个函数。
    **注意**：k-means为无监督学习方法，聚类给的标签是随机的，用a、b、c等字母表示，只要同一数字用同一字母表示就行，后续有监督分类时与最终的数字对应上即可。
<br>

- `train_SVM.py`用于训练SVM模型（注意修改k-means生成字母标签和真实数字标签之间的关系，即`label_dict`）
- `inferece.py`用于使用训练好的SVM模型进行预测，注意修改图像路径（最好改成在命令行里输入）
