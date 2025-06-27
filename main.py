import sys
import os
import numpy as np
from src.util.image_loader import ImageLoader
from src.LoG.laplacian import LaplacianEdgeDetector
from src.DoG.difference_gaussian import DifferenceOfGaussianDetector
from PIL import Image  
import matplotlib.pyplot as plt  

def main():
    """
    主函数：加载并处理图片数据
    """
    # 默认使用项目中的data文件夹
    data_path = "data"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"使用自定义路径: {data_path}")
    else:
        print(f"使用默认路径: {data_path}")
    
    try:
        # 创建图片加载器实例，只加载元数据
        loader = ImageLoader(data_path, load_content=False)
        
        # 获取所有图片元数据
        all_image_data = loader.get_image_data()
        image_count = loader.get_image_count()
        
        print(f"\n共加载 {image_count} 张图片的元数据")
        
        # 处理并显示图片元数据
        process_image_data(all_image_data)
        
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        sys.exit(1)

def process_image_data(image_data):
    """
    处理并显示图片数据
    
    参数:
    image_data (list): 包含图片数据的字典列表
    """
    for idx, img_data in enumerate(image_data):
        print(f"\n图片 #{idx+1}: {img_data['path']}")
        print(f"尺寸: {img_data['width']}x{img_data['height']}")
        print(f"格式: {img_data['format']}")
        print(f"大小: {img_data['size'] / 1024:.2f} KB")
        # 显示EXIF信息
        display_exif_info(img_data)
        # 显示GPS信息
        display_gps_info(img_data)

        edge_detector01 = LaplacianEdgeDetector(image_path=img_data['path'], sigma=5)
        gaussian_blur_image01 = edge_detector01.get_gaussian_blur_image()
        laplacian01 = edge_detector01.get_laplacian()

        edge_detector02 = LaplacianEdgeDetector(image_path=img_data['path'], sigma=12)
        gaussian_blur_image02 = edge_detector02.get_gaussian_blur_image()
        laplacian02 = edge_detector02.get_laplacian()

        edge_detector03 = LaplacianEdgeDetector(image_path=img_data['path'], sigma=8.06)
        gaussian_blur_image03 = edge_detector03.get_gaussian_blur_image()
        laplacian03 = edge_detector03.get_laplacian()

        dog_detector = DifferenceOfGaussianDetector(image_path=img_data['path'], sigma=5, k=2.6)
        gaussian_blur_image1 = dog_detector.get_gaussian_blur_image1()
        gaussian_blur_image2 = dog_detector.get_gaussian_blur_image2()
        dog_image = dog_detector.get_dog_image()

        # 可视化原始图像和拉普拉斯结果
        plt.figure(f"图像处理结果: {img_data['path']}", figsize=(15, 8))
        # 第一行：原始图像和第一组处理结果
        # 原始图像
        plt.subplot(4, 3, 1)
        img = Image.open(img_data['path'])
        plt.imshow(img)
        plt.axis('on')
        plt.title('Original')

        # 第一组高斯模糊图像 (sigma=2)
        plt.subplot(4, 3, 2)
        plt.imshow(gaussian_blur_image01, cmap='gray')
        plt.axis('on')
        plt.title(f'Gaussian Blur (σ=5)')

        # 第一组拉普拉斯边缘检测结果
        plt.subplot(4, 3, 3)
        laplacian_abs01 = np.abs(laplacian01)
        if laplacian_abs01.max() > 0:
            laplacian_abs01 = laplacian_abs01 / laplacian_abs01.max()
        plt.imshow(laplacian_abs01, cmap='gray')
        plt.axis('on')
        plt.title('Laplacian (σ=5)')

        # 第二行：第二组处理结果（第一个位置留空或放置其他内容）
        # 原始图像（灰度）
        plt.subplot(4, 3, 4)
        gray_img = img.convert('L')  # 转换为灰度图
        plt.imshow(gray_img, cmap='gray')
        plt.axis('on')
        plt.title('Original (Gray)')

        # 第二组高斯模糊图像 (sigma=9.5)
        plt.subplot(4, 3, 5)
        plt.imshow(gaussian_blur_image02, cmap='gray')
        plt.axis('on')
        plt.title(f'Gaussian Blur (σ=12)')

        # 第二组拉普拉斯边缘检测结果
        plt.subplot(4, 3, 6)
        laplacian_abs02 = np.abs(laplacian02)
        if laplacian_abs02.max() > 0:
            laplacian_abs02 = laplacian_abs02 / laplacian_abs02.max()
        plt.imshow(laplacian_abs02, cmap='gray')
        plt.axis('on')
        plt.title('Laplacian (σ=12)')

        # 第三行：第三组处理结果（第一个位置留空或放置其他内容）
        plt.subplot(4, 3, 7)
        # 可以放置额外的图像或信息，或者保留空白
        plt.axis('off')  # 关闭坐标轴
        plt.title('Additional Info (Optional)')

        # 第三组高斯模糊图像 (sigma=9.5)
        plt.subplot(4, 3, 8)
        plt.imshow(gaussian_blur_image03, cmap='gray')
        plt.axis('on')
        plt.title(f'Gaussian Blur (σ=8.06)')

        # 第三组拉普拉斯边缘检测结果
        plt.subplot(4, 3, 9)
        laplacian_abs03 = np.abs(laplacian03)
        if laplacian_abs03.max() > 0:
            laplacian_abs03 = laplacian_abs03 / laplacian_abs03.max()
        plt.imshow(laplacian_abs03, cmap='gray')
        plt.axis('on')
        plt.title('Laplacian (σ=8.06)')

        # 第四行：高斯差分结果
        # 第四组高斯模糊图像 (sigma=5)
        plt.subplot(4, 3, 10)
        plt.imshow(gaussian_blur_image1, cmap='gray')
        plt.axis('on')
        plt.title(f'Gaussian Blur (σ=5)')

        # 第四组高斯模糊图像 (sigma=5*1.6)
        plt.subplot(4, 3, 11)
        plt.imshow(gaussian_blur_image2, cmap='gray')
        plt.axis('on')
        plt.title(f'Gaussian Blur (σ=5*1.6)')

        # 第四组高斯差分结果
        plt.subplot(4, 3, 12)
        dog_abs = np.abs(dog_image)
        if dog_abs.max() > 0:
            dog_abs = dog_abs / dog_abs.max()
        plt.imshow(dog_abs, cmap='gray')
        plt.axis('on')
        plt.title(f'DoG (σ=5, k=2.6)')

        # 调整布局并显示
        plt.tight_layout()
        plt.show(block=True)  # 阻塞直到窗口关闭
        print("拉普拉斯边缘检测完成")


def display_exif_info(img_data):
    """显示重要的EXIF信息"""
    exif = img_data['exif']
    if not exif:
        print("无EXIF信息")
        return
    
    print("\nEXIF信息:")
    # 显示关键EXIF字段
    keys_to_display = ['DateTime', 'Make', 'Model', 'ExposureTime', 
                       'FNumber', 'ISOSpeedRatings', 'FocalLength']
    
    for key in keys_to_display:
        if key in exif:
            print(f"  {key}: {exif[key]}")
    
    # 显示其他EXIF字段数量
    other_keys = [k for k in exif.keys() if k not in keys_to_display]
    if other_keys:
        print(f"  其他 {len(other_keys)} 个EXIF字段")

def display_gps_info(img_data):
    """显示GPS信息（如果存在）"""
    gps_info = img_data['gps']
    if not gps_info:
        print("无GPS信息")
        return
    
    print("\nGPS信息:")
    # 显示关键GPS字段
    keys_to_display = ['GPSLatitude', 'GPSLongitude', 'GPSAltitude', 
                       'GPSDateStamp', 'GPSTimeStamp']
    
    for key in keys_to_display:
        if key in gps_info:
            print(f"  {key}: {gps_info[key]}")
    
    # 显示其他GPS字段数量
    other_keys = [k for k in gps_info.keys() if k not in keys_to_display]
    if other_keys:
        print(f"  其他 {len(other_keys)} 个GPS字段")

if __name__ == "__main__":
    main()