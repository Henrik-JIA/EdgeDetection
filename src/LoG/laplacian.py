import numpy as np
from PIL import Image
from scipy.signal import convolve2d
# import cv2 as cv

# LaplacianEdgeDetector使用方法：
# edge_detector01 = LaplacianEdgeDetector(image_path=img_data['path'], sigma=2.0)
# gaussian_blur_image01 = edge_detector01.get_gaussian_blur_image()
# laplacian01 = edge_detector01.get_laplacian()

class LaplacianEdgeDetector:
    """
    拉普拉斯边缘检测器
    """
    def __init__(self, image_path=None, sigma=1.0):
        """
        初始化拉普拉斯边缘检测器
        
        参数:
        image_path (str): 图像路径
        sigma (float): 高斯滤波器的标准差
        operator_type (str): 梯度算子类型，'sobel'或'prewitt'
        """
        self.image_path = image_path
        self.sigma = sigma
        self.image_array = None
        self.gaussian_blur_image = None
        self.laplacian_image = None
        if image_path:
            self.image_array = self._load_image(image_path)
            # 自适应高斯核大小
            kernel_size = int(6 * self.sigma + 1)
            # kernel_size = 5
            self.gaussian_blur_image = self._get_gaussian_blur_image(self.image_array, kernel_size, self.sigma, self.sigma)
            self.laplacian_image = self._laplacian_process(self.gaussian_blur_image)
            # self.laplacian_image = cv.Laplacian(
            #     self.gaussian_blur_image, 
            #     cv.CV_64F,
            #     ksize=3,
            #     scale=1,
            #     delta=0,
            #     borderType=cv.BORDER_DEFAULT
            # )

    def _load_image(self, image_path):
        """
        加载图像并转换为灰度图
        
        参数:
        image_path (str): 图像路径
        
        返回:
        np.array: 灰度图像数组
        """
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')
            
        return np.array(image, dtype=np.float32)
        
    def _get_laplacian_kernel(self):
        """获取拉普拉斯卷积核"""
        return np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])

    def _get_gaussian_blur_image(self, image_array, kernel_size, sigma_x, sigma_y):
        """
        获取高斯模糊图像，利用高斯函数的可分离性进行优化

        参数:
        image_array (np.array): 图像数组
        kernel_size (int): 卷积核大小
        sigma_x (float): x方向高斯分布的标准差
        sigma_y (float): y方向高斯分布的标准差
        
        返回:
        np.array: 高斯模糊图像
        """
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        half_size = kernel_size // 2
        x = np.arange(-half_size, half_size + 1)
        y = np.arange(-half_size, half_size + 1)

        gaussian_kernel_x = (1 / (sigma_x * np.sqrt(2 * np.pi))) * (np.exp(-(x**2) / (2 * sigma_x**2)))
        gaussian_kernel_y = (1 / (sigma_y * np.sqrt(2 * np.pi))) * (np.exp(-(y**2) / (2 * sigma_y**2)))
    
        # 将一维核转换为二维核 (关键修复)
        kernel_2d_x = gaussian_kernel_x.reshape(1, -1)  # 行向量核 (1, kernel_size)
        kernel_2d_y = gaussian_kernel_y.reshape(-1, 1)  # 列向量核 (kernel_size, 1)
    
        # 先在x方向应用一维高斯核
        temp = convolve2d(
            image_array, 
            kernel_2d_x, 
            mode='same',
            boundary='symm'
        )
        
        # 然后在y方向应用一维高斯核
        blurred_image = convolve2d(
            temp, 
            kernel_2d_y, 
            mode='same',
            boundary='symm'
        )
        
        return blurred_image
    
    def _laplacian_process(self, image_array):
        """
        拉普拉斯处理(高斯二阶导数图形处理)
        """
        LoG_image = convolve2d(
            image_array, 
            self._get_laplacian_kernel(), 
            mode='same', 
            boundary='symm'
        )
        return LoG_image

    def get_original_image(self):
        """
        获取原始图像
        """
        return self.image_array

    def get_laplacian(self):
        """
        获取计算后的拉普拉斯结果
        
        返回:
        np.array: 拉普拉斯边缘检测结果
        """
        return self.laplacian_image

    def get_gaussian_blur_image(self):
        """
        获取高斯模糊图像
        """
        return self.gaussian_blur_image
    
    def get_zero_crossings(self, threshold=0):
        """
        获取拉普拉斯边缘检测的零交叉点（更高效的实现）
        
        参数:
        threshold (float): 阈值，用于过滤弱边缘
        
        返回:
        np.array: 零交叉点图像，边缘为255，其他为0
        """
        if self.laplacian_image is None:
            return None
            
        # 创建零交叉点图像
        zero_crossings = np.zeros_like(self.laplacian_image)
        
        # 获取拉普拉斯图像的尺寸
        rows, cols = self.laplacian_image.shape
        
        # 检测水平方向的零交叉点
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # 检查周围8个方向是否有符号变化
                neighbors = [
                    self.laplacian_image[i-1, j],    # 上
                    self.laplacian_image[i+1, j],    # 下
                    self.laplacian_image[i, j-1],    # 左
                    self.laplacian_image[i, j+1],    # 右
                    self.laplacian_image[i-1, j-1],  # 左上
                    self.laplacian_image[i-1, j+1],  # 右上
                    self.laplacian_image[i+1, j-1],  # 左下
                    self.laplacian_image[i+1, j+1]   # 右下
                ]
                
                center = self.laplacian_image[i, j]
                
                # 检查中心点与任一相邻点是否有符号变化
                for neighbor in neighbors:
                    if center * neighbor < 0 and abs(center - neighbor) > threshold:
                        zero_crossings[i, j] = 255
                        break
        
        return zero_crossings
