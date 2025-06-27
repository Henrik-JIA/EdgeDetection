import numpy as np
from PIL import Image
from scipy.signal import convolve2d

class DifferenceOfGaussianDetector:
    """
    高斯差分检测器
    """
    def __init__(self, image_path=None, sigma=1.0, k=1.6):
        """
        高斯差分(Difference of Gaussian)边缘检测器
    
        高斯差分(Difference of Gaussian)是LoG(拉普拉斯高斯)的近似，通过两个不同sigma值的高斯模糊图像差值计算。
        当sigma2 ≈ 1.6 * sigma1时，DoG是LoG的良好近似。
        """
        self.image_path = image_path
        self.sigma1 = sigma
        self.sigma2 = k * sigma  # 第二个高斯的sigma值
        self.k = k
        self.image_array = None
        self.gaussian_blur_image1 = None  # 第一个高斯模糊图像
        self.gaussian_blur_image2 = None  # 第二个高斯模糊图像
        self.dog_image = None  # 高斯差分结果，这里结果是(k-1)倍的拉普拉斯
        

        if image_path:
            self.image_array = self._load_image(image_path)
            self._process_image()

    def _load_image(self, image_path):
        """加载图像并转换为灰度图"""
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')
        return np.array(image, dtype=np.float32)
    
    def _process_image(self):
        """处理图像，计算高斯差分"""
        # 计算第一个高斯核大小
        ksize1 = int(6 * self.sigma1 + 1)
        if ksize1 % 2 == 0:
            ksize1 += 1
            
        # 计算第二个高斯核大小
        ksize2 = int(6 * self.sigma2 + 1)
        if ksize2 % 2 == 0:
            ksize2 += 1
            
        print(f"DoG参数: sigma1={self.sigma1}, sigma2={self.sigma2}, ksize1={ksize1}, ksize2={ksize2}")

        # 应用第一个高斯模糊（使用自定义函数）
        self.gaussian_blur_image1 = self._get_gaussian_blur_image(
            self.image_array,
            kernel_size=ksize1,
            sigma_x=self.sigma1,
            sigma_y=self.sigma1
        )
        
        # 应用第二个高斯模糊（使用自定义函数）
        self.gaussian_blur_image2 = self._get_gaussian_blur_image(
            self.image_array,
            kernel_size=ksize2,
            sigma_x=self.sigma2,
            sigma_y=self.sigma2
        )
        
        # 计算高斯差分
        self.dog_image = self.gaussian_blur_image1 - self.gaussian_blur_image2
        
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


    def get_gaussian_blur_image1(self):
        """获取第一个高斯模糊图像 (sigma1)"""
        return self.gaussian_blur_image1
    
    def get_gaussian_blur_image2(self):
        """获取第二个高斯模糊图像 (sigma2)"""
        return self.gaussian_blur_image2
    
    def get_original_image(self):
        """获取原始图像"""
        return self.image_array
    
    def get_dog_image(self):
        """获取高斯差分图像"""
        return self.dog_image
    
    def get_zero_crossings(self, threshold=0):
        """
        获取DoG的零交叉点（边缘）
        
        参数:
        threshold (float): 阈值，用于过滤弱边缘
        
        返回:
        np.array: 零交叉点图像，边缘为255，其他为0
        """
        if self.dog_image is None:
            return None
        
        # 创建零交叉点图像
        rows, cols = self.dog_image.shape
        zero_crossings = np.zeros_like(self.dog_image)
        
        # 检测水平和垂直方向的零交叉点
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # 水平方向零交叉
                if (self.dog_image[i, j-1] * self.dog_image[i, j+1] < 0) and \
                   (abs(self.dog_image[i, j-1] - self.dog_image[i, j+1]) > threshold):
                    zero_crossings[i, j] = 255
                # 垂直方向零交叉
                elif (self.dog_image[i-1, j] * self.dog_image[i+1, j] < 0) and \
                     (abs(self.dog_image[i-1, j] - self.dog_image[i+1, j]) > threshold):
                    zero_crossings[i, j] = 255
        
        return zero_crossings