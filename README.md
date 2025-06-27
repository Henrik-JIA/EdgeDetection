# EdgeDetection
Implementation of Edge Detection Operators Based on Python

## 项目概述
本项目实现了基于Python的边缘检测算子，包括：
- 拉普拉斯边缘检测器 (LoG)
- 高斯差分边缘检测器 (DoG)

## 主要功能
### 1. 拉普拉斯边缘检测 (LoG)
- 实现高斯模糊预处理
- 拉普拉斯算子计算
- 零交叉点检测
- 支持不同尺度参数(σ)调整

### 2. 高斯差分边缘检测 (DoG)
- 实现两个不同尺度的高斯模糊
- 计算高斯差分结果
- 支持自定义基础尺度(σ)和比例因子(k)
- 提供零交叉点检测功能

## 使用方法
1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/EdgeDetection.git
   cd EdgeDetection
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 运行主程序：
   ```bash
   python main.py
   ```
   默认使用项目中的`data`目录

4. 参数调整：
   在`main.py`中修改以下参数进行实验：
   ```python
   # 拉普拉斯参数
   edge_detector01 = LaplacianEdgeDetector(image_path=img_data['path'], sigma=5.0)
   edge_detector02 = LaplacianEdgeDetector(image_path=img_data['path'], sigma=12.0)
   
   # 高斯差分参数
   dog_detector = DifferenceOfGaussianDetector(image_path=img_data['path'], sigma=5, k=2.6)
   ```

## 示例结果

拉普拉斯算子与高斯差分结果对比：

1. 原始图像（彩色和灰度）
2. 不同尺度的拉普拉斯处理结果
3. 高斯差分处理结果
4. 参数对比（σ值和k值）

![拉普拉斯算子与高斯差分结果对比](./assets/%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%E7%AE%97%E5%AD%90%E4%B8%8E%E9%AB%98%E6%96%AF%E5%B7%AE%E5%88%86%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94.png)

## 技术细节
- **LoG与DoG关系**：当使用参数σ₁和σ₂=k·σ₁时，DoG结果近似于尺度为√(σ₁·σ₂)的LoG结果
- **高效实现**：利用高斯核的可分离性优化卷积计算
- **可视化**：使用Matplotlib展示处理结果对比

## 未来扩展
- 添加更多边缘检测算子（Canny, Sobel等）
- 实现实时视频边缘检测
- 添加参数自动优化功能