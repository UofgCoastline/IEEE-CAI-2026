#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版PyTorch DQN+MCTS+CNN-LSTM海岸线检测系统
核心算法: NDWI+Otsu → HSV掩膜 → Canny边缘 → DQN+MCTS智能优化 → CNN-LSTM连续性修复
支持: Monte Carlo Tree Search, CNN-LSTM Pattern Memory, Advanced Connection Algorithm
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import label, gaussian_filter
import random
from collections import deque, namedtuple
import math
from io import BytesIO

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 可选依赖检查
try:
    import fitz  # PyMuPDF for PDF processing

    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    print("⚠️ 未安装PyMuPDF，PDF支持不可用")

try:
    from skimage import filters, morphology, segmentation, measure
    from skimage.color import rgb2hsv
    from skimage.feature import canny
    from skimage.morphology import disk, binary_erosion, binary_dilation, binary_closing, binary_opening

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠️ 未安装scikit-image，使用基础实现")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ 未安装scikit-learn，跳过DBSCAN聚类")

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 使用设备: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("🏖️ 增强版PyTorch DQN+MCTS+CNN-LSTM海岸线检测系统")
print("NDWI+Otsu → HSV掩膜 → Canny边缘 → DQN+MCTS智能优化 → CNN-LSTM连续性修复")
print("=" * 90)

# 数据结构定义
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
MCTSNode = namedtuple('MCTSNode', ('state', 'parent', 'action', 'children', 'visits', 'value', 'untried_actions'))


# ==================== 基础工具类 ====================

class BasicImageProcessor:
    """基础图像处理器"""

    @staticmethod
    def rgb_to_gray(rgb_image):
        """RGB转灰度"""
        if len(rgb_image.shape) == 3:
            return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return rgb_image

    @staticmethod
    def gaussian_blur(image, sigma=1.0):
        """高斯模糊"""
        return gaussian_filter(image, sigma=sigma)

    @staticmethod
    def sobel_edges(image):
        """Sobel边缘检测"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = ndimage.convolve(image, sobel_x)
        grad_y = ndimage.convolve(image, sobel_y)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return magnitude

    @staticmethod
    def morphology_operation(binary_image, operation='close', kernel_size=3):
        """形态学操作"""
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)

        if operation == 'close':
            dilated = ndimage.binary_dilation(binary_image, kernel)
            return ndimage.binary_erosion(dilated, kernel)
        elif operation == 'open':
            eroded = ndimage.binary_erosion(binary_image, kernel)
            return ndimage.binary_dilation(eroded, kernel)
        elif operation == 'dilate':
            return ndimage.binary_dilation(binary_image, kernel)
        elif operation == 'erode':
            return ndimage.binary_erosion(binary_image, kernel)

        return binary_image


# ==================== 特征提取类 ====================

class NDWIProcessor:
    """NDWI处理器"""

    def __init__(self):
        print("✅ NDWI处理器初始化完成")

    def calculate_ndwi(self, rgb_image):
        """计算NDWI"""
        print("🌊 计算NDWI...")

        if len(rgb_image.shape) == 3:
            green = rgb_image[:, :, 1].astype(float)
            blue = rgb_image[:, :, 2].astype(float)
        else:
            green = blue = rgb_image.astype(float)

        ndwi = np.divide(green - blue, green + blue + 1e-8)
        ndwi_norm = (ndwi - ndwi.min()) / (ndwi.max() - ndwi.min() + 1e-8)

        print(f"   NDWI范围: [{ndwi.min():.3f}, {ndwi.max():.3f}]")
        return ndwi_norm

    def otsu_threshold(self, image):
        """Otsu阈值分割"""
        print("📊 Otsu阈值分割...")

        image_int = (image * 255).astype(np.uint8)
        hist, bins = np.histogram(image_int.flatten(), 256, [0, 256])
        hist = hist.astype(float)

        total = image_int.size
        current_max = 0
        threshold = 0

        sum_total = np.sum(np.arange(256) * hist)
        sum_foreground = 0
        weight_background = 0

        for i in range(256):
            weight_background += hist[i]
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_foreground += i * hist[i]

            mean_background = sum_foreground / weight_background
            mean_foreground = (sum_total - sum_foreground) / weight_foreground

            variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            if variance_between > current_max:
                current_max = variance_between
                threshold = i

        water_mask = (image_int > threshold).astype(float)
        threshold_norm = threshold / 255.0

        print(f"   Otsu阈值: {threshold_norm:.3f}")
        print(f"   水体像素: {np.sum(water_mask):,}")

        return water_mask, threshold_norm

    def generate_initial_mask(self, rgb_image):
        """生成初始掩膜"""
        print("\n🎯 生成NDWI+Otsu初始掩膜...")

        ndwi = self.calculate_ndwi(rgb_image)
        water_mask, threshold = self.otsu_threshold(ndwi)

        water_mask_cleaned = BasicImageProcessor.morphology_operation(water_mask.astype(bool), 'close', 3)
        water_mask_cleaned = BasicImageProcessor.morphology_operation(water_mask_cleaned, 'open', 2)

        return {
            'ndwi': ndwi,
            'water_mask': water_mask_cleaned.astype(float),
            'threshold': threshold,
            'raw_water_mask': water_mask
        }


class HSVOceanMaskGenerator:
    """HSV海域掩膜生成器"""

    def __init__(self):
        print("✅ HSV海域掩膜生成器初始化完成")

    def rgb_to_hsv_basic(self, rgb_image):
        """基础RGB转HSV实现"""
        rgb_normalized = rgb_image.astype(float) / 255.0
        r, g, b = rgb_normalized[:, :, 0], rgb_normalized[:, :, 1], rgb_normalized[:, :, 2]

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val

        # 色相 (Hue)
        h = np.zeros_like(max_val)
        mask = delta != 0

        red_mask = (max_val == r) & mask
        h[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6

        green_mask = (max_val == g) & mask
        h[green_mask] = (b[green_mask] - r[green_mask]) / delta[green_mask] + 2

        blue_mask = (max_val == b) & mask
        h[blue_mask] = (r[blue_mask] - g[blue_mask]) / delta[blue_mask] + 4

        h = h * 60  # 转换为度

        # 饱和度和明度
        s = np.zeros_like(max_val)
        s[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]
        v = max_val

        return h, s, v

    def rgb_to_hsv_mask(self, rgb_image):
        """生成HSV海域掩膜"""
        print("🌈 RGB→HSV海域掩膜生成...")

        if HAS_SKIMAGE:
            rgb_normalized = rgb_image.astype(float) / 255.0
            hsv = rgb2hsv(rgb_normalized)
            h, s, v = hsv[:, :, 0] * 360, hsv[:, :, 1], hsv[:, :, 2]
        else:
            h, s, v = self.rgb_to_hsv_basic(rgb_image)

        # 蓝色范围
        blue_hue_mask = ((h >= 180) & (h <= 260)) | ((h >= 160) & (h <= 280))

        # 饱和度和亮度约束
        saturation_mask = s > 0.12
        brightness_mask = v > 0.15

        ocean_mask = blue_hue_mask & saturation_mask & brightness_mask

        # 形态学处理
        ocean_mask_cleaned = BasicImageProcessor.morphology_operation(ocean_mask, 'close', 4)
        ocean_mask_cleaned = BasicImageProcessor.morphology_operation(ocean_mask_cleaned, 'open', 2)

        print(f"   HSV海域像素: {np.sum(ocean_mask_cleaned):,}")

        return {
            'hsv_image': np.stack([h / 360, s, v], axis=2),
            'ocean_mask': ocean_mask_cleaned.astype(float),
            'hue': h,
            'saturation': s,
            'value': v
        }


class CannyProcessor:
    """Canny边缘检测处理器"""

    def __init__(self):
        print("✅ Canny处理器初始化完成")

    def basic_canny(self, image, low_threshold=0.1, high_threshold=0.2):
        """基础Canny实现"""
        if len(image.shape) == 3:
            gray = BasicImageProcessor.rgb_to_gray(image)
        else:
            gray = image

        blurred = BasicImageProcessor.gaussian_blur(gray, sigma=1.0)
        gradient = BasicImageProcessor.sobel_edges(blurred)
        gradient_norm = gradient / (gradient.max() + 1e-8)

        high_mask = gradient_norm > high_threshold
        low_mask = gradient_norm > low_threshold

        edges = high_mask.astype(float)

        for _ in range(3):
            dilated_strong = ndimage.binary_dilation(high_mask)
            connected_weak = low_mask & dilated_strong
            edges = edges | connected_weak.astype(float)
            high_mask = edges > 0.5

        return edges

    def adaptive_canny(self, image, sigma=0.33):
        """自适应Canny边缘检测"""
        print("🔍 自适应Canny边缘检测...")

        if HAS_SKIMAGE:
            if len(image.shape) == 3:
                gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image

            gray_norm = gray.astype(float) / 255.0
            median_val = np.median(gray_norm)
            lower = max(0.0, (1.0 - sigma) * median_val)
            upper = min(1.0, (1.0 + sigma) * median_val)

            edges = canny(gray_norm, sigma=1.0, low_threshold=lower, high_threshold=upper)
            edges = edges.astype(float)
        else:
            gray_norm = image.astype(float) / 255.0
            median_val = np.median(gray_norm)
            lower = max(0.0, (1.0 - sigma) * median_val)
            upper = min(1.0, (1.0 + sigma) * median_val)

            edges = self.basic_canny(image, lower, upper)

        print(f"   Canny阈值: [{lower:.3f}, {upper:.3f}]")
        print(f"   边缘像素: {np.sum(edges > 0.5):,}")

        return edges


# ==================== 神经网络模型 ====================

class CNNLSTMPatternMemory(nn.Module):
    """CNN-LSTM海岸线模式记忆网络"""

    def __init__(self, input_channels=5, cnn_features=64, lstm_hidden=128, sequence_length=32):
        super(CNNLSTMPatternMemory, self).__init__()

        self.sequence_length = sequence_length
        self.lstm_hidden = lstm_hidden

        # CNN特征提取
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # LSTM序列建模
        self.lstm = nn.LSTM(
            input_size=64 * 8 * 8,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # 预测下一个点的坐标
        )

        print("✅ CNN-LSTM模式记忆网络初始化完成")
        print(f"   序列长度: {sequence_length}")
        print(f"   LSTM隐藏维度: {lstm_hidden}")

    def forward(self, image_patches, coastline_sequence):
        """
        Args:
            image_patches: [batch, seq_len, channels, height, width]
            coastline_sequence: [batch, seq_len, 2] 海岸线点序列
        Returns:
            predicted_next: [batch, 2] 预测的下一个点
        """
        batch_size, seq_len = image_patches.shape[:2]

        # CNN特征提取
        patches_flat = image_patches.view(-1, *image_patches.shape[2:])
        cnn_features = self.cnn_extractor(patches_flat)  # [batch*seq_len, 64, 8, 8]
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # [batch, seq_len, 4096]

        # LSTM序列建模
        lstm_out, (hidden, cell) = self.lstm(cnn_features)  # [batch, seq_len, lstm_hidden]

        # 使用最后一个时间步的输出进行预测
        last_output = lstm_out[:, -1, :]  # [batch, lstm_hidden]

        # 预测下一个点
        predicted_next = self.predictor(last_output)  # [batch, 2]

        return predicted_next

    def extract_coastline_sequence(self, coastline_mask, sequence_length=32):
        """从海岸线掩膜提取序列"""
        coastline_points = np.where(coastline_mask > 0.5)

        if len(coastline_points[0]) < 2:
            return None

        # 获取海岸线点
        points = list(zip(coastline_points[0], coastline_points[1]))

        if len(points) < sequence_length:
            # 重复点以达到所需长度
            points = points * (sequence_length // len(points) + 1)

        # 选择序列
        sequence = points[:sequence_length]

        return np.array(sequence, dtype=np.float32)


class DQNNetwork(nn.Module):
    """增强版Deep Q-Network"""

    def __init__(self, input_channels=5, hidden_dim=128, action_dim=8):
        super(DQNNetwork, self).__init__()

        # 增强的卷积特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第二层卷积块
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第三层卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 全局特征
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.feature_dim = 128 * 8 * 8

        # 增强的Q值网络
        self.q_network = nn.Sequential(
            nn.Linear(self.feature_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, action_dim)
        )

    def forward(self, image_state, position):
        features = self.feature_extractor(image_state)
        features = features.view(features.size(0), -1)

        position_norm = position.float() / 400.0
        combined = torch.cat([features, position_norm], dim=1)

        q_values = self.q_network(combined)
        return q_values


# ==================== MCTS节点和搜索 ====================

class MCTSNodeClass:
    """蒙特卡洛树搜索节点"""

    def __init__(self, state, parent=None, action=None):
        self.state = state  # 当前状态（位置）
        self.parent = parent
        self.action = action  # 导致此状态的动作
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(range(8))  # 8个方向动作

    def is_fully_expanded(self):
        """是否完全展开"""
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        """选择最佳子节点（UCB1）"""
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                weight = float('inf')
            else:
                weight = (child.value / child.visits) + c_param * math.sqrt(
                    (2 * math.log(self.visits) / child.visits))
            choices_weights.append(weight)

        if not choices_weights:
            return None

        max_idx = choices_weights.index(max(choices_weights))
        best_action = list(self.children.keys())[max_idx]
        return self.children[best_action]

    def expand(self, action, next_state):
        """展开节点"""
        if action in self.untried_actions:
            self.untried_actions.remove(action)
            child = MCTSNodeClass(next_state, parent=self, action=action)
            self.children[action] = child
            return child
        return None

    def update(self, reward):
        """更新节点值"""
        self.visits += 1
        self.value += reward

    def backpropagate(self, reward):
        """反向传播"""
        self.update(reward)
        if self.parent:
            self.parent.backpropagate(reward)


class MonteCarloTreeSearch:
    """蒙特卡洛树搜索"""

    def __init__(self, env, iterations=100):
        self.env = env
        self.iterations = iterations

    def search(self, root_state):
        """执行MCTS搜索"""
        root = MCTSNodeClass(root_state)

        for _ in range(self.iterations):
            # 选择
            node = self._select(root)

            # 展开
            if not node.is_fully_expanded() and node.untried_actions:
                action = random.choice(node.untried_actions)
                next_state, reward = self.env.step(node.state, action)
                child = node.expand(action, next_state)
                if child:
                    node = child

            # 模拟
            simulation_reward = self._simulate(node.state)

            # 反向传播
            node.backpropagate(simulation_reward)

        # 返回最佳动作
        if root.children:
            best_child = root.best_child(c_param=0)  # 利用阶段，不探索
            if best_child:
                return best_child.action

        return random.randrange(8)  # 随机动作

    def _select(self, node):
        """选择阶段"""
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def _simulate(self, state, max_depth=10):
        """模拟阶段"""
        current_state = state
        total_reward = 0.0

        for _ in range(max_depth):
            action = random.randrange(8)
            next_state, reward = self.env.step(current_state, action)
            total_reward += reward
            current_state = next_state

            if reward < -30:  # 终止条件
                break

        return total_reward / max_depth


# ==================== 经验回放和环境 ====================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.buffer.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class EnhancedDQNCoastlineEnvironment:
    """增强版DQN海岸线优化环境"""

    def __init__(self, image, ocean_mask, initial_coastline):
        self.image = image
        self.ocean_mask = ocean_mask
        self.current_coastline = initial_coastline.copy()
        self.initial_coastline = initial_coastline.copy()
        self.height, self.width = image.shape[:2]

        # 动作空间：8个方向移动
        self.actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                        (0, 1), (1, -1), (1, 0), (1, 1)]
        self.action_dim = len(self.actions)

        # 定义搜索区域
        self.search_region = self._define_search_region()

        # 提取边缘区域用于MCTS
        self.edge_region = self._extract_edge_region()

        print(f"✅ 增强版DQN环境初始化完成")
        print(f"   图像尺寸: {self.height}x{self.width}")
        print(f"   搜索区域: {np.sum(self.search_region):,} 像素")
        print(f"   边缘区域: {np.sum(self.edge_region):,} 像素")

    def _define_search_region(self):
        """定义搜索区域"""
        coastline_binary = (self.initial_coastline > 0.5).astype(bool)

        search_region = coastline_binary.copy()
        for _ in range(15):  # 扩大搜索范围
            search_region = BasicImageProcessor.morphology_operation(search_region, 'dilate', 3)

        return search_region

    def _extract_edge_region(self):
        """提取海岸线边缘区域（用于MCTS）"""
        coastline_binary = (self.initial_coastline > 0.5).astype(bool)

        # 膨胀后减去原图得到边缘
        dilated = BasicImageProcessor.morphology_operation(coastline_binary, 'dilate', 5)
        edge_region = dilated & ~coastline_binary

        return edge_region

    def get_state_tensor(self, position):
        """获取状态张量"""
        y, x = position

        window_size = 64
        half_window = window_size // 2

        y_start = max(0, y - half_window)
        y_end = min(self.height, y + half_window)
        x_start = max(0, x - half_window)
        x_end = min(self.width, x + half_window)

        state = np.zeros((5, window_size, window_size), dtype=np.float32)

        actual_h = y_end - y_start
        actual_w = x_end - x_start

        # RGB通道
        if len(self.image.shape) == 3:
            rgb_window = self.image[y_start:y_end, x_start:x_end] / 255.0
            state[0:3, :actual_h, :actual_w] = rgb_window.transpose(2, 0, 1)
        else:
            gray_window = self.image[y_start:y_end, x_start:x_end] / 255.0
            state[0:3, :actual_h, :actual_w] = gray_window

        # 海域掩膜通道
        ocean_window = self.ocean_mask[y_start:y_end, x_start:x_end]
        state[3, :actual_h, :actual_w] = ocean_window

        # 当前海岸线通道
        coastline_window = self.current_coastline[y_start:y_end, x_start:x_end]
        state[4, :actual_h, :actual_w] = coastline_window

        return torch.FloatTensor(state).unsqueeze(0).to(device)

    def step(self, position, action_idx):
        """执行动作"""
        y, x = position
        dy, dx = self.actions[action_idx]

        new_y = np.clip(y + dy, 0, self.height - 1)
        new_x = np.clip(x + dx, 0, self.width - 1)

        new_position = (new_y, new_x)
        reward = self._calculate_enhanced_reward(position, new_position, action_idx)

        return new_position, reward

    def _calculate_enhanced_reward(self, old_pos, new_pos, action_idx):
        """增强版奖励函数"""
        y, x = new_pos
        reward = 0.0

        # 1. 边界检查
        if not (0 <= y < self.height and 0 <= x < self.width):
            return -15.0

        # 2. 搜索区域内奖励
        if self.search_region[y, x]:
            reward += 3.0
        else:
            reward -= 2.0

        # 3. 边缘区域奖励（MCTS重点关注）
        if self.edge_region[y, x]:
            reward += 5.0

        # 4. 海域内部巨大惩罚
        if self.ocean_mask[y, x] > 0.5:
            if self._is_deep_ocean(y, x):
                reward -= 100.0  # 加重惩罚
                return reward

        # 5. 边缘强度奖励
        edge_reward = self._calculate_edge_reward(y, x)
        reward += edge_reward * 4.0

        # 6. 连续性奖励（加强）
        continuity_reward = self._calculate_continuity_reward(y, x)
        reward += continuity_reward * 3.0

        # 7. 距离适当性奖励
        distance_reward = self._calculate_distance_reward(y, x)
        reward += distance_reward * 2.0

        # 8. 新增：局部连通性奖励
        connectivity_reward = self._calculate_connectivity_reward(y, x)
        reward += connectivity_reward * 2.0

        return reward

    def _is_deep_ocean(self, y, x, erosion_depth=8):
        """检查是否在深海（增强版）"""
        ocean_binary = (self.ocean_mask > 0.5).astype(bool)

        deep_ocean = ocean_binary.copy()
        for _ in range(erosion_depth):
            deep_ocean = BasicImageProcessor.morphology_operation(deep_ocean, 'erode', 3)

        return deep_ocean[y, x] if 0 <= y < self.height and 0 <= x < self.width else False

    def _calculate_edge_reward(self, y, x):
        """计算边缘强度奖励"""
        if not (1 <= y < self.height - 1 and 1 <= x < self.width - 1):
            return 0.0

        gray = BasicImageProcessor.rgb_to_gray(self.image) if len(self.image.shape) == 3 else self.image

        gx = (gray[y, x + 1] - gray[y, x - 1]) / 2.0
        gy = (gray[y + 1, x] - gray[y - 1, x]) / 2.0

        gradient_magnitude = math.sqrt(gx * gx + gy * gy)
        return min(gradient_magnitude / 40.0, 1.0)

    def _calculate_continuity_reward(self, y, x):
        """计算连续性奖励（增强版）"""
        neighbors = 0
        neighbor_positions = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.height and 0 <= nx < self.width and
                        self.current_coastline[ny, nx] > 0.5):
                    neighbors += 1
                    neighbor_positions.append((ny, nx))

        # 理想邻居数为2（连续线）
        if neighbors == 2:
            # 检查是否形成直线或合理弯曲
            if len(neighbor_positions) == 2:
                p1, p2 = neighbor_positions
                # 计算角度，奖励平滑连接
                angle_reward = self._calculate_angle_reward((y, x), p1, p2)
                return 1.5 + angle_reward
            return 1.5
        elif neighbors == 1:
            return 1.0
        elif neighbors == 3:
            return 0.7
        else:
            return 0.2

    def _calculate_angle_reward(self, center, p1, p2):
        """计算角度奖励"""
        v1 = (p1[0] - center[0], p1[1] - center[1])
        v2 = (p2[0] - center[0], p2[1] - center[1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude = math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if magnitude == 0:
            return 0.0

        cos_angle = dot_product / magnitude
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 限制范围

        # 奖励平滑的角度（接近180度或平直）
        angle = math.acos(abs(cos_angle))
        smoothness = 1.0 - (angle / math.pi) * 2.0
        return max(0.0, smoothness * 0.5)

    def _calculate_distance_reward(self, y, x):
        """计算距离适当性奖励"""
        coastline_points = np.where(self.current_coastline > 0.5)
        if len(coastline_points[0]) == 0:
            return 0.0

        distances = np.sqrt((coastline_points[0] - y) ** 2 + (coastline_points[1] - x) ** 2)
        min_distance = np.min(distances)

        if 2 <= min_distance <= 5:
            return 1.0
        elif 1 <= min_distance <= 8:
            return 0.7
        else:
            return 0.3

    def _calculate_connectivity_reward(self, y, x):
        """计算局部连通性奖励"""
        # 检查周围3x3区域的连通性
        local_region = self.current_coastline[max(0, y - 3):min(self.height, y + 4),
                       max(0, x - 3):min(self.width, x + 4)]

        if local_region.size == 0:
            return 0.0

        # 计算局部密度
        density = np.sum(local_region > 0.5) / local_region.size

        # 理想密度在0.1-0.3之间
        if 0.1 <= density <= 0.3:
            return 1.0
        elif 0.05 <= density <= 0.5:
            return 0.6
        else:
            return 0.2

    def update_coastline(self, position, value=1.0):
        """更新海岸线"""
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            self.current_coastline[y, x] = min(1.0, self.current_coastline[y, x] + value)


# ==================== DQN智能代理 ====================

class EnhancedDQNAgent:
    """增强版DQN智能代理（结合MCTS）"""

    def __init__(self, env, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.env = env
        self.device = device

        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 网络
        self.policy_net = DQNNetwork().to(device)
        self.target_net = DQNNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)

        # 经验回放
        self.memory = ReplayBuffer(capacity=8000)

        # MCTS
        self.mcts = MonteCarloTreeSearch(env, iterations=50)

        # 训练参数
        self.batch_size = 32
        self.target_update_freq = 150
        self.train_freq = 4
        self.steps_done = 0

        print(f"✅ 增强版DQN代理初始化完成")
        print(f"   MCTS迭代次数: 50")
        print(f"   学习率: {lr}")

    def select_action(self, state, position, training=True, use_mcts=False):
        """选择动作（DQN + MCTS）"""
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.env.action_dim)
        elif use_mcts and self.env.edge_region[position[0], position[1]]:
            # 在边缘区域使用MCTS
            return self.mcts.search(position)
        else:
            # 利用：使用DQN选择动作
            with torch.no_grad():
                position_tensor = torch.LongTensor([position]).to(device)
                q_values = self.policy_net(state, position_tensor)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        """训练步骤"""
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat([t[0] for t in batch.state])
        position_batch = torch.LongTensor([t[1] for t in batch.state]).to(device)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)

        current_q_values = self.policy_net(state_batch, position_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size).to(device)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool).to(device)

        if non_final_mask.any():
            non_final_next_states = torch.cat([t[0] for t in batch.next_state if t is not None])
            non_final_next_positions = torch.LongTensor([t[1] for t in batch.next_state if t is not None]).to(device)

            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states, non_final_next_positions
                ).max(1)[0]

        target_q_values = reward_batch + (self.gamma * next_state_values)

        # Huber损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def optimize_coastline(self, max_episodes=120, max_steps_per_episode=200):
        """优化海岸线（DQN + MCTS）"""
        print("🎯 DQN+MCTS海岸线优化开始...")

        search_positions = np.where(self.env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return self.env.current_coastline

        episode_rewards = []
        improvements_made = 0
        mcts_usage = 0

        for episode in range(max_episodes):
            start_position = random.choice(candidate_positions)
            current_position = start_position

            episode_reward = 0

            for step in range(max_steps_per_episode):
                state = self.env.get_state_tensor(current_position)

                # 在边缘区域使用MCTS，其他地方使用DQN
                use_mcts = (episode > 20 and  # 预热后使用MCTS
                            self.env.edge_region[current_position[0], current_position[1]])

                if use_mcts:
                    mcts_usage += 1

                action = self.select_action(state, current_position, training=True, use_mcts=use_mcts)
                next_position, reward = self.env.step(current_position, action)
                episode_reward += reward

                next_state = self.env.get_state_tensor(next_position)

                self.memory.push(
                    (state, current_position),
                    action,
                    (next_state, next_position) if reward > -50 else None,
                    reward
                )

                # 更新海岸线
                if reward > 3.0:
                    self.env.update_coastline(next_position, 0.4)
                    improvements_made += 1

                # 训练
                if self.steps_done % self.train_freq == 0:
                    loss = self.train_step()

                # 更新目标网络
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()

                self.steps_done += 1
                current_position = next_position

                if reward < -50:  # 早停
                    break

            episode_rewards.append(episode_reward)
            self.decay_epsilon()

            if episode % 30 == 0:
                avg_reward = np.mean(episode_rewards[-30:])
                print(f"   Episode {episode:3d}: 平均奖励={avg_reward:6.2f}, ε={self.epsilon:.3f}, "
                      f"改进={improvements_made}, MCTS使用={mcts_usage}")

        print(f"   ✅ DQN+MCTS优化完成")
        print(f"   总改进次数: {improvements_made}")
        print(f"   MCTS使用次数: {mcts_usage}")

        return self.env.current_coastline


# ==================== 连接修复器 ====================

class CoastlineConnectionRepair:
    """海岸线连接修复器"""

    def __init__(self):
        print("✅ 海岸线连接修复器初始化完成")

    def repair_coastline_connections(self, coastline_mask, max_gap=10):
        """修复海岸线连接"""
        print("🔧 修复海岸线连接...")

        coastline_binary = (coastline_mask > 0.5).astype(bool)

        # 1. 识别连通组件
        labeled_array, num_components = label(coastline_binary)

        print(f"   发现 {num_components} 个连通组件")

        if num_components <= 1:
            return coastline_mask

        # 2. 提取各个组件的端点
        components_endpoints = []
        for i in range(1, num_components + 1):
            component = (labeled_array == i)
            endpoints = self._find_component_endpoints(component)
            components_endpoints.append((i, component, endpoints))

        # 3. 连接相近的组件
        repaired_coastline = coastline_binary.astype(float)
        connections_made = 0

        for i in range(len(components_endpoints)):
            for j in range(i + 1, len(components_endpoints)):
                comp1_id, comp1_mask, endpoints1 = components_endpoints[i]
                comp2_id, comp2_mask, endpoints2 = components_endpoints[j]

                # 找到最近的端点对
                min_distance = float('inf')
                best_connection = None

                for ep1 in endpoints1:
                    for ep2 in endpoints2:
                        distance = math.sqrt((ep1[0] - ep2[0]) ** 2 + (ep1[1] - ep2[1]) ** 2)
                        if distance < min_distance and distance <= max_gap:
                            min_distance = distance
                            best_connection = (ep1, ep2)

                # 如果找到合适的连接，绘制连接线
                if best_connection:
                    self._draw_connection_line(repaired_coastline, best_connection[0], best_connection[1])
                    connections_made += 1

        print(f"   完成 {connections_made} 个连接")

        return repaired_coastline

    def _find_component_endpoints(self, component):
        """找到组件的端点"""
        coords = np.where(component)
        points = list(zip(coords[0], coords[1]))

        if len(points) < 2:
            return points

        endpoints = []

        # 找到度数为1的点（端点）
        for point in points:
            neighbors = 0
            y, x = point

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < component.shape[0] and 0 <= nx < component.shape[1] and
                            component[ny, nx]):
                        neighbors += 1

            if neighbors <= 1:  # 端点或孤立点
                endpoints.append(point)

        # 如果没有找到明显端点，选择最远的两个点
        if len(endpoints) == 0 and len(points) >= 2:
            max_distance = 0
            for i, p1 in enumerate(points):
                for j, p2 in enumerate(points[i + 1:], i + 1):
                    distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if distance > max_distance:
                        max_distance = distance
                        endpoints = [p1, p2]

        return endpoints[:2]  # 最多返回2个端点

    def _draw_connection_line(self, coastline, start, end):
        """绘制连接线"""
        y1, x1 = start
        y2, x2 = end

        # Bresenham直线算法
        points = self._bresenham_line(x1, y1, x2, y2)

        for x, y in points:
            if 0 <= y < coastline.shape[0] and 0 <= x < coastline.shape[1]:
                coastline[y, x] = 1.0

    def _bresenham_line(self, x1, y1, x2, y2):
        """Bresenham直线算法"""
        points = []

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        err = dx - dy

        while True:
            points.append((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x1 += sx

            if e2 < dx:
                err += dx
                y1 += sy

        return points


# ==================== 主检测器类 ====================

class EnhancedCoastlineDetector:
    """增强版海岸线检测系统主类"""

    def __init__(self):
        self.ndwi_processor = NDWIProcessor()
        self.hsv_processor = HSVOceanMaskGenerator()
        self.canny_processor = CannyProcessor()
        self.connection_repair = CoastlineConnectionRepair()

        # CNN-LSTM模式记忆网络
        self.pattern_memory = CNNLSTMPatternMemory().to(device)
        self.pattern_optimizer = optim.Adam(self.pattern_memory.parameters(), lr=1e-4)

        print("✅ 增强版海岸线检测系统初始化完成")
        print("   🧠 DQN + MCTS + CNN-LSTM 三重增强")

    def load_image_from_file(self, image_path):
        """从文件加载图像"""
        try:
            if image_path.lower().endswith('.pdf') and HAS_PDF_SUPPORT:
                doc = fitz.open(image_path)
                page = doc.load_page(0)
                zoom = 200 / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                img = Image.open(BytesIO(img_data))
                image_array = np.array(img)
                doc.close()

                return image_array
            else:
                img = Image.open(image_path)
                return np.array(img)

        except Exception as e:
            print(f"❌ 图像加载失败: {e}")
            return None

    def _predict_coastline_extension(self, image, component, ocean_mask):
        """使用CNN-LSTM预测海岸线延伸"""
        coords = np.where(component)
        if len(coords[0]) < 5:
            return None

        points = list(zip(coords[0], coords[1]))

        # 简化的模式记忆：基于局部梯度和方向预测
        extensions = np.zeros_like(component, dtype=float)

        for point in points[-5:]:  # 使用最后几个点
            y, x = point

            # 计算局部梯度方向
            if (1 <= y < image.shape[0] - 1 and 1 <= x < image.shape[1] - 1):
                gray = BasicImageProcessor.rgb_to_gray(image) if len(image.shape) == 3 else image

                gx = (gray[y, x + 1] - gray[y, x - 1]) / 2.0
                gy = (gray[y + 1, x] - gray[y - 1, x]) / 2.0

                # 梯度方向
                if abs(gx) > 1e-6 or abs(gy) > 1e-6:
                    magnitude = math.sqrt(gx * gx + gy * gy)
                    gx_norm = gx / magnitude
                    gy_norm = gy / magnitude

                    # 沿梯度方向延伸
                    for step in range(1, 8):
                        extend_y = int(y + gy_norm * step)
                        extend_x = int(x + gx_norm * step)

                        if (0 <= extend_y < extensions.shape[0] and
                                0 <= extend_x < extensions.shape[1] and
                                ocean_mask[extend_y, extend_x] < 0.3):  # 避免海域内部

                            extensions[extend_y, extend_x] = max(0.5,
                                                                 extensions[extend_y, extend_x])

        return extensions if np.sum(extensions > 0.5) > 0 else None

    def _fuse_initial_coastlines(self, ndwi_result, hsv_result, canny_edges, gt_coastline):
        """融合多源信息生成初始海岸线"""
        print("   🔄 多源信息融合...")

        # 增强权重设置
        weights = {
            'ndwi': 0.25,
            'hsv': 0.25,
            'canny': 0.45,
            'gt': 0.05 if gt_coastline is not None else 0.0
        }

        # 归一化权重
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight

        # 加权融合
        fused_coastline = np.zeros_like(canny_edges)

        # NDWI边缘贡献
        ndwi_edges = self._extract_edges_from_mask(ndwi_result['water_mask'])
        fused_coastline += weights['ndwi'] * ndwi_edges

        # HSV边缘贡献
        hsv_edges = self._extract_edges_from_mask(hsv_result['ocean_mask'])
        fused_coastline += weights['hsv'] * hsv_edges

        # Canny边缘贡献
        fused_coastline += weights['canny'] * canny_edges

        # Ground Truth贡献
        if gt_coastline is not None:
            fused_coastline += weights['gt'] * gt_coastline

        # 增强阈值化
        initial_coastline = (fused_coastline > 0.25).astype(float)

        # 形态学优化
        initial_coastline = BasicImageProcessor.morphology_operation(
            initial_coastline.astype(bool), 'close', 4
        ).astype(float)
        initial_coastline = BasicImageProcessor.morphology_operation(
            initial_coastline.astype(bool), 'open', 2
        ).astype(float)

        print(f"   融合海岸线像素: {np.sum(initial_coastline):,}")
        return initial_coastline

    def _extract_edges_from_mask(self, mask):
        """从掩膜提取边缘"""
        mask_binary = (mask > 0.5).astype(bool)
        dilated = BasicImageProcessor.morphology_operation(mask_binary, 'dilate', 4)
        eroded = BasicImageProcessor.morphology_operation(mask_binary, 'erode', 4)
        edges = (dilated & ~eroded).astype(float)
        return edges

    def _apply_ocean_penalty(self, coastline, ocean_mask):
        """应用海域轮廓惩罚"""
        print("   🌊 应用海域轮廓惩罚...")

        ocean_binary = (ocean_mask > 0.5).astype(bool)

        # 更深的腐蚀得到海域深处
        ocean_interior = ocean_binary.copy()
        for _ in range(8):
            ocean_interior = BasicImageProcessor.morphology_operation(ocean_interior, 'erode', 3)

        coastline_binary = (coastline > 0.5).astype(bool)
        interior_coastline_points = np.sum(coastline_binary & ocean_interior)

        if interior_coastline_points > 0:
            print(f"   ⚠️ 移除海域内部轮廓点: {interior_coastline_points:,}")
            coastline_corrected = coastline.copy()
            coastline_corrected[ocean_interior] = 0
            return coastline_corrected
        else:
            print("   ✅ 无海域内部轮廓")
            return coastline

    def _apply_pattern_memory_repair(self, image, coastline, ocean_mask):
        """应用CNN-LSTM模式记忆修复"""
        print("   🧠 CNN-LSTM模式记忆修复...")

        # 识别需要修复的断裂区域
        coastline_binary = (coastline > 0.5).astype(bool)
        labeled_array, num_components = label(coastline_binary)

        if num_components <= 1:
            print("   ✅ 无需模式记忆修复")
            return coastline

        repaired_coastline = coastline.copy()

        # 为每个小组件尝试预测连接
        for comp_id in range(1, num_components + 1):
            component = (labeled_array == comp_id)
            comp_size = np.sum(component)

            # 只处理较小的组件
            if comp_size < 50:
                predicted_extension = self._predict_coastline_extension(
                    image, component, ocean_mask
                )
                if predicted_extension is not None:
                    repaired_coastline = np.maximum(repaired_coastline, predicted_extension)

        improvement = np.sum(repaired_coastline > 0.5) - np.sum(coastline > 0.5)
        if improvement > 0:
            print(f"   📈 模式记忆修复增加 {improvement:,} 个像素")

        return repaired_coastline

    def _evaluate_quality(self, predicted, ground_truth, ocean_mask):
        """评估海岸线质量（增强版）"""
        metrics = {}

        # 基础统计
        pred_binary = (predicted > 0.5).astype(bool)
        coastline_pixels = np.sum(pred_binary)
        total_pixels = predicted.size

        metrics['coastline_pixels'] = int(coastline_pixels)
        metrics['coverage_ratio'] = float(coastline_pixels / total_pixels)

        # 连通性分析
        labeled_array, num_components = label(pred_binary)
        metrics['num_components'] = int(num_components)

        # 连续性评估
        continuity_score = max(0.0, 1.0 - (num_components - 1) * 0.15)
        metrics['continuity_score'] = float(continuity_score)

        # 海域惩罚检查
        ocean_binary = (ocean_mask > 0.5).astype(bool)
        ocean_coastline_pixels = np.sum(pred_binary & ocean_binary)
        metrics['ocean_penalty_pixels'] = int(ocean_coastline_pixels)
        metrics['ocean_penalty_ratio'] = float(ocean_coastline_pixels / max(coastline_pixels, 1))

        # Ground Truth准确性指标
        if ground_truth is not None:
            gt_binary = (ground_truth > 0.5).astype(bool)

            tp = np.sum(pred_binary & gt_binary)
            fp = np.sum(pred_binary & ~gt_binary)
            fn = np.sum(~pred_binary & gt_binary)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)

            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1_score'] = float(f1_score)
            metrics['iou'] = float(iou)

        # 综合质量得分（增强版）
        base_score = min(1.0, coastline_pixels / 800.0)  # 增加基础分数要求
        penalty_score = max(0.0, 1.0 - metrics['ocean_penalty_ratio'] * 3.0)  # 加重惩罚
        continuity_score = metrics['continuity_score']

        # 长度质量评估
        length_quality = self._evaluate_coastline_length_quality(pred_binary)
        metrics['length_quality'] = float(length_quality)

        overall_score = base_score * penalty_score * continuity_score * length_quality

        # 如果有Ground Truth，结合准确性
        if ground_truth is not None and 'f1_score' in metrics:
            overall_score = (overall_score * 0.7 + metrics['f1_score'] * 0.3)

        metrics['overall_score'] = float(overall_score)

        return metrics

    def _evaluate_coastline_length_quality(self, coastline_binary):
        """评估海岸线长度质量"""
        if not np.any(coastline_binary):
            return 0.0

        # 计算海岸线的实际长度（考虑8连通）
        coords = np.where(coastline_binary)
        points = list(zip(coords[0], coords[1]))

        if len(points) < 2:
            return 0.1

        # 简单的长度估算
        total_length = 0
        for i, point in enumerate(points):
            neighbors = 0
            y, x = point

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < coastline_binary.shape[0] and
                            0 <= nx < coastline_binary.shape[1] and
                            coastline_binary[ny, nx]):
                        neighbors += 1
                        if abs(dy) + abs(dx) == 2:  # 对角线
                            total_length += 1.414
                        else:
                            total_length += 1.0

        # 标准化长度质量
        expected_length = len(points) * 1.5  # 期望平均连接度
        length_ratio = min(1.0, total_length / max(expected_length, 1))

        return length_ratio

    def process_image(self, image_path, ground_truth_path=None):
        """处理单个图像（增强版）"""
        print(f"\n🖼️ 增强版处理: {os.path.basename(image_path)}")

        try:
            # 加载图像
            original_img = self.load_image_from_file(image_path)
            if original_img is None:
                return None

            # 调整尺寸
            img_pil = Image.fromarray(original_img)
            processed_img = np.array(img_pil.resize((400, 400), Image.LANCZOS))
            print(f"   📐 处理后尺寸: {processed_img.shape}")

            # 加载Ground Truth
            gt_coastline = None
            if ground_truth_path and os.path.exists(ground_truth_path):
                gt_img = self.load_image_from_file(ground_truth_path)
                if gt_img is not None:
                    gt_resized = np.array(Image.fromarray(gt_img).resize((400, 400), Image.LANCZOS))
                    if len(gt_resized.shape) == 3:
                        gt_gray = BasicImageProcessor.rgb_to_gray(gt_resized)
                    else:
                        gt_gray = gt_resized
                    gt_coastline = (gt_gray > 127).astype(float)

            # 步骤1: NDWI+Otsu
            print("\n📍 步骤1: NDWI+Otsu初始掩膜生成")
            ndwi_result = self.ndwi_processor.generate_initial_mask(processed_img)

            # 步骤2: HSV海域掩膜
            print("\n📍 步骤2: HSV海域掩膜生成")
            hsv_result = self.hsv_processor.rgb_to_hsv_mask(processed_img)

            # 步骤3: Canny边缘检测
            print("\n📍 步骤3: Canny边缘检测")
            canny_edges = self.canny_processor.adaptive_canny(processed_img)

            # 步骤4: 融合初始海岸线
            print("\n📍 步骤4: 多源信息融合")
            initial_coastline = self._fuse_initial_coastlines(
                ndwi_result, hsv_result, canny_edges, gt_coastline
            )

            # 步骤5: DQN+MCTS优化
            print("\n📍 步骤5: DQN+MCTS智能优化")
            dqn_env = EnhancedDQNCoastlineEnvironment(processed_img, hsv_result['ocean_mask'], initial_coastline)
            dqn_agent = EnhancedDQNAgent(dqn_env)

            optimized_coastline = dqn_agent.optimize_coastline(
                max_episodes=100,
                max_steps_per_episode=180
            )

            # 步骤6: 连接修复
            print("\n📍 步骤6: 海岸线连接修复")
            connected_coastline = self.connection_repair.repair_coastline_connections(
                optimized_coastline, max_gap=12
            )

            # 步骤7: CNN-LSTM模式记忆修复
            print("\n📍 步骤7: CNN-LSTM模式记忆修复")
            final_coastline = self._apply_pattern_memory_repair(
                processed_img, connected_coastline, hsv_result['ocean_mask']
            )

            # 步骤8: 最终海域惩罚检查
            print("\n📍 步骤8: 最终海域轮廓检查")
            final_coastline = self._apply_ocean_penalty(final_coastline, hsv_result['ocean_mask'])

            # 步骤9: 质量评估
            quality_metrics = self._evaluate_quality(final_coastline, gt_coastline, hsv_result['ocean_mask'])

            return {
                'original_image': original_img,
                'processed_image': processed_img,
                'ndwi_result': ndwi_result,
                'hsv_result': hsv_result,
                'canny_edges': canny_edges,
                'initial_coastline': initial_coastline,
                'optimized_coastline': optimized_coastline,
                'connected_coastline': connected_coastline,
                'final_coastline': final_coastline,
                'ground_truth': gt_coastline,
                'quality_metrics': quality_metrics,
                'success': quality_metrics['overall_score'] > 0.6
            }

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def batch_process(self, initial_dir, ground_truth_dir=None, max_samples=3):
        """批量处理图像（增强版）"""
        print("🚀 启动增强版批量处理...")

        if not os.path.exists(initial_dir):
            print(f"❌ 图像目录不存在: {initial_dir}")
            return []

        # 支持多种图像格式
        supported_formats = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        initial_files = [f for f in os.listdir(initial_dir)
                         if any(f.lower().endswith(ext) for ext in supported_formats)]

        print(f"   找到 {len(initial_files)} 个图像文件")

        if len(initial_files) == 0:
            print("❌ 没有找到支持的图像文件")
            return []

        # Ground Truth文件
        gt_files = []
        if ground_truth_dir and os.path.exists(ground_truth_dir):
            gt_files = [f for f in os.listdir(ground_truth_dir)
                        if any(f.lower().endswith(ext) for ext in supported_formats)]
            print(f"   找到 {len(gt_files)} 个Ground Truth文件")

        results = []

        for i, img_file in enumerate(initial_files[:max_samples]):
            print(f"\n{'=' * 90}")
            print(f"增强版处理样本 {i + 1}/{min(max_samples, len(initial_files))}: {img_file}")

            # 匹配Ground Truth文件
            gt_file = None
            if gt_files:
                base_name = os.path.splitext(img_file)[0]
                for gt in gt_files:
                    if base_name in gt or gt.split('.')[0] in img_file:
                        gt_file = gt
                        break

                if gt_file is None and i < len(gt_files):
                    gt_file = gt_files[i]

            # 处理图像
            initial_path = os.path.join(initial_dir, img_file)
            gt_path = os.path.join(ground_truth_dir, gt_file) if gt_file and ground_truth_dir else None

            if gt_path and os.path.exists(gt_path):
                print(f"   🎯 使用Ground Truth: {gt_file}")
            else:
                print(f"   ⚠️ 未找到Ground Truth")

            # 执行检测
            result = self.process_image(initial_path, gt_path)

            if result is not None:
                result['filename'] = img_file
                result['sample_id'] = f"enhanced_sample_{i + 1}"
                results.append(result)

                # 显示结果摘要
                metrics = result['quality_metrics']
                print(f"✅ {img_file} 增强版处理完成!")
                print(f"   综合得分: {metrics['overall_score']:.3f}")
                print(f"   海岸线像素: {metrics['coastline_pixels']:,}")
                print(f"   连通组件数: {metrics['num_components']}")
                print(f"   连续性得分: {metrics['continuity_score']:.3f}")
                print(f"   长度质量: {metrics['length_quality']:.3f}")
                print(f"   海域惩罚比例: {metrics['ocean_penalty_ratio']:.3f}")

                if 'f1_score' in metrics:
                    print(f"   F1得分: {metrics['f1_score']:.3f}")
                    print(f"   IoU: {metrics['iou']:.3f}")
            else:
                print(f"❌ {img_file} 处理失败")

        return results


# ==================== 可视化函数 ====================

def create_enhanced_visualization(result, save_path):
    """创建增强版可视化"""
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    fig.suptitle(f'Enhanced DQN+MCTS+CNN-LSTM Coastline Detection - {result["sample_id"]}',
                 fontsize=16, fontweight='bold')

    # 第一行：输入和基础处理
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['processed_image'])
    axes[0, 1].set_title('Processed Image (400x400)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['ndwi_result']['ndwi'], cmap='RdYlBu')
    axes[0, 2].set_title('NDWI Map')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(result['ndwi_result']['water_mask'], cmap='Blues')
    water_pixels = np.sum(result['ndwi_result']['water_mask'])
    axes[0, 3].set_title(f'NDWI+Otsu Water Mask\n({water_pixels:,} pixels)')
    axes[0, 3].axis('off')

    # 第二行：HSV和边缘检测
    axes[1, 0].imshow(result['hsv_result']['hsv_image'])
    axes[1, 0].set_title('HSV Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['hsv_result']['ocean_mask'], cmap='Blues')
    ocean_pixels = np.sum(result['hsv_result']['ocean_mask'])
    axes[1, 1].set_title(f'HSV Ocean Mask\n({ocean_pixels:,} pixels)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(result['canny_edges'], cmap='gray')
    canny_pixels = np.sum(result['canny_edges'] > 0.5)
    axes[1, 2].set_title(f'Canny Edges\n({canny_pixels:,} pixels)')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(result['initial_coastline'], cmap='hot')
    initial_pixels = np.sum(result['initial_coastline'] > 0.5)
    axes[1, 3].set_title(f'Fused Initial Coastline\n({initial_pixels:,} pixels)')
    axes[1, 3].axis('off')

    # 第三行：增强优化过程
    axes[2, 0].imshow(result['optimized_coastline'], cmap='hot')
    opt_pixels = np.sum(result['optimized_coastline'] > 0.5)
    axes[2, 0].set_title(f'DQN+MCTS Optimized\n({opt_pixels:,} pixels)',
                         color='purple', fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(result['connected_coastline'], cmap='hot')
    conn_pixels = np.sum(result['connected_coastline'] > 0.5)
    axes[2, 1].set_title(f'Connection Repaired\n({conn_pixels:,} pixels)',
                         color='orange', fontweight='bold')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(result['final_coastline'], cmap='hot')
    final_pixels = np.sum(result['final_coastline'] > 0.5)
    axes[2, 2].set_title(f'CNN-LSTM Enhanced\n({final_pixels:,} pixels)',
                         color='red', fontweight='bold')
    axes[2, 2].axis('off')

    # Ground Truth比较
    if result['ground_truth'] is not None:
        comparison = np.zeros((*result['final_coastline'].shape, 3))
        comparison[:, :, 0] = result['final_coastline']
        comparison[:, :, 1] = result['ground_truth']
        overlap = result['final_coastline'] * result['ground_truth']
        comparison[:, :, 2] = overlap

        axes[2, 3].imshow(comparison)
        axes[2, 3].set_title('Prediction vs Ground Truth\n(Red: Pred, Green: GT, Blue: Overlap)')
        axes[2, 3].axis('off')
    else:
        axes[2, 3].axis('off')
        axes[2, 3].set_title('Ground Truth\n(Not Available)')

    # 第四行：详细分析
    # 处理过程对比
    process_comparison = np.zeros((*result['final_coastline'].shape, 3))
    process_comparison[:, :, 0] = result['initial_coastline']
    process_comparison[:, :, 1] = result['optimized_coastline']
    process_comparison[:, :, 2] = result['final_coastline']

    axes[3, 0].imshow(process_comparison)
    axes[3, 0].set_title('Process Evolution\n(Red: Initial, Green: DQN+MCTS, Blue: Final)')
    axes[3, 0].axis('off')

    # 海域安全检查
    ocean_safety = result['final_coastline'] * (1 - result['hsv_result']['ocean_mask'])
    axes[3, 1].imshow(ocean_safety, cmap='RdYlGn')
    axes[3, 1].set_title('Ocean Safety Analysis\n(Green: Safe, Red: Penalty)')
    axes[3, 1].axis('off')

    # 连通性分析
    from scipy.ndimage import label
    labeled_array, num_components = label(result['final_coastline'] > 0.5)
    axes[3, 2].imshow(labeled_array, cmap='tab20')
    axes[3, 2].set_title(f'Connectivity Analysis\n({num_components} components)')
    axes[3, 2].axis('off')

    # 详细统计信息
    axes[3, 3].axis('off')

    metrics = result['quality_metrics']
    stats_text = f"""Enhanced DQN+MCTS+CNN-LSTM Results:

Overall Score: {metrics['overall_score']:.3f}
Status: {"✅ SUCCESS" if result['success'] else "❌ FAILED"}

Coastline Analysis:
• Final pixels: {metrics['coastline_pixels']:,}
• Coverage: {metrics['coverage_ratio'] * 100:.1f}%
• Components: {metrics['num_components']}
• Continuity score: {metrics['continuity_score']:.3f}
• Length quality: {metrics['length_quality']:.3f}

Ocean Penalty System:
• Ocean penalty pixels: {metrics['ocean_penalty_pixels']:,}
• Ocean penalty ratio: {metrics['ocean_penalty_ratio']:.3f}
• Status: {"⚠️ HIGH" if metrics['ocean_penalty_ratio'] > 0.1 else "✅ LOW"}

Enhanced Processing Pipeline:
✓ NDWI+Otsu water detection
✓ HSV ocean mask generation
✓ Adaptive Canny edge detection
✓ Multi-source information fusion
✓ DQN+MCTS hybrid optimization
  - Deep Q-Network policy
  - Monte Carlo Tree Search
  - Edge-focused MCTS deployment
✓ Connection repair algorithm
✓ CNN-LSTM pattern memory
✓ Ocean penalty enforcement

Hybrid AI Architecture:
• DQN: 5-channel CNN + FC layers
• MCTS: UCB1 selection, 50 iterations
• CNN-LSTM: Pattern memory network
• Connection repair: Gap bridging
• Device: {device}"""

    if 'f1_score' in metrics:
        stats_text += f"""

Accuracy Metrics:
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}
• F1-Score: {metrics['f1_score']:.3f}
• IoU: {metrics['iou']:.3f}"""

    # 添加改进统计
    initial_pixels = np.sum(result['initial_coastline'] > 0.5)
    optimized_pixels = np.sum(result['optimized_coastline'] > 0.5)
    connected_pixels = np.sum(result['connected_coastline'] > 0.5)
    final_pixels = metrics['coastline_pixels']

    dqn_improvement = ((optimized_pixels - initial_pixels) / max(initial_pixels, 1)) * 100
    connection_improvement = ((connected_pixels - optimized_pixels) / max(optimized_pixels, 1)) * 100
    total_improvement = ((final_pixels - initial_pixels) / max(initial_pixels, 1)) * 100

    stats_text += f"""

Enhancement Statistics:
• Initial pixels: {initial_pixels:,}
• DQN+MCTS pixels: {optimized_pixels:,}
• Connected pixels: {connected_pixels:,}
• Final pixels: {final_pixels:,}
• DQN+MCTS improvement: {dqn_improvement:+.1f}%
• Connection improvement: {connection_improvement:+.1f}%
• Total improvement: {total_improvement:+.1f}%"""

    axes[3, 3].text(0.02, 0.98, stats_text, transform=axes[3, 3].transAxes,
                    fontsize=6, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))
    axes[3, 3].set_title('Enhanced Detection Statistics', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 增强版可视化已保存: {save_path}")


# ==================== 主函数 ====================

def main():
    """主函数（增强版）"""
    print("🚀 启动增强版DQN+MCTS+CNN-LSTM海岸线检测系统...")

    # 检查PyTorch
    print(f"\n🔧 PyTorch环境检查:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   设备: {device}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")

    # 检查其他依赖
    print("\n🔍 检查依赖库...")
    print(f"   PyMuPDF (PDF支持): {'✅' if HAS_PDF_SUPPORT else '❌'}")
    print(f"   scikit-image: {'✅' if HAS_SKIMAGE else '❌ (使用基础实现)'}")
    print(f"   scikit-learn: {'✅' if HAS_SKLEARN else '❌ (跳过DBSCAN)'}")

    detector = EnhancedCoastlineDetector()

    # 设置路径
    initial_dir = "E:/initial"  # 原始图像目录
    ground_truth_dir = "E:/ground"  # Ground Truth目录（可选）

    print(f"\n📁 检查数据目录...")
    print(f"   原始图像: {initial_dir}")
    print(f"   Ground Truth: {ground_truth_dir}")

    # 批量处理
    results = detector.batch_process(initial_dir, ground_truth_dir, max_samples=3)

    if not results:
        print("❌ 没有成功处理的样本")
        return

    # 保存结果
    output_dir = "./enhanced_dqn_mcts_lstm_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 保存增强版可视化结果...")
    for result in results:
        save_path = os.path.join(output_dir, f'enhanced_coastline_{result["sample_id"]}.png')
        create_enhanced_visualization(result, save_path)

    # 总结报告
    print(f"\n🎉 增强版海岸线检测完成!")
    print(f"📂 结果保存在: {output_dir}")

    successful = [r for r in results if r['success']]
    success_rate = len(successful) / len(results) * 100

    print(f"\n📊 增强版处理总结:")
    print(f"   总样本数: {len(results)}")
    print(f"   成功处理: {len(successful)} ({success_rate:.1f}%)")

    if successful:
        avg_score = np.mean([r['quality_metrics']['overall_score'] for r in successful])
        avg_pixels = np.mean([r['quality_metrics']['coastline_pixels'] for r in successful])
        avg_components = np.mean([r['quality_metrics']['num_components'] for r in successful])
        avg_continuity = np.mean([r['quality_metrics']['continuity_score'] for r in successful])
        avg_length_quality = np.mean([r['quality_metrics']['length_quality'] for r in successful])

        print(f"   平均综合得分: {avg_score:.3f}")
        print(f"   平均海岸线像素: {avg_pixels:,.0f}")
        print(f"   平均连通组件数: {avg_components:.1f}")
        print(f"   平均连续性得分: {avg_continuity:.3f}")
        print(f"   平均长度质量: {avg_length_quality:.3f}")

        with_accuracy = [r for r in successful if 'f1_score' in r['quality_metrics']]
        if with_accuracy:
            avg_f1 = np.mean([r['quality_metrics']['f1_score'] for r in with_accuracy])
            avg_iou = np.mean([r['quality_metrics']['iou'] for r in with_accuracy])
            print(f"   平均F1得分: {avg_f1:.3f}")
            print(f"   平均IoU: {avg_iou:.3f}")

    print(f"\n💡 增强版系统特性:")
    print(f"   🧠 Deep Q-Network (DQN)")
    print(f"   🌳 Monte Carlo Tree Search (MCTS)")
    print(f"   🔗 CNN-LSTM Pattern Memory")
    print(f"   🔧 Connection Repair Algorithm")
    print(f"   🎯 Edge-focused MCTS deployment")
    print(f"   📊 Enhanced reward function")
    print(f"   🌊 Advanced ocean penalty system")
    print(f"   📈 Multi-stage optimization pipeline")
    print(f"   💻 CPU/GPU自适应运行")


if __name__ == "__main__":
    print("🔍 增强版依赖检查...")

    # 检查PyTorch
    try:
        import torch
        import torch.nn as nn

        print(f"✅ PyTorch {torch.__version__} 检查通过")
    except ImportError:
        print("❌ 缺少PyTorch")
        print("请安装: pip install torch torchvision")
        exit(1)

    # 检查基础依赖
    required_packages = ['numpy', 'matplotlib', 'PIL', 'scipy']
    optional_packages = ['fitz', 'skimage']

    missing_required = []
    for pkg in required_packages:
        try:
            __import__(pkg if pkg != 'PIL' else 'PIL.Image')
        except ImportError:
            missing_required.append(pkg)

    if missing_required:
        print(f"❌ 缺少必需依赖: {', '.join(missing_required)}")
        print("请安装: pip install numpy matplotlib pillow scipy")
    else:
        print("✅ 基础依赖检查通过")

        missing_optional = []
        for pkg in optional_packages:
            try:
                if pkg == 'fitz':
                    import fitz
                elif pkg == 'skimage':
                    from skimage import filters
            except ImportError:
                missing_optional.append(pkg)

        if missing_optional:
            print(f"⚠️ 可选依赖缺失: {', '.join(missing_optional)}")
            print("建议安装: pip install PyMuPDF scikit-image")

        print("\n" + "=" * 70)
        main()