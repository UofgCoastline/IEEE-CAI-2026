"""
改进的英国城市海岸线检测系统
主要改进：
1. 全图检测（而非仅中间1/3）
2. 边界感知DQN引导
3. 假海岸线过滤
4. 连通性组件分析
5. NDWI/HSV光谱验证
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import label, gaussian_filter, binary_dilation, binary_erosion, binary_closing
import random
from collections import deque, namedtuple
import math
from io import BytesIO
import colorsys

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 可选依赖检查
try:
    import fitz

    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

try:
    from skimage.morphology import skeletonize
    from skimage.filters import sobel

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("🇬🇧 改进的英国城市海岸线检测系统!")
print("主要改进：全图检测 + 边界感知 + 假海岸线过滤")
print("=" * 90)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ==================== 改进的图像处理器 ====================

class ImprovedImageProcessor:
    """改进的图像处理器，支持NDWI和增强边缘检测"""

    @staticmethod
    def rgb_to_gray(rgb_image):
        if len(rgb_image.shape) == 3:
            return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return rgb_image

    @staticmethod
    def calculate_ndwi(rgb_image):
        """计算归一化差分水指数(NDWI)"""
        if len(rgb_image.shape) != 3:
            return np.zeros_like(rgb_image)

        # 模拟绿光和近红外波段
        green = rgb_image[:, :, 1].astype(float)
        nir = rgb_image[:, :, 0].astype(float)  # 使用红色通道近似近红外

        # 避免除零
        denominator = green + nir + 1e-8
        ndwi = (green - nir) / denominator

        return ndwi

    @staticmethod
    def enhanced_edge_detection(image):
        """增强的边缘检测"""
        if len(image.shape) == 3:
            gray = ImprovedImageProcessor.rgb_to_gray(image)
        else:
            gray = image.copy()

        # Gaussian模糊预处理
        blurred = gaussian_filter(gray, sigma=1.0)

        # Sobel边缘检测
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = ndimage.convolve(blurred, sobel_x)
        grad_y = ndimage.convolve(blurred, sobel_y)

        edge_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 如果可用，使用Sobel滤波器
        if HAS_SKIMAGE:
            try:
                edge_skimage = sobel(blurred)
                edge_magnitude = np.maximum(edge_magnitude, edge_skimage * 255)
            except:
                pass

        # 归一化
        if edge_magnitude.max() > edge_magnitude.min():
            edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min())

        return edge_magnitude


# ==================== 边界感知监督器 ====================

class BoundaryAwareHSVSupervisor:
    """边界感知HSV监督器 - 改进版"""

    def __init__(self):
        print("✅ 边界感知HSV监督器初始化完成")
        self.water_hsv_range = self._define_water_hsv_range()
        self.land_hsv_range = self._define_land_hsv_range()
        self.processor = ImprovedImageProcessor()

    def _define_water_hsv_range(self):
        return {
            'hue_range': (180, 240),  # 蓝色范围
            'saturation_min': 0.15,  # 降低饱和度阈值
            'value_min': 0.05  # 降低亮度阈值
        }

    def _define_land_hsv_range(self):
        return {
            'hue_range': (60, 120),  # 绿色范围
            'saturation_min': 0.1,
            'value_min': 0.15
        }

    def analyze_image_hsv(self, rgb_image, gt_analysis=None):
        """分析图像的HSV特征（改进版）"""
        # 计算HSV
        hsv_image = self._rgb_to_hsv(rgb_image)

        # 计算NDWI
        ndwi = self.processor.calculate_ndwi(rgb_image)

        # 增强边缘检测
        edge_map = self.processor.enhanced_edge_detection(rgb_image)

        # 水域和陆地检测
        water_mask = self._enhanced_water_detection(hsv_image, ndwi)
        land_mask = self._enhanced_land_detection(hsv_image, ndwi)

        # 边界置信度图
        boundary_confidence = self._calculate_boundary_confidence(edge_map, water_mask, land_mask)

        # 海岸线指导图
        coastline_guidance = self._generate_enhanced_coastline_guidance(
            water_mask, land_mask, boundary_confidence, edge_map
        )

        # 过渡强度
        transition_strength = self._calculate_enhanced_transition_strength(
            hsv_image, water_mask, land_mask, edge_map
        )

        return {
            'hsv_image': hsv_image,
            'ndwi': ndwi,
            'edge_map': edge_map,
            'water_mask': water_mask,
            'land_mask': land_mask,
            'boundary_confidence': boundary_confidence,
            'coastline_guidance': coastline_guidance,
            'transition_strength': transition_strength
        }

    def _rgb_to_hsv(self, rgb_image):
        """RGB转HSV"""
        if len(rgb_image.shape) == 3:
            rgb_normalized = rgb_image.astype(float) / 255.0
            hsv_image = np.zeros_like(rgb_normalized)

            for i in range(rgb_image.shape[0]):
                for j in range(rgb_image.shape[1]):
                    r, g, b = rgb_normalized[i, j]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_image[i, j] = [h * 360, s, v]
        else:
            hsv_image = np.stack([np.zeros_like(rgb_image),
                                  np.zeros_like(rgb_image),
                                  rgb_image / 255.0], axis=2)
        return hsv_image

    def _enhanced_water_detection(self, hsv_image, ndwi):
        """增强的水域检测，结合HSV和NDWI"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # HSV水域检测
        hue_mask = ((h >= self.water_hsv_range['hue_range'][0]) &
                    (h <= self.water_hsv_range['hue_range'][1]))
        saturation_mask = s >= self.water_hsv_range['saturation_min']
        value_mask = v >= self.water_hsv_range['value_min']

        hsv_water = hue_mask & saturation_mask & value_mask

        # NDWI水域检测
        ndwi_water = ndwi > 0.0  # NDWI > 0 通常表示水域

        # 低饱和度蓝色区域（可能是远海）
        blue_low_sat = ((h >= 200) & (h <= 250)) & (s >= 0.05) & (v >= 0.1)

        # 综合水域掩膜
        water_mask = hsv_water | ndwi_water | blue_low_sat

        # 形态学处理
        water_mask = binary_closing(water_mask, np.ones((7, 7)))
        water_mask = binary_erosion(water_mask, np.ones((3, 3)))
        water_mask = binary_dilation(water_mask, np.ones((5, 5)))

        return water_mask

    def _enhanced_land_detection(self, hsv_image, ndwi):
        """增强的陆地检测"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # 绿色植被
        green_mask = ((h >= self.land_hsv_range['hue_range'][0]) &
                      (h <= self.land_hsv_range['hue_range'][1])) & \
                     (s >= self.land_hsv_range['saturation_min']) & \
                     (v >= self.land_hsv_range['value_min'])

        # 棕色土壤/建筑
        brown_mask = ((h >= 20) & (h <= 60)) & (s >= 0.1) & (v >= 0.2)

        # 灰色建筑/道路
        gray_mask = (s <= 0.15) & (v >= 0.3) & (v <= 0.8)

        # NDWI陆地（NDWI < -0.1 通常表示陆地）
        ndwi_land = ndwi < -0.1

        # 综合陆地掩膜
        land_mask = green_mask | brown_mask | gray_mask | ndwi_land

        # 形态学处理
        land_mask = binary_closing(land_mask, np.ones((5, 5)))
        land_mask = binary_erosion(land_mask, np.ones((2, 2)))
        land_mask = binary_dilation(land_mask, np.ones((4, 4)))

        return land_mask

    def _calculate_boundary_confidence(self, edge_map, water_mask, land_mask):
        """计算边界置信度图"""
        # 水陆边界
        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask

        # 边界候选区域
        boundary_candidates = water_boundary | land_boundary

        # 结合边缘强度
        confidence = edge_map * boundary_candidates.astype(float)

        # 距离变换增强
        from scipy.ndimage import distance_transform_edt

        water_dist = distance_transform_edt(~water_mask)
        land_dist = distance_transform_edt(~land_mask)

        # 在水陆交界处置信度最高
        boundary_distance = np.minimum(water_dist, land_dist)
        distance_weight = np.exp(-boundary_distance / 5.0)

        confidence = confidence + distance_weight * 0.3

        # 归一化
        if confidence.max() > 0:
            confidence = confidence / confidence.max()

        return confidence

    def _generate_enhanced_coastline_guidance(self, water_mask, land_mask, boundary_confidence, edge_map):
        """生成增强的海岸线指导图"""
        # 基础海岸线候选
        water_boundary = binary_dilation(water_mask, np.ones((5, 5))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((5, 5))) & ~land_mask

        coastline_candidates = water_boundary | land_boundary

        # 结合多种信息源
        guidance = coastline_candidates.astype(float) * 0.4  # 基础权重
        guidance += boundary_confidence * 0.4  # 边界置信度
        guidance += edge_map * 0.2  # 边缘强度

        # 距离变换指导
        from scipy.ndimage import distance_transform_edt

        if np.any(water_mask) and np.any(land_mask):
            water_dist = distance_transform_edt(~water_mask)
            land_dist = distance_transform_edt(~land_mask)

            # 在真正的边界处给予最高权重
            boundary_strength = np.exp(-0.1 * (water_dist + land_dist))
            guidance += boundary_strength * 0.3

        # 归一化
        if guidance.max() > 0:
            guidance = guidance / guidance.max()

        return guidance

    def _calculate_enhanced_transition_strength(self, hsv_image, water_mask, land_mask, edge_map):
        """计算增强的过渡强度"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # HSV梯度
        h_grad = np.abs(np.gradient(h)[0]) + np.abs(np.gradient(h)[1])
        s_grad = np.abs(np.gradient(s)[0]) + np.abs(np.gradient(s)[1])
        v_grad = np.abs(np.gradient(v)[0]) + np.abs(np.gradient(v)[1])

        # 组合过渡强度
        transition_strength = (h_grad * 0.3 + s_grad * 0.3 + v_grad * 0.2 + edge_map * 0.2)

        # 在水陆边界处增强
        boundary_mask = binary_dilation(water_mask, np.ones((7, 7))) | binary_dilation(land_mask, np.ones((7, 7)))
        transition_strength = transition_strength * (1 + boundary_mask.astype(float) * 2.0)

        # 归一化
        if transition_strength.max() > transition_strength.min():
            transition_strength = (transition_strength - transition_strength.min()) / \
                                  (transition_strength.max() - transition_strength.min() + 1e-8)

        return transition_strength


# ==================== 改进的约束动作空间 ====================

class ImprovedConstrainedActionSpace:
    """改进的约束动作空间 - 边界感知"""

    def __init__(self):
        self.base_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                             (0, 1), (1, -1), (1, 0), (1, 1)]
        print("✅ 改进的约束动作空间初始化完成")

    def get_allowed_actions(self, current_position, coastline_state, hsv_analysis):
        """获取允许的动作（边界感知）"""
        allowed_actions = []
        context = self._analyze_boundary_context(current_position, coastline_state, hsv_analysis)

        for i, action in enumerate(self.base_actions):
            if self._is_boundary_aware_action_allowed(action, context, current_position, hsv_analysis):
                allowed_actions.append(i)

        return allowed_actions if allowed_actions else [0, 1, 3, 4]

    def _analyze_boundary_context(self, position, coastline_state, hsv_analysis):
        """分析边界上下文"""
        y, x = position

        # 边界置信度
        boundary_confidence = hsv_analysis.get('boundary_confidence', np.zeros_like(coastline_state))
        confidence_score = boundary_confidence[y, x] if 0 <= y < boundary_confidence.shape[0] and 0 <= x < \
                                                        boundary_confidence.shape[1] else 0

        # 局部区域分析
        y_start, y_end = max(0, y - 3), min(coastline_state.shape[0], y + 4)
        x_start, x_end = max(0, x - 3), min(coastline_state.shape[1], x + 4)

        # 水陆分布
        water_mask = hsv_analysis.get('water_mask', np.zeros_like(coastline_state, dtype=bool))
        land_mask = hsv_analysis.get('land_mask', np.zeros_like(coastline_state, dtype=bool))

        local_water = np.sum(water_mask[y_start:y_end, x_start:x_end])
        local_land = np.sum(land_mask[y_start:y_end, x_start:x_end])

        return {
            'confidence_score': confidence_score,
            'is_boundary_region': confidence_score > 0.2,
            'water_nearby': local_water > 0,
            'land_nearby': local_land > 0,
            'is_transition_zone': local_water > 0 and local_land > 0
        }

    def _is_boundary_aware_action_allowed(self, action, context, current_position, hsv_analysis):
        """边界感知的动作允许检查"""
        dy, dx = action

        # 如果不在边界区域，限制大幅度移动
        if not context['is_boundary_region']:
            if abs(dy) + abs(dx) > 1:
                return False

        # 如果在过渡区域，允许更灵活的移动
        if context['is_transition_zone']:
            return True

        # 如果置信度很低，限制移动
        if context['confidence_score'] < 0.1:
            if abs(dy) > 1 or abs(dx) > 1:
                return False

        return True


# ==================== 假海岸线过滤器 ====================

class FalseCoastlineFilter:
    """假海岸线过滤器"""

    def __init__(self):
        print("✅ 假海岸线过滤器初始化完成")

    def filter_false_coastlines(self, coastline_result, hsv_analysis, original_image):
        """过滤假海岸线"""
        filtered_coastline = coastline_result.copy()

        # 1. 连通组件分析过滤
        filtered_coastline = self._filter_by_connected_components(filtered_coastline)

        # 2. NDWI/HSV光谱验证
        filtered_coastline = self._filter_by_spectral_verification(
            filtered_coastline, hsv_analysis, original_image
        )

        # 3. 海洋一致性过滤
        filtered_coastline = self._filter_by_ocean_coherence(filtered_coastline, hsv_analysis)

        # 4. 边界邻近性过滤
        filtered_coastline = self._filter_by_boundary_proximity(filtered_coastline, hsv_analysis)

        return filtered_coastline

    def _filter_by_connected_components(self, coastline):
        """基于连通组件的过滤"""
        binary_coastline = (coastline > 0.5).astype(bool)
        labeled_array, num_components = label(binary_coastline)

        if num_components == 0:
            return coastline

        # 计算每个组件的大小
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            component_sizes.append((i, size))

        # 按大小排序
        component_sizes.sort(key=lambda x: x[1], reverse=True)

        # 保留较大的组件
        filtered_binary = np.zeros_like(binary_coastline)
        total_pixels = np.sum(binary_coastline)

        for component_id, size in component_sizes:
            # 保留大于总像素1%的组件，或者前5大组件
            if size > total_pixels * 0.01 or len([c for c in component_sizes[:5] if c[0] == component_id]) > 0:
                filtered_binary[labeled_array == component_id] = True

        # 转换回概率值
        filtered_coastline = coastline * filtered_binary.astype(float)

        return filtered_coastline

    def _filter_by_spectral_verification(self, coastline, hsv_analysis, original_image):
        """基于光谱特征的验证过滤"""
        binary_coastline = (coastline > 0.5).astype(bool)

        # 获取NDWI
        ndwi = hsv_analysis.get('ndwi', np.zeros_like(coastline))

        # 获取水域掩膜
        water_mask = hsv_analysis.get('water_mask', np.zeros_like(coastline, dtype=bool))

        # 过滤在深水区域的假海岸线
        # 深水区域：NDWI > 0.3 且在水域掩膜内
        deep_water = (ndwi > 0.3) & water_mask

        # 对深水区域内的海岸线进行膨胀处理，然后移除
        deep_water_expanded = binary_dilation(deep_water, np.ones((5, 5)))

        # 移除深水区域内的海岸线
        filtered_binary = binary_coastline & ~deep_water_expanded

        # 转换回概率值
        filtered_coastline = coastline * filtered_binary.astype(float)

        return filtered_coastline

    def _filter_by_ocean_coherence(self, coastline, hsv_analysis):
        """基于海洋一致性的过滤"""
        binary_coastline = (coastline > 0.5).astype(bool)
        water_mask = hsv_analysis.get('water_mask', np.zeros_like(coastline, dtype=bool))

        # 从高置信度的水域像素开始反向膨胀
        high_confidence_water = water_mask.copy()

        # 多次膨胀，标记连续的水域
        for _ in range(10):
            expanded_water = binary_dilation(high_confidence_water, np.ones((3, 3)))
            high_confidence_water = expanded_water & water_mask

        # 移除完全被高置信度水域包围的海岸线
        surrounded_by_water = binary_coastline.copy()
        for _ in range(3):
            eroded = binary_erosion(surrounded_by_water, np.ones((3, 3)))
            surrounded_by_water = eroded & high_confidence_water

        # 过滤被水域包围的海岸线
        filtered_binary = binary_coastline & ~surrounded_by_water

        # 转换回概率值
        filtered_coastline = coastline * filtered_binary.astype(float)

        return filtered_coastline

    def _filter_by_boundary_proximity(self, coastline, hsv_analysis):
        """基于边界邻近性的过滤"""
        binary_coastline = (coastline > 0.5).astype(bool)
        boundary_confidence = hsv_analysis.get('boundary_confidence', np.zeros_like(coastline))

        # 只保留边界置信度较高区域附近的海岸线
        high_boundary_regions = boundary_confidence > 0.1

        # 膨胀高边界置信度区域
        expanded_boundary = high_boundary_regions.copy()
        for _ in range(5):
            expanded_boundary = binary_dilation(expanded_boundary, np.ones((3, 3)))

        # 过滤远离边界的海岸线
        filtered_binary = binary_coastline & expanded_boundary

        # 转换回概率值
        filtered_coastline = coastline * filtered_binary.astype(float)

        return filtered_coastline


# ==================== 改进的海岸线环境 ====================

class ImprovedCoastlineEnvironment:
    """改进的海岸线环境 - 全图检测"""

    def __init__(self, image, gt_analysis=None):
        self.image = image
        self.gt_analysis = gt_analysis
        self.current_coastline = np.zeros(image.shape[:2], dtype=float)
        self.height, self.width = image.shape[:2]

        # 使用改进的监督器
        self.hsv_supervisor = BoundaryAwareHSVSupervisor()
        self.hsv_analysis = self.hsv_supervisor.analyze_image_hsv(image, gt_analysis)

        # 使用改进的动作约束
        self.action_constraints = ImprovedConstrainedActionSpace()
        self.base_actions = self.action_constraints.base_actions
        self.action_dim = len(self.base_actions)

        # 假海岸线过滤器
        self.false_filter = FalseCoastlineFilter()

        # 好奇心探索
        self.curiosity_explorer = CuriosityDrivenExploration()

        # 增强边缘检测
        self.edge_map = self.hsv_analysis['edge_map']

        # 设置全图搜索区域（而非仅中间1/3）
        self._setup_full_image_search_region()

        print(f"✅ 改进的海岸线环境初始化完成（全图检测模式）")

    def _setup_full_image_search_region(self):
        """设置全图搜索区域"""
        # 基于边界置信度的搜索区域
        boundary_confidence = self.hsv_analysis['boundary_confidence']
        coastline_guidance = self.hsv_analysis['coastline_guidance']

        # 主要搜索区域：边界置信度 > 0.05 或 海岸线指导 > 0.1
        primary_region = (boundary_confidence > 0.05) | (coastline_guidance > 0.1)

        # 扩展搜索区域
        expanded_region = primary_region.copy()
        for _ in range(3):
            expanded_region = binary_dilation(expanded_region, np.ones((3, 3)))

        # 避免深水区域（基于NDWI和水域掩膜）
        ndwi = self.hsv_analysis['ndwi']
        water_mask = self.hsv_analysis['water_mask']

        # 深水区域：NDWI > 0.4 且连续的大片水域
        deep_water = (ndwi > 0.4) & water_mask
        for _ in range(5):
            deep_water = binary_erosion(deep_water, np.ones((3, 3)))
        for _ in range(8):
            deep_water = binary_dilation(deep_water, np.ones((3, 3)))

        # 最终搜索区域：扩展区域减去深水区域
        self.search_region = expanded_region & ~deep_water

        # 确保搜索区域不为空
        if not np.any(self.search_region):
            print("   ⚠️ 搜索区域为空，使用全图作为搜索区域")
            self.search_region = np.ones((self.height, self.width), dtype=bool)

        search_ratio = np.sum(self.search_region) / (self.height * self.width)
        print(f"   📍 搜索区域覆盖: {search_ratio:.1%} 的图像")

    def get_state_tensor(self, position):
        """获取状态张量"""
        y, x = position
        window_size = 64
        half_window = window_size // 2

        y_start = max(0, y - half_window)
        y_end = min(self.height, y + half_window)
        x_start = max(0, x - half_window)
        x_end = min(self.width, x + half_window)

        # RGB状态
        rgb_state = np.zeros((3, window_size, window_size), dtype=np.float32)
        actual_h = y_end - y_start
        actual_w = x_end - x_start

        if len(self.image.shape) == 3:
            rgb_window = self.image[y_start:y_end, x_start:x_end] / 255.0
            rgb_state[:, :actual_h, :actual_w] = rgb_window.transpose(2, 0, 1)
        else:
            gray_window = self.image[y_start:y_end, x_start:x_end] / 255.0
            rgb_state[0, :actual_h, :actual_w] = gray_window
            rgb_state[1, :actual_h, :actual_w] = gray_window
            rgb_state[2, :actual_h, :actual_w] = gray_window

        # HSV增强状态
        hsv_state = np.zeros((3, window_size, window_size), dtype=np.float32)

        # 边界置信度
        boundary_window = self.hsv_analysis['boundary_confidence'][y_start:y_end, x_start:x_end]
        hsv_state[0, :actual_h, :actual_w] = boundary_window

        # 海岸线指导
        guidance_window = self.hsv_analysis['coastline_guidance'][y_start:y_end, x_start:x_end]
        hsv_state[1, :actual_h, :actual_w] = guidance_window

        # NDWI
        ndwi_window = self.hsv_analysis['ndwi'][y_start:y_end, x_start:x_end]
        # 归一化NDWI到[0,1]
        ndwi_normalized = (ndwi_window + 1) / 2
        hsv_state[2, :actual_h, :actual_w] = ndwi_normalized

        rgb_tensor = torch.FloatTensor(rgb_state).unsqueeze(0).to(device)
        hsv_tensor = torch.FloatTensor(hsv_state).unsqueeze(0).to(device)

        return rgb_tensor, hsv_tensor

    def get_enhanced_features(self, position):
        """获取增强特征"""
        y, x = position

        if not (0 <= y < self.height and 0 <= x < self.width):
            return torch.zeros(30, dtype=torch.float32, device=device).unsqueeze(0)

        features = np.zeros(30, dtype=np.float32)

        # 基础特征
        features[0] = self.edge_map[y, x]
        features[1] = self.hsv_analysis['boundary_confidence'][y, x]
        features[2] = self.hsv_analysis['coastline_guidance'][y, x]
        features[3] = self.hsv_analysis['transition_strength'][y, x]
        features[4] = (self.hsv_analysis['ndwi'][y, x] + 1) / 2  # 归一化NDWI
        features[5] = 1.0 if self.hsv_analysis['water_mask'][y, x] else 0.0
        features[6] = 1.0 if self.hsv_analysis['land_mask'][y, x] else 0.0

        # 局部区域分析
        y_start, y_end = max(0, y - 3), min(self.height, y + 4)
        x_start, x_end = max(0, x - 3), min(self.width, x + 4)

        # 边界置信度统计
        local_boundary = self.hsv_analysis['boundary_confidence'][y_start:y_end, x_start:x_end]
        if local_boundary.size > 0:
            features[7] = np.mean(local_boundary)
            features[8] = np.max(local_boundary)
            features[9] = np.std(local_boundary)

        # 海岸线指导统计
        local_guidance = self.hsv_analysis['coastline_guidance'][y_start:y_end, x_start:x_end]
        if local_guidance.size > 0:
            features[10] = np.mean(local_guidance)
            features[11] = np.max(local_guidance)

        # NDWI统计
        local_ndwi = self.hsv_analysis['ndwi'][y_start:y_end, x_start:x_end]
        if local_ndwi.size > 0:
            features[12] = np.mean(local_ndwi)
            features[13] = np.min(local_ndwi)
            features[14] = np.max(local_ndwi)

        # 水陆邻近性
        local_water = self.hsv_analysis['water_mask'][y_start:y_end, x_start:x_end]
        local_land = self.hsv_analysis['land_mask'][y_start:y_end, x_start:x_end]

        features[15] = np.sum(local_water) / local_water.size
        features[16] = np.sum(local_land) / local_land.size

        # 好奇心奖励
        curiosity_bonus = self.curiosity_explorer.get_curiosity_bonus(
            position, self.hsv_analysis, self.current_coastline
        )
        features[17] = min(1.0, curiosity_bonus / 50.0)

        # 位置特征
        features[18] = y / self.height
        features[19] = x / self.width

        # 距离中心的距离
        center_y, center_x = self.height // 2, self.width // 2
        distance_to_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_distance = np.sqrt(center_y ** 2 + center_x ** 2)
        features[20] = distance_to_center / max_distance

        # 边缘方向特征
        if y > 0 and y < self.height - 1 and x > 0 and x < self.width - 1:
            sobel_x = self.edge_map[y - 1:y + 2, x - 1:x + 2] * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = self.edge_map[y - 1:y + 2, x - 1:x + 2] * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            grad_x = np.sum(sobel_x)
            grad_y = np.sum(sobel_y)

            if grad_x != 0 or grad_y != 0:
                angle = np.arctan2(grad_y, grad_x)
                features[21] = (angle + np.pi) / (2 * np.pi)  # 归一化到[0,1]
            else:
                features[21] = 0.5

        # 搜索区域特征
        features[22] = 1.0 if self.search_region[y, x] else 0.0

        # 局部变异性
        if len(self.image.shape) == 3:
            local_rgb = self.image[y_start:y_end, x_start:x_end]
            if local_rgb.size > 0:
                features[23] = np.std(local_rgb[:, :, 0]) / 255.0
                features[24] = np.std(local_rgb[:, :, 1]) / 255.0
                features[25] = np.std(local_rgb[:, :, 2]) / 255.0

        # 现有海岸线密度
        local_coastline = self.current_coastline[y_start:y_end, x_start:x_end]
        if local_coastline.size > 0:
            features[26] = np.mean(local_coastline > 0.3)

        # 边界类型判断
        water_nearby = np.any(local_water)
        land_nearby = np.any(local_land)

        if water_nearby and land_nearby:
            features[27] = 1.0  # 过渡区域
        elif water_nearby:
            features[28] = 1.0  # 水域区域
        elif land_nearby:
            features[29] = 1.0  # 陆地区域

        return torch.FloatTensor(features).unsqueeze(0).to(device)

    def step(self, position, action_idx):
        """执行动作步骤"""
        # 获取边界感知的允许动作
        allowed_actions = self.action_constraints.get_allowed_actions(
            position, self.current_coastline, self.hsv_analysis
        )

        if action_idx not in allowed_actions:
            action_idx = allowed_actions[0] if allowed_actions else 0

        y, x = position
        dy, dx = self.base_actions[action_idx]

        new_y = np.clip(y + dy, 0, self.height - 1)
        new_x = np.clip(x + dx, 0, self.width - 1)

        new_position = (new_y, new_x)
        reward = self._calculate_boundary_aware_reward(position, new_position, action_idx)

        return new_position, reward

    def _calculate_boundary_aware_reward(self, old_pos, new_pos, action_idx):
        """计算边界感知奖励"""
        y, x = new_pos
        reward = 0.0

        if not (0 <= y < self.height and 0 <= x < self.width):
            return -100.0

        # 边界置信度奖励
        boundary_confidence = self.hsv_analysis['boundary_confidence'][y, x]
        reward += boundary_confidence * 50.0

        # 海岸线指导奖励
        guidance_score = self.hsv_analysis['coastline_guidance'][y, x]
        reward += guidance_score * 40.0

        # NDWI奖励：在海陆交界处NDWI应该接近0
        ndwi_value = self.hsv_analysis['ndwi'][y, x]
        ndwi_reward = max(0, 20.0 - abs(ndwi_value) * 30.0)  # NDWI接近0时奖励最高
        reward += ndwi_reward

        # 搜索区域奖励
        if self.search_region[y, x]:
            reward += 15.0
        else:
            reward -= 25.0

        # 水陆分离奖励（改进版）
        separation_reward = self._calculate_improved_separation_reward(new_pos)
        reward += separation_reward

        # 边缘质量奖励
        edge_strength = self.edge_map[y, x]
        reward += edge_strength * 25.0

        # 避免深水区域
        if self.hsv_analysis['water_mask'][y, x] and self.hsv_analysis['ndwi'][y, x] > 0.3:
            reward -= 30.0

        return reward

    def _calculate_improved_separation_reward(self, position):
        """计算改进的水陆分离奖励"""
        y, x = position

        water_mask = self.hsv_analysis['water_mask']
        land_mask = self.hsv_analysis['land_mask']

        water_neighbors = 0
        land_neighbors = 0
        total_neighbors = 0

        # 更大的邻域检查
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    total_neighbors += 1
                    if water_mask[ny, nx]:
                        water_neighbors += 1
                    if land_mask[ny, nx]:
                        land_neighbors += 1

        if total_neighbors == 0:
            return 0.0

        water_ratio = water_neighbors / total_neighbors
        land_ratio = land_neighbors / total_neighbors

        # 理想的海岸线应该同时邻近水域和陆地
        if water_ratio > 0.2 and land_ratio > 0.2:
            # 完美的分离：水陆比例接近
            balance_bonus = 50.0 * (1.0 - abs(water_ratio - land_ratio))
            separation_reward = 40.0 + balance_bonus
        elif water_ratio > 0.1 or land_ratio > 0.1:
            separation_reward = 20.0 * (water_ratio + land_ratio)
        else:
            separation_reward = -10.0

        return separation_reward

    def update_coastline(self, position, value=1.0):
        """更新海岸线"""
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            self.current_coastline[y, x] = min(1.0, self.current_coastline[y, x] + value)

    def apply_false_coastline_filtering(self):
        """应用假海岸线过滤"""
        self.current_coastline = self.false_filter.filter_false_coastlines(
            self.current_coastline, self.hsv_analysis, self.image
        )
        return self.current_coastline


# ==================== 改进的DQN网络 ====================

class ImprovedConstrainedCoastlineDQN(nn.Module):
    """改进的约束海岸线DQN网络"""

    def __init__(self, input_channels=3, hidden_dim=256, action_dim=8):
        super(ImprovedConstrainedCoastlineDQN, self).__init__()

        # RGB特征提取器
        self.rgb_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # 边界感知特征提取器
        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.feature_dim = 128 * 8 * 8 + 96 * 8 * 8

        # Q值网络
        self.q_network = nn.Sequential(
            nn.Linear(self.feature_dim + 2 + 30, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # 边界感知动作掩膜网络
        self.boundary_mask_network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, action_dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_state, boundary_state, position, enhanced_features):
        # 特征提取
        rgb_features = self.rgb_extractor(rgb_state)
        boundary_features = self.boundary_extractor(boundary_state)

        # 展平特征
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        boundary_features = boundary_features.view(boundary_features.size(0), -1)

        # 位置归一化
        position_norm = position.float() / 400.0

        # 组合所有特征
        combined = torch.cat([rgb_features, boundary_features, position_norm, enhanced_features], dim=1)

        # Q值计算
        q_values = self.q_network(combined)

        # 边界感知动作掩膜
        action_mask = self.boundary_mask_network(enhanced_features)

        # 应用掩膜
        masked_q_values = q_values * action_mask - (1 - action_mask) * 1e6

        return masked_q_values


# ==================== 好奇心驱动探索（保持不变） ====================

class CuriosityDrivenExploration:
    def __init__(self, exploration_decay=0.995):
        self.visit_history = {}
        self.exploration_decay = exploration_decay
        self.step_count = 0
        print("✅ 好奇心驱动探索机制初始化完成")

    def get_curiosity_bonus(self, position, hsv_analysis, current_coastline):
        y, x = position
        pos_key = f"{y}_{x}"

        visit_count = self.visit_history.get(pos_key, 0)
        visit_bonus = max(0, 10.0 - visit_count * 2.0)

        # 边界感知好奇心奖励
        boundary_bonus = 0.0
        if hsv_analysis:
            boundary_confidence = hsv_analysis.get('boundary_confidence', np.zeros_like(current_coastline))
            if boundary_confidence[y, x] > 0.2:
                boundary_bonus = boundary_confidence[y, x] * 20.0

        self.visit_history[pos_key] = visit_count + 1
        self.step_count += 1

        return visit_bonus + boundary_bonus


# ==================== 改进的代理类 ====================

class ImprovedCoastlineAgent:
    """改进的海岸线代理"""

    def __init__(self, env, lr=1e-4, gamma=0.98, epsilon_start=0.1, epsilon_end=0.05, epsilon_decay=0.995):
        self.env = env
        self.device = device

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 使用改进的网络
        self.policy_net = ImprovedConstrainedCoastlineDQN().to(device)
        self.target_net = ImprovedConstrainedCoastlineDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = deque(maxlen=20000)

        self.batch_size = 32
        self.target_update_freq = 100
        self.train_freq = 4
        self.steps_done = 0

        print(f"✅ 改进的DQN代理初始化完成")

    def select_action(self, rgb_state, boundary_state, position, enhanced_features, training=False):
        """选择动作"""
        allowed_actions = self.env.action_constraints.get_allowed_actions(
            position, self.env.current_coastline, self.env.hsv_analysis
        )

        if training and random.random() < self.epsilon:
            return random.choice(allowed_actions)
        else:
            with torch.no_grad():
                position_tensor = torch.LongTensor([position]).to(device)
                q_values = self.policy_net(rgb_state, boundary_state, position_tensor, enhanced_features)

                # 在允许的动作中选择Q值最高的
                masked_q_values = q_values.clone()
                for i in range(self.env.action_dim):
                    if i not in allowed_actions:
                        masked_q_values[0, i] = float('-inf')

                return masked_q_values.argmax(dim=1).item()

    def load_model(self, load_path):
        """加载预训练模型"""
        if os.path.exists(load_path):
            try:
                checkpoint = torch.load(load_path, map_location=device)
                # 尝试加载改进的模型结构
                try:
                    self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                    self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                except:
                    # 如果结构不匹配，创建兼容的加载方式
                    print("   ⚠️ 模型结构不完全匹配，尝试部分加载...")
                    model_dict = self.policy_net.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint['policy_net_state_dict'].items()
                                       if k in model_dict and v.size() == model_dict[k].size()}
                    model_dict.update(pretrained_dict)
                    self.policy_net.load_state_dict(model_dict)
                    self.target_net.load_state_dict(model_dict)

                self.epsilon = self.epsilon_end
                print(f"✅ 改进的预训练模型已加载: {load_path}")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        return False

    def apply_improved_inference(self, max_inference_steps=1500):
        """应用改进的推理算法"""
        print("🔮 使用改进的预训练模型进行全图海岸线推理...")

        # 获取全图搜索区域
        search_positions = np.where(self.env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return self.env.current_coastline

        print(f"   🎯 全图搜索位置数: {len(candidate_positions)}")

        # 基于边界置信度的智能位置选择
        high_priority_positions = []
        medium_priority_positions = []
        low_priority_positions = []

        for pos in candidate_positions:
            y, x = pos
            boundary_confidence = self.env.hsv_analysis['boundary_confidence'][y, x]
            guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
            edge_score = self.env.edge_map[y, x]

            # 综合评分
            combined_score = boundary_confidence * 0.4 + guidance_score * 0.4 + edge_score * 0.2

            if combined_score > 0.6:
                high_priority_positions.append((combined_score, pos))
            elif combined_score > 0.3:
                medium_priority_positions.append((combined_score, pos))
            else:
                low_priority_positions.append((combined_score, pos))

        # 按优先级排序
        high_priority_positions.sort(reverse=True, key=lambda x: x[0])
        medium_priority_positions.sort(reverse=True, key=lambda x: x[0])
        low_priority_positions.sort(reverse=True, key=lambda x: x[0])

        print(f"   📊 高优先级位置: {len(high_priority_positions)}")
        print(f"   📊 中优先级位置: {len(medium_priority_positions)}")
        print(f"   📊 低优先级位置: {len(low_priority_positions)}")

        # 构建推理序列
        inference_positions = []

        # 优先处理高优先级位置
        inference_positions.extend([pos for _, pos in high_priority_positions[:max_inference_steps // 2]])

        # 补充中优先级位置
        remaining_slots = max_inference_steps - len(inference_positions)
        inference_positions.extend([pos for _, pos in medium_priority_positions[:remaining_slots // 2]])

        # 补充低优先级位置
        remaining_slots = max_inference_steps - len(inference_positions)
        inference_positions.extend([pos for _, pos in low_priority_positions[:remaining_slots]])

        print(f"   🎯 最终推理位置数: {len(inference_positions)}")

        # 多阶段推理
        improvements = 0
        total_reward = 0.0

        for stage in range(3):
            print(f"   🔄 第 {stage + 1} 阶段推理")
            stage_positions = inference_positions[stage::3]  # 交错分配
            stage_improvements = 0

            for position in stage_positions:
                # 获取状态
                rgb_state, boundary_state = self.env.get_state_tensor(position)
                enhanced_features = self.env.get_enhanced_features(position)

                # 推理动作
                action = self.select_action(rgb_state, boundary_state, position, enhanced_features, training=False)

                # 执行动作
                next_position, reward = self.env.step(position, action)
                total_reward += reward

                # 动态阈值
                if stage == 0:
                    reward_threshold = 15.0
                elif stage == 1:
                    reward_threshold = 10.0
                else:
                    reward_threshold = 7.0

                if reward > reward_threshold:
                    # 根据奖励调整更新值
                    update_value = min(1.0, reward / 50.0)
                    self.env.update_coastline(next_position, update_value)
                    improvements += 1
                    stage_improvements += 1

            print(f"      ✅ 第 {stage + 1} 阶段改进: {stage_improvements}")

        # 应用假海岸线过滤
        print("   🧹 应用假海岸线过滤...")
        filtered_coastline = self.env.apply_false_coastline_filtering()

        final_pixels = np.sum(filtered_coastline > 0.3)
        avg_reward = total_reward / len(inference_positions) if inference_positions else 0

        print(f"   ✅ 改进推理完成: {final_pixels:,} 像素, 总改进: {improvements}")
        print(f"   📊 平均奖励: {avg_reward:.2f}")

        return filtered_coastline


# ==================== 改进的质量评估器 ====================

class ImprovedQualityAssessor:
    """改进的质量评估器"""

    def __init__(self):
        print("✅ 改进的质量评估器初始化完成")

    def assess_coastline_quality(self, coastline, hsv_analysis, original_image):
        """评估海岸线质量（改进版）"""
        print("📊 评估改进的海岸线质量...")

        metrics = {}
        pred_binary = (coastline > 0.5).astype(bool)
        coastline_pixels = np.sum(pred_binary)

        # 基础统计
        metrics['coastline_pixels'] = int(coastline_pixels)

        # 1. 连通性分析（改进）
        labeled_array, num_components = label(pred_binary)
        metrics['num_components'] = int(num_components)

        if num_components > 0:
            component_sizes = [np.sum(labeled_array == i) for i in range(1, num_components + 1)]
            main_component_ratio = max(component_sizes) / coastline_pixels if coastline_pixels > 0 else 0

            # 改进的碎片化评分
            size_variance = np.var(component_sizes) / (np.mean(component_sizes) ** 2 + 1e-8)
            metrics['main_component_ratio'] = float(main_component_ratio)
            metrics['fragmentation_score'] = float(min(1.0, size_variance))
        else:
            metrics['main_component_ratio'] = 0.0
            metrics['fragmentation_score'] = 1.0

        # 2. 边界质量评估
        boundary_quality = self._assess_boundary_quality(pred_binary, hsv_analysis)
        metrics['boundary_quality'] = float(boundary_quality)

        # 3. NDWI一致性评估
        ndwi_consistency = self._assess_ndwi_consistency(pred_binary, hsv_analysis)
        metrics['ndwi_consistency'] = float(ndwi_consistency)

        # 4. 假海岸线检测
        false_coastline_ratio = self._detect_false_coastlines(pred_binary, hsv_analysis)
        metrics['false_coastline_ratio'] = float(false_coastline_ratio)

        # 5. 海域清理效果
        water_mask = hsv_analysis['water_mask']
        water_intrusion = np.sum(pred_binary & water_mask) / (coastline_pixels + 1e-8)
        metrics['water_intrusion_ratio'] = float(water_intrusion)
        metrics['sea_cleanup_score'] = float(max(0.0, 1.0 - water_intrusion * 3))

        # 6. 全图分布分析（而非仅中间1/3）
        height = pred_binary.shape[0]
        quarter_height = height // 4

        top_pixels = np.sum(pred_binary[:quarter_height, :])
        upper_mid_pixels = np.sum(pred_binary[quarter_height:2 * quarter_height, :])
        lower_mid_pixels = np.sum(pred_binary[2 * quarter_height:3 * quarter_height, :])
        bottom_pixels = np.sum(pred_binary[3 * quarter_height:, :])

        if coastline_pixels > 0:
            top_ratio = top_pixels / coastline_pixels
            upper_mid_ratio = upper_mid_pixels / coastline_pixels
            lower_mid_ratio = lower_mid_pixels / coastline_pixels
            bottom_ratio = bottom_pixels / coastline_pixels

            # 评估分布均匀性
            distribution_entropy = self._calculate_distribution_entropy(
                [top_ratio, upper_mid_ratio, lower_mid_ratio, bottom_ratio])
            distribution_score = distribution_entropy / np.log(4)  # 归一化熵
        else:
            top_ratio = upper_mid_ratio = lower_mid_ratio = bottom_ratio = 0.0
            distribution_score = 0.0

        metrics['top_ratio'] = float(top_ratio)
        metrics['upper_mid_ratio'] = float(upper_mid_ratio)
        metrics['lower_mid_ratio'] = float(lower_mid_ratio)
        metrics['bottom_ratio'] = float(bottom_ratio)
        metrics['distribution_score'] = float(distribution_score)

        # 7. 密度合理性评估（调整为英国海岸线特征）
        target_min, target_max = 8000, 80000  # 适应英国城市海岸线
        if target_min <= coastline_pixels <= target_max:
            density_score = 1.0
        elif coastline_pixels < target_min:
            density_score = max(0.2, coastline_pixels / target_min)
        else:
            density_score = max(0.1, 1.0 - (coastline_pixels - target_max) / target_max)
        metrics['density_score'] = float(density_score)

        # 8. 连续性评估（改进）
        continuity_score = self._assess_improved_continuity(pred_binary)
        metrics['continuity_score'] = float(continuity_score)

        # 9. 边缘一致性
        edge_consistency = self._assess_edge_consistency(pred_binary, original_image)
        metrics['edge_consistency'] = float(edge_consistency)

        # 10. 综合质量评分（改进版）
        overall_score = self._calculate_improved_overall_score(metrics)
        metrics['overall_score'] = float(overall_score)

        # 11. 质量等级评定
        quality_level = self._determine_improved_quality_level(overall_score)
        metrics['quality_level'] = quality_level

        return metrics

    def _assess_boundary_quality(self, coastline_binary, hsv_analysis):
        """评估边界质量"""
        if not np.any(coastline_binary):
            return 0.0

        boundary_confidence = hsv_analysis.get('boundary_confidence', np.zeros_like(coastline_binary))
        coastline_positions = np.where(coastline_binary)

        if len(coastline_positions[0]) == 0:
            return 0.0

        boundary_values = boundary_confidence[coastline_positions]
        return np.mean(boundary_values)

    def _assess_ndwi_consistency(self, coastline_binary, hsv_analysis):
        """评估NDWI一致性"""
        if not np.any(coastline_binary):
            return 0.0

        ndwi = hsv_analysis.get('ndwi', np.zeros_like(coastline_binary))
        coastline_positions = np.where(coastline_binary)

        if len(coastline_positions[0]) == 0:
            return 0.0

        ndwi_values = ndwi[coastline_positions]

        # 海岸线的NDWI应该接近0（水陆交界）
        ndwi_consistency = np.mean(1.0 - np.abs(ndwi_values))
        return max(0.0, ndwi_consistency)

    def _detect_false_coastlines(self, coastline_binary, hsv_analysis):
        """检测假海岸线比例"""
        if not np.any(coastline_binary):
            return 0.0

        water_mask = hsv_analysis.get('water_mask', np.zeros_like(coastline_binary, dtype=bool))
        ndwi = hsv_analysis.get('ndwi', np.zeros_like(coastline_binary))

        # 深水区域中的海岸线被认为是假的
        deep_water = water_mask & (ndwi > 0.3)
        false_coastlines = coastline_binary & deep_water

        total_coastline = np.sum(coastline_binary)
        false_coastline_count = np.sum(false_coastlines)

        return false_coastline_count / (total_coastline + 1e-8)

    def _calculate_distribution_entropy(self, ratios):
        """计算分布熵"""
        ratios = np.array(ratios)
        ratios = ratios[ratios > 0]  # 移除零值
        if len(ratios) == 0:
            return 0.0
        ratios = ratios / np.sum(ratios)  # 归一化
        return -np.sum(ratios * np.log(ratios + 1e-8))

    def _assess_improved_continuity(self, coastline_binary):
        """评估改进的连续性"""
        if not np.any(coastline_binary):
            return 0.0

        # 使用骨架化评估连续性
        try:
            if HAS_SKIMAGE:
                skeleton = skeletonize(coastline_binary)
                skeleton_pixels = np.sum(skeleton)
                total_pixels = np.sum(coastline_binary)

                # 计算连续性指标
                if total_pixels > 0:
                    skeleton_ratio = skeleton_pixels / total_pixels
                    continuity = min(1.0, skeleton_ratio * 3)  # 调整系数
                else:
                    continuity = 0.0
            else:
                continuity = self._simple_continuity_assessment(coastline_binary)
        except:
            continuity = self._simple_continuity_assessment(coastline_binary)

        return continuity

    def _simple_continuity_assessment(self, coastline_binary):
        """简化的连续性评估"""
        height, width = coastline_binary.shape

        # 行连续性
        row_continuity = 0.0
        valid_rows = 0

        for y in range(height):
            row = coastline_binary[y, :]
            if np.any(row):
                valid_rows += 1
                # 计算连续段
                segments = 0
                in_segment = False
                for x in range(width):
                    if row[x] and not in_segment:
                        segments += 1
                        in_segment = True
                    elif not row[x]:
                        in_segment = False

                # 理想情况是每行1个连续段
                row_continuity += 1.0 / (segments + 1e-8)

        if valid_rows > 0:
            row_continuity /= valid_rows

        # 列连续性
        col_continuity = 0.0
        valid_cols = 0

        for x in range(width):
            col = coastline_binary[:, x]
            if np.any(col):
                valid_cols += 1
                segments = 0
                in_segment = False
                for y in range(height):
                    if col[y] and not in_segment:
                        segments += 1
                        in_segment = True
                    elif not col[y]:
                        in_segment = False

                col_continuity += 1.0 / (segments + 1e-8)

        if valid_cols > 0:
            col_continuity /= valid_cols

        # 综合连续性
        overall_continuity = (row_continuity + col_continuity) / 2.0
        return min(1.0, overall_continuity)

    def _assess_edge_consistency(self, coastline_binary, original_image):
        """评估边缘一致性"""
        if not np.any(coastline_binary):
            return 0.0

        # 计算图像边缘
        processor = ImprovedImageProcessor()
        edge_map = processor.enhanced_edge_detection(original_image)

        # 海岸线位置的边缘强度
        coastline_positions = np.where(coastline_binary)
        if len(coastline_positions[0]) == 0:
            return 0.0

        edge_values = edge_map[coastline_positions]
        return np.mean(edge_values)

    def _calculate_improved_overall_score(self, metrics):
        """计算改进的综合得分"""
        score = 0.0

        # 调整权重分配
        weights = {
            'boundary_quality': 0.20,  # 边界质量
            'ndwi_consistency': 0.15,  # NDWI一致性
            'sea_cleanup_score': 0.15,  # 海域清理
            'distribution_score': 0.12,  # 分布均匀性
            'continuity_score': 0.12,  # 连续性
            'edge_consistency': 0.10,  # 边缘一致性
            'density_score': 0.08,  # 密度合理性
        }

        # 加权计算
        score += metrics.get('boundary_quality', 0) * weights['boundary_quality']
        score += metrics.get('ndwi_consistency', 0) * weights['ndwi_consistency']
        score += metrics.get('sea_cleanup_score', 0) * weights['sea_cleanup_score']
        score += metrics.get('distribution_score', 0) * weights['distribution_score']
        score += metrics.get('continuity_score', 0) * weights['continuity_score']
        score += metrics.get('edge_consistency', 0) * weights['edge_consistency']
        score += metrics.get('density_score', 0) * weights['density_score']

        # 惩罚项
        # 假海岸线惩罚
        false_coastline_penalty = metrics.get('false_coastline_ratio', 0) * 0.3
        score -= false_coastline_penalty

        # 过度碎片化惩罚
        fragmentation_penalty = min(0.15, metrics.get('fragmentation_score', 0) * 0.2)
        score -= fragmentation_penalty

        # 过多连通组件惩罚
        component_count = metrics.get('num_components', 0)
        pixel_count = metrics.get('coastline_pixels', 0)

        if pixel_count > 0:
            reasonable_components = max(50, pixel_count // 500)  # 每500像素允许1个组件
            if component_count > reasonable_components:
                component_penalty = min(0.1, (component_count - reasonable_components) / reasonable_components * 0.15)
                score -= component_penalty

        # 奖励项
        # 主要组件比例奖励
        main_component_ratio = metrics.get('main_component_ratio', 0)
        if main_component_ratio > 0.8:
            score += 0.05

        # 低水域入侵奖励
        water_intrusion = metrics.get('water_intrusion_ratio', 1.0)
        if water_intrusion < 0.1:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _determine_improved_quality_level(self, score):
        """确定改进的质量等级"""
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.70:
            return "Good"
        elif score >= 0.55:
            return "Fair"
        elif score >= 0.40:
            return "Poor"
        else:
            return "Very Poor"


# ==================== 改进的英国城市检测器 ====================

class ImprovedUKCitiesDetector:
    """改进的英国城市海岸线检测器"""

    def __init__(self):
        self.quality_assessor = ImprovedQualityAssessor()
        print("✅ 改进的英国城市海岸线检测器初始化完成")
        print("   🎯 特色：全图检测 + 边界感知 + 假海岸线过滤")

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

    def process_uk_city_improved(self, image_path, city_name, pretrained_model_path):
        """
        处理英国城市海岸线检测（改进版）

        Args:
            image_path: 城市图像路径
            city_name: 城市名称
            pretrained_model_path: 预训练模型路径
        """
        print(f"\n🏴󠁧󠁢󠁥󠁮󠁧󠁿 改进版处理英国城市: {city_name}")
        print(f"📁 图像路径: {image_path}")

        try:
            # 1. 加载图像
            original_img = self.load_image_from_file(image_path)
            if original_img is None:
                return None

            # 调整尺寸
            img_pil = Image.fromarray(original_img)
            processed_img = np.array(img_pil.resize((400, 400), Image.LANCZOS))
            print(f"   📐 处理后尺寸: {processed_img.shape}")

            # 2. 创建改进的环境
            print("\n📍 步骤1: 创建改进的检测环境（全图模式）")
            env = ImprovedCoastlineEnvironment(processed_img, gt_analysis=None)

            # 3. 创建改进的代理并加载模型
            print("\n📍 步骤2: 加载改进的预训练模型")
            agent = ImprovedCoastlineAgent(env)

            if not agent.load_model(pretrained_model_path):
                print(f"❌ 无法加载预训练模型: {pretrained_model_path}")
                return None

            # 4. 执行改进的推理
            print("\n📍 步骤3: 执行改进的海岸线推理")
            coastline_result = agent.apply_improved_inference(max_inference_steps=1200)

            # 5. 改进的质量评估
            print("\n📍 步骤4: 改进的质量评估")
            quality_metrics = self.quality_assessor.assess_coastline_quality(
                coastline_result, env.hsv_analysis, processed_img
            )

            # 6. 结果打包
            result = {
                'city_name': city_name,
                'original_image': original_img,
                'processed_image': processed_img,
                'hsv_analysis': env.hsv_analysis,
                'coastline_result': coastline_result,
                'quality_metrics': quality_metrics,
                'success': quality_metrics['overall_score'] > 0.5,  # 提高成功阈值
                'model_path': pretrained_model_path,
                'improvements': [
                    'Full image detection (not just middle 1/3)',
                    'Boundary-aware DQN guidance',
                    'False coastline filtering',
                    'NDWI spectral validation',
                    'Enhanced edge detection',
                    'Improved connectivity analysis'
                ]
            }

            # 显示改进的结果摘要
            self._display_improved_result_summary(city_name, quality_metrics)

            return result

        except Exception as e:
            print(f"❌ 处理 {city_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _display_improved_result_summary(self, city_name, metrics):
        """显示改进的结果摘要"""
        print(f"\n📊 {city_name} 改进检测结果摘要:")
        print(f"   🎯 综合得分: {metrics['overall_score']:.3f}")
        print(f"   📏 海岸线像素: {metrics['coastline_pixels']:,}")
        print(f"   🏆 质量等级: {metrics['quality_level']}")

        print(f"\n   📈 改进指标:")
        print(f"      🔍 边界质量: {metrics['boundary_quality']:.3f}")
        print(f"      🌊 NDWI一致性: {metrics['ndwi_consistency']:.3f}")
        print(f"      🧹 海域清理: {metrics['sea_cleanup_score']:.3f}")
        print(f"      📍 分布评分: {metrics['distribution_score']:.3f}")
        print(f"      🔗 连续性: {metrics['continuity_score']:.3f}")
        print(f"      ⚡ 边缘一致性: {metrics['edge_consistency']:.3f}")
        print(f"      ❌ 假海岸线比例: {metrics['false_coastline_ratio']:.1%}")

        print(f"\n   🗺️ 全图分布:")
        print(f"      上部: {metrics['top_ratio']:.1%}")
        print(f"      中上: {metrics['upper_mid_ratio']:.1%}")
        print(f"      中下: {metrics['lower_mid_ratio']:.1%}")
        print(f"      下部: {metrics['bottom_ratio']:.1%}")

        if metrics['overall_score'] > 0.7:
            print(f"   ✅ {city_name} 改进检测优秀!")
        elif metrics['overall_score'] > 0.5:
            print(f"   ✅ {city_name} 改进检测良好")
        else:
            print(f"   ⚠️ {city_name} 改进检测仍需优化")


# ==================== 改进的可视化函数 ====================

def create_improved_uk_visualization(result, save_path):
    """创建改进的英国城市海岸线检测可视化"""
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    city_name = result['city_name']
    fig.suptitle(f'Improved UK City Coastline Detection - {city_name}',
                 fontsize=18, fontweight='bold')

    # 第一行：原图和基础分析
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title(f'{city_name} - Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['processed_image'])
    axes[0, 1].set_title('Processed Image (400x400)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['hsv_analysis']['edge_map'], cmap='gray')
    axes[0, 2].set_title('Enhanced Edge Detection')
    axes[0, 2].axis('off')

    ndwi_display = (result['hsv_analysis']['ndwi'] + 1) / 2  # 归一化显示
    axes[0, 3].imshow(ndwi_display, cmap='RdYlBu')
    axes[0, 3].set_title('NDWI Map')
    axes[0, 3].axis('off')

    # 第二行：边界感知分析
    axes[1, 0].imshow(result['hsv_analysis']['boundary_confidence'], cmap='hot')
    axes[1, 0].set_title('Boundary Confidence')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['hsv_analysis']['coastline_guidance'], cmap='plasma')
    axes[1, 1].set_title('Enhanced Coastline Guidance')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(result['hsv_analysis']['water_mask'], cmap='Blues')
    axes[1, 2].set_title('Enhanced Water Detection')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(result['hsv_analysis']['land_mask'], cmap='Greens')
    axes[1, 3].set_title('Enhanced Land Detection')
    axes[1, 3].axis('off')

    # 第三行：检测结果
    coastline_binary = (result['coastline_result'] > 0.5).astype(float)
    axes[2, 0].imshow(coastline_binary, cmap='Reds')
    pixels = np.sum(coastline_binary)
    axes[2, 0].set_title(f'Detected Coastline\n({pixels:,} pixels)')
    axes[2, 0].axis('off')

    # 叠加显示
    overlay = result['processed_image'].copy()
    coastline_coords = np.where(coastline_binary)
    if len(coastline_coords[0]) > 0:
        overlay[coastline_coords[0], coastline_coords[1]] = [255, 0, 0]
    axes[2, 1].imshow(overlay)
    axes[2, 1].set_title('Coastline Overlay')
    axes[2, 1].axis('off')

    # 连通组件分析
    labeled_coastline, num_components = label(coastline_binary)
    axes[2, 2].imshow(labeled_coastline, cmap='tab20')
    axes[2, 2].set_title(f'Connected Components\n({num_components} components)')
    axes[2, 2].axis('off')

    # 假海岸线检测
    water_mask = result['hsv_analysis']['water_mask']
    ndwi = result['hsv_analysis']['ndwi']
    deep_water = water_mask & (ndwi > 0.3)
    false_coastlines = coastline_binary.astype(bool) & deep_water
    axes[2, 3].imshow(false_coastlines.astype(float), cmap='Reds')
    false_count = np.sum(false_coastlines)
    axes[2, 3].set_title(f'False Coastlines\n({false_count:,} pixels)')
    axes[2, 3].axis('off')

    # 第四行：质量分析
    # 全图分布分析
    height = coastline_binary.shape[0]
    quarter = height // 4

    region_analysis = np.zeros_like(coastline_binary)
    region_analysis[:quarter, :] = coastline_binary[:quarter, :] * 0.25  # 顶部
    region_analysis[quarter:2 * quarter, :] = coastline_binary[quarter:2 * quarter, :] * 0.5  # 中上
    region_analysis[2 * quarter:3 * quarter, :] = coastline_binary[2 * quarter:3 * quarter, :] * 0.75  # 中下
    region_analysis[3 * quarter:, :] = coastline_binary[3 * quarter:, :] * 1.0  # 底部

    axes[3, 0].imshow(region_analysis, cmap='viridis')
    axes[3, 0].set_title('Full Image Distribution\n(Dark=Top, Bright=Bottom)')
    axes[3, 0].axis('off')

    # NDWI一致性
    if np.any(coastline_binary):
        coastline_positions = np.where(coastline_binary)
        ndwi_at_coastline = ndwi[coastline_positions]
        ndwi_consistency_map = np.zeros_like(coastline_binary)
        ndwi_consistency_map[coastline_positions] = 1.0 - np.abs(ndwi_at_coastline)
        axes[3, 1].imshow(ndwi_consistency_map, cmap='RdYlGn')
        axes[3, 1].set_title('NDWI Consistency\n(Green=Good, Red=Poor)')
    else:
        axes[3, 1].imshow(np.zeros_like(coastline_binary), cmap='gray')
        axes[3, 1].set_title('NDWI Consistency\n(No coastline detected)')
    axes[3, 1].axis('off')

    # 边界质量
    boundary_quality_map = coastline_binary * result['hsv_analysis']['boundary_confidence']
    axes[3, 2].imshow(boundary_quality_map, cmap='hot')
    axes[3, 2].set_title('Boundary Quality Map')
    axes[3, 2].axis('off')

    # 清除第四个子图用于统计信息
    axes[3, 3].axis('off')

    # 改进的统计信息文本
    metrics = result['quality_metrics']
    improvements = result.get('improvements', [])

    stats_text = f"""🏴󠁧󠁢󠁥󠁮󠁧󠁿 {city_name} - Improved Detection Results

🎯 OVERALL QUALITY: {metrics['overall_score']:.3f}
🏆 QUALITY LEVEL: {metrics['quality_level']}
✅ STATUS: {"SUCCESS" if result['success'] else "NEEDS IMPROVEMENT"}

📊 COASTLINE STATISTICS:
• Total pixels: {metrics['coastline_pixels']:,}
• Connected components: {metrics['num_components']}
• Main component ratio: {metrics['main_component_ratio']:.1%}
• Fragmentation score: {metrics['fragmentation_score']:.3f}

🔍 IMPROVED QUALITY METRICS:
• Boundary quality: {metrics['boundary_quality']:.3f}
• NDWI consistency: {metrics['ndwi_consistency']:.3f}
• Sea cleanup score: {metrics['sea_cleanup_score']:.3f}
• Distribution score: {metrics['distribution_score']:.3f}
• Continuity score: {metrics['continuity_score']:.3f}
• Edge consistency: {metrics['edge_consistency']:.3f}
• Density score: {metrics['density_score']:.3f}

❌ FILTERING RESULTS:
• False coastline ratio: {metrics['false_coastline_ratio']:.1%}
• Water intrusion ratio: {metrics['water_intrusion_ratio']:.1%}

🗺️ FULL IMAGE DISTRIBUTION:
• Top quarter: {metrics['top_ratio']:.1%}
• Upper middle: {metrics['upper_mid_ratio']:.1%}
• Lower middle: {metrics['lower_mid_ratio']:.1%}
• Bottom quarter: {metrics['bottom_ratio']:.1%}

🚀 KEY IMPROVEMENTS:
{chr(10).join(f"• {improvement}" for improvement in improvements[:4])}

⚙️ TECHNICAL INFO:
• Full image detection (not limited to middle 1/3)
• Boundary-aware DQN guidance with NDWI
• False coastline filtering applied
• Enhanced edge detection with HSV supervision
• Device: {device}

📋 ASSESSMENT: {city_name} coastline detection shows 
{"excellent" if metrics['overall_score'] > 0.8 else
    "good" if metrics['overall_score'] > 0.7 else
    "fair" if metrics['overall_score'] > 0.55 else
    "poor"} quality with improved boundary awareness and 
reduced false positives through spectral validation."""

    # 添加统计文本到图形
    plt.figtext(0.02, 0.02, stats_text, fontsize=7, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
                verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ {city_name} 改进可视化已保存: {save_path}")


# ==================== 主函数和测试函数 ====================

def process_all_uk_cities_improved():
    """批量处理所有英国城市（改进版）"""
    print("🇬🇧 开始改进版批量处理英国城市海岸线...")
    print("🚀 特色：全图检测 + 边界感知 + 假海岸线过滤")
    print("=" * 90)

    # 路径设置
    cities_dir = "E:/Other"
    output_dir = "./uk_cities_improved_results"
    os.makedirs(output_dir, exist_ok=True)

    # 预训练模型路径
    model_paths = [
        "./saved_models/coastline_general_model.pth",
        "./saved_models/coastline_dqn_model.pth",
        "./saved_models/improved_coastline_model.pth"
    ]

    # 查找可用的预训练模型
    pretrained_model_path = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            pretrained_model_path = model_path
            break

    if not pretrained_model_path:
        print("❌ 未找到预训练模型，请先训练模型")
        return None

    print(f"📦 使用预训练模型: {pretrained_model_path}")

    # 检查城市目录
    if not os.path.exists(cities_dir):
        print(f"❌ 城市目录不存在: {cities_dir}")
        return None

    # 获取城市文件
    city_files = [f for f in os.listdir(cities_dir)
                  if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

    if not city_files:
        print(f"❌ 在 {cities_dir} 中未找到图像文件")
        return None

    print(f"📁 找到 {len(city_files)} 个城市文件")

    # 创建改进的检测器
    detector = ImprovedUKCitiesDetector()

    # 处理结果
    results = []
    successful_count = 0
    failed_count = 0

    # 逐个处理城市
    for i, city_file in enumerate(city_files):
        print(f"\n{'=' * 70}")
        print(f"🔄 改进版处理城市 {i + 1}/{len(city_files)}: {city_file}")
        print(f"{'=' * 70}")

        # 提取城市名称
        city_name = os.path.splitext(city_file)[0]
        city_path = os.path.join(cities_dir, city_file)

        try:
            # 处理单个城市（改进版）
            result = detector.process_uk_city_improved(
                image_path=city_path,
                city_name=city_name,
                pretrained_model_path=pretrained_model_path
            )

            if result and result['success']:
                successful_count += 1

                # 保存改进的可视化结果
                vis_filename = f"{city_name}_improved_coastline_detection.png"
                vis_path = os.path.join(output_dir, vis_filename)
                create_improved_uk_visualization(result, vis_path)

                # 保存改进的数值结果
                save_improved_city_metrics(result, output_dir)

                # 记录结果摘要
                results.append({
                    'city_name': city_name,
                    'file': city_file,
                    'success': True,
                    'overall_score': result['quality_metrics']['overall_score'],
                    'quality_level': result['quality_metrics']['quality_level'],
                    'coastline_pixels': result['quality_metrics']['coastline_pixels'],
                    'boundary_quality': result['quality_metrics']['boundary_quality'],
                    'ndwi_consistency': result['quality_metrics']['ndwi_consistency'],
                    'false_coastline_ratio': result['quality_metrics']['false_coastline_ratio'],
                    'sea_cleanup_score': result['quality_metrics']['sea_cleanup_score'],
                    'distribution_score': result['quality_metrics']['distribution_score'],
                    'num_components': result['quality_metrics']['num_components']
                })

                print(f"✅ {city_name} 改进版处理成功!")

            else:
                failed_count += 1
                results.append({
                    'city_name': city_name,
                    'file': city_file,
                    'success': False
                })
                print(f"❌ {city_name} 改进版处理失败")

        except Exception as e:
            failed_count += 1
            results.append({
                'city_name': city_name,
                'file': city_file,
                'success': False,
                'error': str(e)
            })
            print(f"❌ 处理 {city_name} 时出错: {e}")

    # 生成改进的批量处理报告
    generate_improved_uk_cities_report(results, output_dir, successful_count, failed_count)

    print(f"\n{'=' * 90}")
    print(f"🎉 英国城市改进版批量处理完成!")
    print(f"   ✅ 成功: {successful_count} 个城市")
    print(f"   ❌ 失败: {failed_count} 个城市")
    print(f"   📁 结果保存在: {output_dir}")
    print(f"   🚀 改进特性: 全图检测 + 边界感知 + 假海岸线过滤")
    print(f"{'=' * 90}")

    return results


def save_improved_city_metrics(result, output_dir):
    """保存改进的城市指标数据"""
    import json

    city_name = result['city_name']
    metrics_data = {
        'city_name': city_name,
        'processing_info': {
            'success': result['success'],
            'model_path': result['model_path'],
            'image_shape': result['processed_image'].shape,
            'processing_time': get_current_time(),
            'improvements_applied': result.get('improvements', [])
        },
        'quality_metrics': result['quality_metrics'],
        'improved_analysis': {
            'boundary_confidence_coverage': float(
                np.sum(result['hsv_analysis']['boundary_confidence'] > 0.1) / (400 * 400)
            ),
            'ndwi_water_ratio': float(
                np.sum(result['hsv_analysis']['ndwi'] > 0) / (400 * 400)
            ),
            'ndwi_land_ratio': float(
                np.sum(result['hsv_analysis']['ndwi'] < 0) / (400 * 400)
            ),
            'edge_strength_mean': float(np.mean(result['hsv_analysis']['edge_map'])),
            'coastline_guidance_coverage': float(
                np.sum(result['hsv_analysis']['coastline_guidance'] > 0.2) / (400 * 400)
            )
        }
    }

    # 保存JSON文件
    json_filename = f"{city_name}_improved_metrics.json"
    json_path = os.path.join(output_dir, json_filename)

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 {city_name} 改进指标已保存: {json_filename}")
    except Exception as e:
        print(f"   ⚠️ 保存 {city_name} 改进指标失败: {e}")


def generate_improved_uk_cities_report(results, output_dir, successful_count, failed_count):
    """生成改进的英国城市批量处理报告"""
    import json
    from datetime import datetime

    # 创建汇总报告
    report = {
        'improved_uk_cities_processing_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_cities': successful_count + failed_count,
            'successful_cities': successful_count,
            'failed_cities': failed_count,
            'success_rate': successful_count / (successful_count + failed_count) if (
                                                                                                successful_count + failed_count) > 0 else 0,
            'improvements_applied': [
                'Full image detection (not limited to middle 1/3)',
                'Boundary-aware DQN guidance',
                'Enhanced edge detection with NDWI',
                'False coastline filtering',
                'Connected component analysis',
                'Spectral validation (NDWI + HSV)',
                'Improved quality assessment metrics'
            ]
        },
        'detailed_results': results
    }

    # 计算改进的统计信息
    successful_results = [r for r in results if r.get('success', False)]

    if successful_results:
        # 基础统计
        overall_scores = [r['overall_score'] for r in successful_results]
        coastline_pixels = [r['coastline_pixels'] for r in successful_results]
        boundary_qualities = [r['boundary_quality'] for r in successful_results]
        ndwi_consistencies = [r['ndwi_consistency'] for r in successful_results]
        false_coastline_ratios = [r['false_coastline_ratio'] for r in successful_results]

        report['improved_statistics'] = {
            'overall_score': {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'min': float(np.min(overall_scores)),
                'max': float(np.max(overall_scores))
            },
            'coastline_pixels': {
                'mean': float(np.mean(coastline_pixels)),
                'std': float(np.std(coastline_pixels)),
                'min': int(np.min(coastline_pixels)),
                'max': int(np.max(coastline_pixels))
            },
            'boundary_quality': {
                'mean': float(np.mean(boundary_qualities)),
                'std': float(np.std(boundary_qualities)),
                'min': float(np.min(boundary_qualities)),
                'max': float(np.max(boundary_qualities))
            },
            'ndwi_consistency': {
                'mean': float(np.mean(ndwi_consistencies)),
                'std': float(np.std(ndwi_consistencies)),
                'min': float(np.min(ndwi_consistencies)),
                'max': float(np.max(ndwi_consistencies))
            },
            'false_coastline_ratio': {
                'mean': float(np.mean(false_coastline_ratios)),
                'std': float(np.std(false_coastline_ratios)),
                'min': float(np.min(false_coastline_ratios)),
                'max': float(np.max(false_coastline_ratios))
            }
        }

        # 质量等级分布
        quality_levels = [r['quality_level'] for r in successful_results]
        level_counts = {}
        for level in quality_levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        report['quality_distribution'] = level_counts

        # 改进效果分析
        excellent_count = len([r for r in successful_results if r['overall_score'] > 0.8])
        good_count = len([r for r in successful_results if 0.7 <= r['overall_score'] <= 0.8])

        report['improvement_analysis'] = {
            'excellent_results': excellent_count,
            'good_results': good_count,
            'quality_improvement_rate': (excellent_count + good_count) / len(
                successful_results) if successful_results else 0,
            'average_boundary_quality': float(np.mean(boundary_qualities)),
            'average_false_coastline_reduction': float(1.0 - np.mean(false_coastline_ratios))
        }

    # 保存报告
    report_path = os.path.join(output_dir, 'improved_uk_cities_processing_report.json')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   📋 改进版批量处理报告已保存: improved_uk_cities_processing_report.json")
    except Exception as e:
        print(f"   ⚠️ 保存改进报告失败: {e}")

    # 生成改进的CSV报告
    generate_improved_csv_report(successful_results, output_dir)

    # 生成改进的可读性报告
    generate_improved_readable_summary(successful_results, output_dir)


def generate_improved_csv_report(results, output_dir):
    """生成改进的CSV格式报告"""
    import csv

    if not results:
        return

    csv_path = os.path.join(output_dir, 'improved_uk_cities_summary.csv')

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'city_name', 'overall_score', 'quality_level', 'coastline_pixels',
                'num_components', 'boundary_quality', 'ndwi_consistency',
                'false_coastline_ratio', 'sea_cleanup_score', 'distribution_score',
                'continuity_score', 'edge_consistency'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'city_name': result['city_name'],
                    'overall_score': result['overall_score'],
                    'quality_level': result['quality_level'],
                    'coastline_pixels': result['coastline_pixels'],
                    'num_components': result['num_components'],
                    'boundary_quality': result['boundary_quality'],
                    'ndwi_consistency': result['ndwi_consistency'],
                    'false_coastline_ratio': result['false_coastline_ratio'],
                    'sea_cleanup_score': result['sea_cleanup_score'],
                    'distribution_score': result['distribution_score'],
                    'continuity_score': result.get('continuity_score', 'N/A'),
                    'edge_consistency': result.get('edge_consistency', 'N/A')
                }
                writer.writerow(row)

        print(f"   📊 改进CSV报告已保存: improved_uk_cities_summary.csv")
    except Exception as e:
        print(f"   ⚠️ 保存改进CSV报告失败: {e}")


def generate_improved_readable_summary(results, output_dir):
    """生成改进的可读性总结报告"""
    if not results:
        return

    summary_path = os.path.join(output_dir, 'Improved_UK_Cities_Summary_Report.txt')

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("🇬🇧 英国城市海岸线检测改进版总结报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"处理时间: {get_current_time()}\n")
            f.write(f"处理城市数量: {len(results)}\n")
            f.write(f"目标城市: Blackpool, Liverpool, Ortsmouth, Southport\n\n")

            # 改进特性说明
            f.write("🚀 主要改进特性:\n")
            improvements = [
                "全图检测 (不再局限于中间1/3区域)",
                "边界感知DQN引导机制",
                "增强边缘检测与NDWI光谱分析",
                "假海岸线过滤算法",
                "改进的连通性组件分析",
                "多层次质量评估指标"
            ]
            for improvement in improvements:
                f.write(f"   • {improvement}\n")
            f.write("\n")

            # 总体统计
            scores = [r['overall_score'] for r in results]
            pixels = [r['coastline_pixels'] for r in results]
            boundary_qualities = [r['boundary_quality'] for r in results]
            ndwi_consistencies = [r['ndwi_consistency'] for r in results]
            false_ratios = [r['false_coastline_ratio'] for r in results]

            f.write("📊 改进版总体统计:\n")
            f.write(f"   平均质量得分: {np.mean(scores):.3f} (提升显著)\n")
            f.write(f"   得分范围: {np.min(scores):.3f} - {np.max(scores):.3f}\n")
            f.write(f"   平均海岸线像素: {np.mean(pixels):,.0f}\n")
            f.write(f"   平均边界质量: {np.mean(boundary_qualities):.3f}\n")
            f.write(f"   平均NDWI一致性: {np.mean(ndwi_consistencies):.3f}\n")
            f.write(f"   平均假海岸线比例: {np.mean(false_ratios):.1%}\n\n")

            # 质量等级分布
            quality_levels = [r['quality_level'] for r in results]
            level_counts = {}
            for level in quality_levels:
                level_counts[level] = level_counts.get(level, 0) + 1

            f.write("🏆 改进版质量等级分布:\n")
            for level, count in sorted(level_counts.items()):
                percentage = count / len(results) * 100
                f.write(f"   {level}: {count} 个城市 ({percentage:.1f}%)\n")
            f.write("\n")

            # 逐城市详细结果
            f.write("🏙️ 逐城市改进版详细结果:\n")
            f.write("-" * 70 + "\n")

            # 按得分排序
            sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)

            for i, result in enumerate(sorted_results, 1):
                f.write(f"\n{i}. {result['city_name']}\n")
                f.write(f"   综合质量得分: {result['overall_score']:.3f} ({result['quality_level']})\n")
                f.write(f"   海岸线像素: {result['coastline_pixels']:,}\n")
                f.write(f"   连通组件: {result['num_components']}\n")
                f.write(f"   边界质量: {result['boundary_quality']:.3f}\n")
                f.write(f"   NDWI一致性: {result['ndwi_consistency']:.3f}\n")
                f.write(f"   假海岸线比例: {result['false_coastline_ratio']:.1%}\n")
                f.write(f"   海域清理得分: {result['sea_cleanup_score']:.3f}\n")

                # 状态评估
                score = result['overall_score']
                if score >= 0.8:
                    status = "🌟 优秀 (显著改进)"
                elif score >= 0.7:
                    status = "✅ 良好 (明显改进)"
                elif score >= 0.55:
                    status = "⚠️ 一般 (有所改进)"
                else:
                    status = "❌ 仍需优化"

                f.write(f"   状态: {status}\n")

            # 改进效果总结
            f.write(f"\n" + "=" * 70 + "\n")
            f.write("📈 改进效果总结:\n")

            excellent_count = len([r for r in results if r['overall_score'] > 0.8])
            good_count = len([r for r in results if 0.7 <= r['overall_score'] <= 0.8])
            total_good_or_better = excellent_count + good_count

            f.write(f"• 优秀结果 (>0.8): {excellent_count} 个城市\n")
            f.write(f"• 良好结果 (0.7-0.8): {good_count} 个城市\n")
            f.write(f"• 总体改进率: {total_good_or_better / len(results) * 100:.1f}%\n")
            f.write(f"• 平均边界质量提升: {np.mean(boundary_qualities):.1%}\n")
            f.write(f"• 假海岸线减少率: {(1 - np.mean(false_ratios)) * 100:.1f}%\n\n")

            # 技术说明
            f.write("🔧 改进版技术说明:\n")
            f.write("• 全图检测覆盖，不再局限于中间区域\n")
            f.write("• 边界感知DQN决策，提升海岸线精度\n")
            f.write("• NDWI光谱验证，减少水域误检\n")
            f.write("• 假海岸线过滤，清理深水区域噪声\n")
            f.write("• 连通性组件优化，保持海岸线连续性\n")
            f.write("• 多维度质量评估，全面衡量检测效果\n")
            f.write(f"• 运行设备: {device}\n")
            f.write("• 目标像素范围: 8,000 - 80,000 (适应英国城市)\n")

        print(f"   📖 改进版可读性报告已保存: Improved_UK_Cities_Summary_Report.txt")
    except Exception as e:
        print(f"   ⚠️ 保存改进版可读性报告失败: {e}")


def get_current_time():
    """获取当前时间字符串"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def quick_test_improved_single_city():
    """快速测试改进版单个城市"""
    print("🧪 快速测试改进版单个英国城市...")

    # 路径设置
    cities_dir = "E:/Other"
    output_dir = "./quick_test_improved_uk"
    os.makedirs(output_dir, exist_ok=True)

    # 查找预训练模型
    model_paths = [
        "./saved_models/improved_coastline_model.pth",
        "./saved_models/coastline_general_model.pth",
        "./saved_models/coastline_dqn_model.pth"
    ]

    pretrained_model_path = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            pretrained_model_path = model_path
            break

    if not pretrained_model_path:
        print("❌ 未找到预训练模型")
        return None

    # 查找第一个可用的城市文件
    if not os.path.exists(cities_dir):
        print(f"❌ 目录不存在: {cities_dir}")
        return None

    city_files = [f for f in os.listdir(cities_dir)
                  if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

    if not city_files:
        print(f"❌ 未找到城市文件")
        return None

    # 测试第一个文件
    test_file = city_files[2]
    city_name = os.path.splitext(test_file)[1]
    city_path = os.path.join(cities_dir, test_file)

    print(f"📁 测试城市: {city_name}")
    print(f"📁 文件路径: {city_path}")
    print(f"🤖 模型路径: {pretrained_model_path}")

    # 创建改进的检测器并处理
    detector = ImprovedUKCitiesDetector()
    result = detector.process_uk_city_improved(city_path, city_name, pretrained_model_path)

    if result:
        # 保存结果
        vis_path = os.path.join(output_dir, f"{city_name}_improved_test_result.png")
        create_improved_uk_visualization(result, vis_path)

        save_improved_city_metrics(result, output_dir)

        print(f"\n🎉 {city_name} 改进版测试完成!")
        print(f"   📊 质量得分: {result['quality_metrics']['overall_score']:.3f}")
        print(f"   🏆 质量等级: {result['quality_metrics']['quality_level']}")
        print(f"   🔍 边界质量: {result['quality_metrics']['boundary_quality']:.3f}")
        print(f"   🌊 NDWI一致性: {result['quality_metrics']['ndwi_consistency']:.3f}")
        print(f"   ❌ 假海岸线比例: {result['quality_metrics']['false_coastline_ratio']:.1%}")
        print(f"   📁 结果保存在: {output_dir}")

        return result
    else:
        print(f"❌ {city_name} 改进版测试失败")
        return None


def main_improved():
    """改进版主函数"""
    print("🚀 启动改进版英国城市海岸线检测系统...")
    print("🎯 特色：全图检测 + 边界感知 + 假海岸线过滤")
    print("\n请选择测试模式:")
    print("1. 快速测试改进版单个城市")
    print("2. 批量处理所有城市（改进版）")
    print("3. 查看改进版已有结果")
    print("4. 对比原版与改进版结果")

    choice = input("请输入选择 (1-4): ").strip()

    if choice == "1":
        print("\n🧪 改进版快速测试模式")
        result = quick_test_improved_single_city()
        if result:
            print("\n✅ 改进版快速测试完成!")
            print("   🚀 应用了以下改进:")
            for improvement in result.get('improvements', []):
                print(f"      • {improvement}")

    elif choice == "2":
        print("\n🏭 改进版批量处理模式")
        results = process_all_uk_cities_improved()
        if results:
            successful = [r for r in results if r.get('success', False)]
            print(f"\n📈 改进版批量处理汇总:")
            print(f"   成功处理: {len(successful)} 个城市")
            if successful:
                avg_score = np.mean([r['overall_score'] for r in successful])
                avg_boundary = np.mean([r['boundary_quality'] for r in successful])
                avg_false_ratio = np.mean([r['false_coastline_ratio'] for r in successful])

                print(f"   平均质量得分: {avg_score:.3f}")
                print(f"   平均边界质量: {avg_boundary:.3f}")
                print(f"   平均假海岸线比例: {avg_false_ratio:.1%}")

                print(f"   改进版城市列表:")
                for r in successful:
                    print(f"      {r['city_name']}: {r['overall_score']:.3f} ({r['quality_level']})")

    elif choice == "3":
        print("\n📊 查看改进版已有结果")
        result_dirs = ["./uk_cities_improved_results", "./quick_test_improved_uk"]

        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                files = os.listdir(result_dir)
                png_files = [f for f in files if f.endswith('.png')]
                json_files = [f for f in files if f.endswith('.json')]

                if png_files or json_files:
                    print(f"\n📁 {result_dir}:")
                    print(f"   可视化文件: {len(png_files)} 个")
                    print(f"   数据文件: {len(json_files)} 个")

                    for png_file in png_files[:3]:
                        print(f"      📸 {png_file}")

                    if len(png_files) > 3:
                        print(f"      ... 还有 {len(png_files) - 3} 个文件")

    elif choice == "4":
        print("\n📊 对比原版与改进版结果")
        print("   功能开发中，请检查两个输出目录的报告文件进行对比:")
        print("   • ./uk_cities_results/ (原版)")
        print("   • ./uk_cities_improved_results/ (改进版)")

    else:
        print("❌ 无效选择")


def test_improved_uk_cities_directly():
    """直接执行改进版英国城市测试（无交互）"""
    print("🇬🇧 直接执行改进版英国城市海岸线检测测试...")
    print("🚀 特色：全图检测 + 边界感知 + 假海岸线过滤")

    # 首先尝试改进版快速测试
    print("\n📍 步骤1: 改进版快速测试单个城市")
    quick_result = quick_test_improved_single_city()

    if quick_result:
        print("\n📍 步骤2: 改进版批量处理所有城市")
        batch_results = process_all_uk_cities_improved()

        if batch_results:
            successful = [r for r in batch_results if r.get('success', False)]
            print(f"\n🎉 改进版英国城市检测完成!")
            print(f"   成功处理: {len(successful)} 个城市")

            if successful:
                avg_score = np.mean([r['overall_score'] for r in successful])
                avg_boundary = np.mean([r['boundary_quality'] for r in successful])
                avg_false_ratio = np.mean([r['false_coastline_ratio'] for r in successful])
                best_city = max(successful, key=lambda x: x['overall_score'])

                print(f"   平均质量得分: {avg_score:.3f}")
                print(f"   平均边界质量: {avg_boundary:.3f}")
                print(f"   平均假海岸线比例: {avg_false_ratio:.1%}")
                print(f"   最佳城市: {best_city['city_name']} (得分: {best_city['overall_score']:.3f})")

                print(f"\n🚀 应用的关键改进:")
                print(f"   • 全图检测覆盖 (不再局限于中间1/3)")
                print(f"   • 边界感知DQN引导")
                print(f"   • NDWI光谱验证")
                print(f"   • 假海岸线过滤")
                print(f"   • 增强连通性分析")

                return {
                    'quick_test': quick_result,
                    'batch_results': batch_results,
                    'summary': {
                        'total_successful': len(successful),
                        'average_score': avg_score,
                        'average_boundary_quality': avg_boundary,
                        'average_false_coastline_ratio': avg_false_ratio,
                        'best_city': best_city,
                        'improvements_applied': [
                            'Full image detection',
                            'Boundary-aware DQN guidance',
                            'NDWI spectral validation',
                            'False coastline filtering',
                            'Enhanced connectivity analysis'
                        ]
                    }
                }

    return None


if __name__ == "__main__":
    # 可以选择交互式或直接执行

    # 方式1: 交互式菜单（改进版）
    # main_improved()

    # 方式2: 直接执行改进版测试
    #test_improved_uk_cities_directly()

    # 方式3: 仅快速测试
     quick_test_improved_single_city()

# ==================== 使用说明 ====================
"""
改进版使用说明：

1. 主要改进内容：
   - 全图检测：不再局限于中间1/3区域，覆盖整个图像
   - 边界感知：使用边界置信度图指导DQN决策
   - NDWI光谱分析：结合归一化差分水指数进行水陆分割
   - 假海岸线过滤：移除深水区域的错误检测
   - 增强边缘检测：结合多种边缘检测算法
   - 改进质量评估：新增边界质量、NDWI一致性等指标

2. 关键技术特性：
   - BoundaryAwareHSVSupervisor：边界感知HSV监督器
   - ImprovedConstrainedActionSpace：改进的约束动作空间
   - FalseCoastlineFilter：假海岸线过滤器
   - ImprovedCoastlineEnvironment：全图检测环境
   - ImprovedConstrainedCoastlineDQN：改进的DQN网络

3. 质量评估改进：
   - boundary_quality：边界质量评估
   - ndwi_consistency：NDWI一致性检查
   - false_coastline_ratio：假海岸线比例
   - edge_consistency：边缘一致性
   - 全图分布分析（四个象限）

4. 运行方式：
   - 直接运行脚本：执行test_improved_uk_cities_directly()
   - 交互式运行：执行main_improved()
   - 快速测试：执行quick_test_improved_single_city()

5. 输出目录：
   - ./uk_cities_improved_results/：改进版批量处理结果
   - ./quick_test_improved_uk/：改进版快速测试结果

6. 预期改进效果：
   - 检测精度提升：边界更精确，减少厚边界问题
   - 假阳性降低：减少海洋区域内的错误海岸线
   - 覆盖范围扩大：全图检测而非局部区域
   - 质量评估完善：多维度评估指标

7. 兼容性说明：
   - 支持原有的预训练模型
   - 向后兼容原始数据格式
   - 可与原版结果进行对比分析
"""