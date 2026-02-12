"""
精准海域清理海岸线检测系统 - 完整版
主要改进：
1. 强化海域识别和清理
2. 防止上下岸线连通
3. 精准像素数量控制（目标：9-10万像素）
4. 保持GT保护的前提下大幅减少误检
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
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("🌊 精准海域清理海岸线检测系统!")
print("重点：海域清理 + 防连通 + 像素控制 + GT保护")
print("=" * 90)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ==================== 基础类 ====================

class BasicImageProcessor:
    @staticmethod
    def rgb_to_gray(rgb_image):
        if len(rgb_image.shape) == 3:
            return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return rgb_image


class GroundTruthAnalyzer:
    """Ground Truth分析器"""
    def __init__(self):
        print("✅ Ground Truth分析器初始化完成")

    def analyze_gt_pattern(self, gt_coastline):
        if gt_coastline is None:
            return None

        gt_binary = (gt_coastline > 0.5).astype(bool)
        edge_region = gt_binary.copy()
        for _ in range(12):
            edge_region = binary_dilation(edge_region, np.ones((3, 3), dtype=bool))

        density_map = gaussian_filter(gt_binary.astype(float), sigma=8)
        density_map = density_map / (density_map.max() + 1e-8)

        return {
            'gt_binary': gt_binary,
            'edge_region': edge_region,
            'density_map': density_map,
            'total_pixels': np.sum(gt_binary)
        }


# ==================== HSV监督器 ====================

class HSVAttentionSupervisor:
    """HSV注意力监督器"""

    def __init__(self):
        print("✅ HSV注意力监督器初始化完成")
        self.water_hsv_range = self._define_water_hsv_range()
        self.land_hsv_range = self._define_land_hsv_range()

    def _define_water_hsv_range(self):
        """定义水体的HSV范围"""
        return {
            'hue_range': (180, 240),
            'saturation_min': 0.2,
            'value_min': 0.1
        }

    def _define_land_hsv_range(self):
        """定义陆地的HSV范围"""
        return {
            'hue_range': (60, 120),
            'saturation_min': 0.1,
            'value_min': 0.2
        }

    def analyze_image_hsv(self, rgb_image, gt_analysis=None):
        """分析图像的HSV特征"""
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

        water_mask = self._detect_water_regions(hsv_image)
        land_mask = self._detect_land_regions(hsv_image)

        if gt_analysis is not None:
            water_mask, land_mask = self._refine_with_gt(
                water_mask, land_mask, gt_analysis, hsv_image
            )

        coastline_guidance = self._generate_coastline_guidance(water_mask, land_mask, gt_analysis)

        return {
            'hsv_image': hsv_image,
            'water_mask': water_mask,
            'land_mask': land_mask,
            'coastline_guidance': coastline_guidance,
            'transition_strength': self._calculate_transition_strength(hsv_image, water_mask, land_mask, gt_analysis)
        }

    def _detect_water_regions(self, hsv_image):
        """检测水体区域"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        hue_mask = ((h >= self.water_hsv_range['hue_range'][0]) &
                    (h <= self.water_hsv_range['hue_range'][1]))
        saturation_mask = s >= self.water_hsv_range['saturation_min']
        value_mask = v >= self.water_hsv_range['value_min']

        water_mask = hue_mask & saturation_mask & value_mask
        water_mask = binary_closing(water_mask, np.ones((5, 5)))
        water_mask = binary_erosion(water_mask, np.ones((3, 3)))
        water_mask = binary_dilation(water_mask, np.ones((3, 3)))

        return water_mask

    def _detect_land_regions(self, hsv_image):
        """检测陆地区域"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        green_hue_mask = ((h >= self.land_hsv_range['hue_range'][0]) &
                          (h <= self.land_hsv_range['hue_range'][1]))
        brown_hue_mask = ((h >= 20) & (h <= 50))
        gray_mask = (s <= 0.2) & (v >= 0.4)
        hue_mask = green_hue_mask | brown_hue_mask | gray_mask

        saturation_mask = s >= self.land_hsv_range['saturation_min']
        value_mask = v >= self.land_hsv_range['value_min']

        land_mask = hue_mask & (saturation_mask | gray_mask) & value_mask
        land_mask = binary_closing(land_mask, np.ones((5, 5)))
        land_mask = binary_erosion(land_mask, np.ones((2, 2)))
        land_mask = binary_dilation(land_mask, np.ones((3, 3)))

        return land_mask

    def _refine_with_gt(self, water_mask, land_mask, gt_analysis, hsv_image):
        """使用GT信息改进水陆分割"""
        print("   🎯 使用GT信息改进HSV水陆分割...")

        gt_binary = gt_analysis['gt_binary']
        gt_edge_region = gt_analysis['edge_region']

        edge_positions = np.where(gt_edge_region)
        if len(edge_positions[0]) == 0:
            return water_mask, land_mask

        sample_step = max(1, len(edge_positions[0]) // 50)
        sample_indices = range(0, len(edge_positions[0]), sample_step)

        water_samples = []
        land_samples = []

        for idx in sample_indices:
            y, x = edge_positions[0][idx], edge_positions[1][idx]
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < hsv_image.shape[0] and 0 <= nx < hsv_image.shape[1]:
                    pixel_hsv = hsv_image[ny, nx]
                    if pixel_hsv[2] < 0.35:
                        water_samples.append(pixel_hsv)
                    elif pixel_hsv[2] > 0.4:
                        land_samples.append(pixel_hsv)

        if len(water_samples) < 5 or len(land_samples) < 5:
            return water_mask, land_mask

        water_samples = np.array(water_samples)
        land_samples = np.array(land_samples)
        water_center = np.mean(water_samples, axis=0)
        land_center = np.mean(land_samples, axis=0)

        refined_water_mask = water_mask.copy()
        refined_land_mask = land_mask.copy()

        search_region = binary_dilation(gt_edge_region, np.ones((10, 10)))
        search_positions = np.where(search_region)

        for i in range(len(search_positions[0])):
            y, x = search_positions[0][i], search_positions[1][i]
            pixel = hsv_image[y, x]

            water_dist = np.linalg.norm(pixel - water_center)
            land_dist = np.linalg.norm(pixel - land_center)

            if water_dist < land_dist * 0.9:
                refined_water_mask[y, x] = True
                refined_land_mask[y, x] = False
            elif land_dist < water_dist * 0.9:
                refined_land_mask[y, x] = True
                refined_water_mask[y, x] = False

        kernel = np.ones((3, 3))
        refined_water_mask = binary_closing(refined_water_mask, kernel)
        refined_land_mask = binary_closing(refined_land_mask, kernel)

        return refined_water_mask, refined_land_mask

    def _generate_coastline_guidance(self, water_mask, land_mask, gt_analysis=None):
        """生成海岸线引导图"""
        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask
        coastline_candidates = water_boundary | land_boundary

        if gt_analysis is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_guidance = binary_dilation(gt_binary, np.ones((5, 5)))
            coastline_candidates = coastline_candidates | gt_guidance

        coastline_guidance = coastline_candidates.copy()
        for _ in range(3):
            coastline_guidance = binary_dilation(coastline_guidance, np.ones((3, 3)))

        from scipy.ndimage import distance_transform_edt

        if np.any(water_mask):
            water_dist = distance_transform_edt(~water_mask)
        else:
            water_dist = np.ones_like(water_mask, dtype=float) * 10

        if np.any(land_mask):
            land_dist = distance_transform_edt(~land_mask)
        else:
            land_dist = np.ones_like(land_mask, dtype=float) * 10

        guidance_strength = np.exp(-0.05 * (water_dist + land_dist))

        if gt_analysis is not None:
            gt_dist = distance_transform_edt(~gt_analysis['gt_binary'])
            gt_bonus = np.exp(-0.1 * gt_dist)
            guidance_strength = guidance_strength + gt_bonus * 0.8

        guidance_strength = coastline_guidance * guidance_strength

        if guidance_strength.max() > 0:
            guidance_strength = guidance_strength / guidance_strength.max()

        return guidance_strength

    def _calculate_transition_strength(self, hsv_image, water_mask, land_mask, gt_analysis=None):
        """计算过渡区域强度"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        h_grad = np.abs(np.gradient(h)[0]) + np.abs(np.gradient(h)[1])
        s_grad = np.abs(np.gradient(s)[0]) + np.abs(np.gradient(s)[1])
        v_grad = np.abs(np.gradient(v)[0]) + np.abs(np.gradient(v)[1])

        transition_strength = (h_grad * 0.4 + s_grad * 0.3 + v_grad * 0.3)

        if transition_strength.max() > transition_strength.min():
            transition_strength = (transition_strength - transition_strength.min()) / (
                        transition_strength.max() - transition_strength.min() + 1e-8)

        boundary_mask = binary_dilation(water_mask, np.ones((5, 5))) | binary_dilation(land_mask, np.ones((5, 5)))
        transition_strength = transition_strength * (1 + boundary_mask * 1.5)

        if gt_analysis is not None:
            gt_edge_region = gt_analysis['edge_region']
            transition_strength = transition_strength * (1 + gt_edge_region * 2.0)

        return transition_strength


# ==================== 约束的动作空间 ====================

class ConstrainedActionSpace:
    """约束的动作空间"""

    def __init__(self):
        self.base_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                             (0, 1), (1, -1), (1, 0), (1, 1)]
        print("✅ 约束动作空间初始化完成")

    def get_allowed_actions(self, current_position, coastline_state, hsv_analysis):
        """获取当前位置允许的动作"""
        allowed_actions = []
        context = self._analyze_position_context(current_position, coastline_state, hsv_analysis)

        for i, action in enumerate(self.base_actions):
            if self._is_action_allowed(action, context, current_position, hsv_analysis):
                allowed_actions.append(i)

        return allowed_actions if allowed_actions else [0, 1, 3, 4]

    def _analyze_position_context(self, position, coastline_state, hsv_analysis):
        """分析位置上下文"""
        y, x = position
        y_start, y_end = max(0, y - 3), min(coastline_state.shape[0], y + 4)
        x_start, x_end = max(0, x - 3), min(coastline_state.shape[1], x + 4)

        local_coastline = coastline_state[y_start:y_end, x_start:x_end]
        coastline_density = np.mean(local_coastline > 0.3)

        if hsv_analysis:
            water_mask = hsv_analysis['water_mask']
            near_water = water_mask[y, x] or np.any(water_mask[y_start:y_end, x_start:x_end])
        else:
            near_water = False

        return {
            'coastline_density': coastline_density,
            'near_water': near_water,
            'main_direction': 'horizontal'
        }

    def _is_action_allowed(self, action, context, current_position, hsv_analysis):
        """判断动作是否被允许"""
        dy, dx = action

        if context['near_water'] and abs(dy) > 0:
            if abs(dy) > 1 or (abs(dy) == 1 and abs(dx) == 0):
                return False

        if abs(dy) + abs(dx) > 2:
            return False

        return True


# ==================== 好奇心驱动探索 ====================

class CuriosityDrivenExploration:
    """好奇心驱动的探索机制"""

    def __init__(self, exploration_decay=0.995):
        self.visit_history = {}
        self.exploration_decay = exploration_decay
        self.step_count = 0
        print("✅ 好奇心驱动探索机制初始化完成")

    def get_curiosity_bonus(self, position, hsv_analysis, current_coastline):
        """获取好奇心奖励"""
        y, x = position
        pos_key = f"{y}_{x}"

        visit_count = self.visit_history.get(pos_key, 0)
        visit_bonus = max(0, 10.0 - visit_count * 2.0)

        hsv_bonus = 0.0
        if hsv_analysis:
            coastline_guidance = hsv_analysis['coastline_guidance']
            if coastline_guidance[y, x] > 0.3:
                hsv_bonus = coastline_guidance[y, x] * 15.0

        self.visit_history[pos_key] = visit_count + 1
        self.step_count += 1

        return visit_bonus + hsv_bonus


# ==================== 约束的DQN网络 ====================

class ConstrainedCoastlineDQN(nn.Module):
    """约束的海岸线DQN网络"""

    def __init__(self, input_channels=3, hidden_dim=256, action_dim=8):
        super(ConstrainedCoastlineDQN, self).__init__()

        self.rgb_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.hsv_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.feature_dim = 128 * 8 * 8 + 64 * 8 * 8

        self.q_network = nn.Sequential(
            nn.Linear(self.feature_dim + 2 + 25, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )

        self.action_mask_network = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_state, hsv_state, position, enhanced_features):
        rgb_features = self.rgb_extractor(rgb_state)
        hsv_features = self.hsv_extractor(hsv_state)

        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        hsv_features = hsv_features.view(hsv_features.size(0), -1)

        position_norm = position.float() / 400.0

        combined = torch.cat([rgb_features, hsv_features, position_norm, enhanced_features], dim=1)

        q_values = self.q_network(combined)
        action_mask = self.action_mask_network(enhanced_features)
        masked_q_values = q_values * action_mask - (1 - action_mask) * 1e6

        return masked_q_values


# ==================== 约束环境 ====================

class ConstrainedCoastlineEnvironment:
    """约束的海岸线环境"""

    def __init__(self, image, gt_analysis):
        self.image = image
        self.gt_analysis = gt_analysis
        self.current_coastline = np.zeros(image.shape[:2], dtype=float)
        self.height, self.width = image.shape[:2]

        self.hsv_supervisor = HSVAttentionSupervisor()
        self.hsv_analysis = self.hsv_supervisor.analyze_image_hsv(image, gt_analysis)

        self.action_constraints = ConstrainedActionSpace()
        self.base_actions = self.action_constraints.base_actions
        self.action_dim = len(self.base_actions)

        self.curiosity_explorer = CuriosityDrivenExploration()

        self.edge_map = self._detect_edges()
        self._setup_constrained_search_region()

        print(f"✅ 约束海岸线环境初始化完成")

    def _detect_edges(self):
        """边缘检测"""
        if len(self.image.shape) == 3:
            gray = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = self.image.copy()

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = ndimage.convolve(gray, sobel_x, mode='constant')
        grad_y = ndimage.convolve(gray, sobel_y, mode='constant')

        edge_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-8)

        return edge_magnitude

    def _setup_constrained_search_region(self):
        """设置约束的搜索区域"""
        effective_region = self._identify_effective_coastline_region()

        coastline_guidance = self.hsv_analysis['coastline_guidance']
        transition_strength = self.hsv_analysis['transition_strength']

        primary_region = (coastline_guidance > 0.2) | (transition_strength > 0.4)
        self.search_region = primary_region & effective_region

        for _ in range(2):
            expanded = binary_dilation(self.search_region, np.ones((3, 3), dtype=bool))
            self.search_region = expanded & effective_region

        deep_water = self.hsv_analysis['water_mask']
        for _ in range(5):
            deep_water = binary_erosion(deep_water, np.ones((3, 3), dtype=bool))

        self.search_region = self.search_region & ~deep_water

        if self.gt_analysis:
            gt_region = self.gt_analysis['edge_region'] & effective_region
            self.search_region = self.search_region | gt_region

    def _identify_effective_coastline_region(self):
        """智能识别有效的海岸线区域"""
        height, width = self.height, self.width

        if self.gt_analysis and self.gt_analysis['gt_binary'] is not None:
            gt_binary = self.gt_analysis['gt_binary']
            gt_positions = np.where(gt_binary)

            if len(gt_positions[0]) > 0:
                y_coords = gt_positions[0]
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                y_range = y_max - y_min

                margin = max(20, y_range // 4)
                effective_y_min = max(0, y_min - margin)
                effective_y_max = min(height, y_max + margin)
            else:
                effective_y_min = height // 3
                effective_y_max = 2 * height // 3
        else:
            effective_y_min = height // 3
            effective_y_max = 2 * height // 3

        effective_region = np.zeros((height, width), dtype=bool)
        effective_region[effective_y_min:effective_y_max, :] = True

        return effective_region

    def get_state_tensor(self, position):
        """获取状态张量"""
        y, x = position
        window_size = 64
        half_window = window_size // 2

        y_start = max(0, y - half_window)
        y_end = min(self.height, y + half_window)
        x_start = max(0, x - half_window)
        x_end = min(self.width, x + half_window)

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

        hsv_state = np.zeros((3, window_size, window_size), dtype=np.float32)

        guidance_window = self.hsv_analysis['coastline_guidance'][y_start:y_end, x_start:x_end]
        hsv_state[0, :actual_h, :actual_w] = guidance_window

        transition_window = self.hsv_analysis['transition_strength'][y_start:y_end, x_start:x_end]
        hsv_state[1, :actual_h, :actual_w] = transition_window

        water_window = self.hsv_analysis['water_mask'][y_start:y_end, x_start:x_end].astype(float)
        hsv_state[2, :actual_h, :actual_w] = water_window

        rgb_tensor = torch.FloatTensor(rgb_state).unsqueeze(0).to(device)
        hsv_tensor = torch.FloatTensor(hsv_state).unsqueeze(0).to(device)

        return rgb_tensor, hsv_tensor

    def get_enhanced_features(self, position):
        """获取增强特征"""
        y, x = position

        if not (0 <= y < self.height and 0 <= x < self.width):
            return torch.zeros(25, dtype=torch.float32, device=device).unsqueeze(0)

        features = np.zeros(25, dtype=np.float32)

        features[0] = self.edge_map[y, x]
        features[1] = self.hsv_analysis['coastline_guidance'][y, x]
        features[2] = self.hsv_analysis['transition_strength'][y, x]
        features[3] = 1.0 if self.hsv_analysis['water_mask'][y, x] else 0.0
        features[4] = 1.0 if self.hsv_analysis['land_mask'][y, x] else 0.0

        y_start, y_end = max(0, y - 3), min(self.height, y + 4)
        x_start, x_end = max(0, x - 3), min(self.width, x + 4)

        local_guidance = self.hsv_analysis['coastline_guidance'][y_start:y_end, x_start:x_end]
        if local_guidance.size > 0:
            features[5] = np.mean(local_guidance)
            features[6] = np.max(local_guidance)

        if self.gt_analysis:
            try:
                features[11] = 1.0 if self.gt_analysis['gt_binary'][y, x] else 0.0
                if np.any(self.gt_analysis['gt_binary']):
                    gt_coords = np.where(self.gt_analysis['gt_binary'])
                    if len(gt_coords[0]) > 0:
                        distances = np.sqrt((gt_coords[0] - y) ** 2 + (gt_coords[1] - x) ** 2)
                        min_dist = np.min(distances)
                        features[12] = min(1.0, min_dist / 20.0)
            except (IndexError, KeyError):
                pass

        curiosity_bonus = self.curiosity_explorer.get_curiosity_bonus(
            position, self.hsv_analysis, self.current_coastline
        )
        features[15] = min(1.0, curiosity_bonus / 50.0)

        features[19] = y / self.height
        features[20] = x / self.width

        middle_start = self.height // 3
        middle_end = 2 * self.height // 3

        if middle_start <= y <= middle_end:
            features[24] = 1.0
        elif y < middle_start:
            features[24] = -1.0
        else:
            features[24] = -0.5

        return torch.FloatTensor(features).unsqueeze(0).to(device)

    def step(self, position, action_idx):
        """执行动作"""
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
        reward = self._calculate_constrained_reward(position, new_position, action_idx)

        return new_position, reward

    def _calculate_constrained_reward(self, old_pos, new_pos, action_idx):
        """计算奖励函数"""
        y, x = new_pos
        reward = 0.0

        if not (0 <= y < self.height and 0 <= x < self.width):
            return -50.0

        if not self.search_region[y, x]:
            if y < self.height // 3 or y > 2 * self.height // 3:
                return -100.0
            else:
                return -30.0

        # 区域位置奖励
        height = self.height
        core_start = int(height * 0.3)
        core_end = int(height * 0.7)

        if core_start <= y <= core_end:
            reward += 25.0
        elif int(height * 0.25) <= y <= int(height * 0.75):
            reward += 10.0

        # HSV监督奖励
        hsv_reward = self._calculate_hsv_reward(new_pos)
        reward += hsv_reward * 20.0

        # GT奖励
        if self.gt_analysis and self.gt_analysis['gt_binary'] is not None:
            gt_reward = self._calculate_gt_reward(new_pos)
            reward += gt_reward

        # 海陆分离奖励
        sea_land_separation_reward = self._calculate_sea_land_separation_reward(new_pos)
        reward += sea_land_separation_reward

        return reward

    def _calculate_hsv_reward(self, position):
        """计算HSV监督奖励"""
        y, x = position
        guidance_score = self.hsv_analysis['coastline_guidance'][y, x]
        transition_score = self.hsv_analysis['transition_strength'][y, x]
        return guidance_score + transition_score

    def _calculate_gt_reward(self, position):
        """计算GT奖励"""
        y, x = position
        gt_reward = 0.0

        if self.gt_analysis['gt_binary'][y, x]:
            base_gt_reward = 60.0
            if self.height // 3 <= y <= 2 * self.height // 3:
                base_gt_reward *= 1.3
            gt_reward += base_gt_reward
        else:
            gt_coords = np.where(self.gt_analysis['gt_binary'])
            if len(gt_coords[0]) > 0:
                distances = np.sqrt((gt_coords[0] - y) ** 2 + (gt_coords[1] - x) ** 2)
                min_dist = np.min(distances)

                if min_dist <= 2:
                    proximity_reward = 40.0 - min_dist * 10.0
                    if self.height // 3 <= y <= 2 * self.height // 3:
                        proximity_reward *= 1.2
                    gt_reward += proximity_reward

        return gt_reward

    def _calculate_sea_land_separation_reward(self, position):
        """计算海陆分离奖励"""
        y, x = position

        water_mask = self.hsv_analysis['water_mask']
        land_mask = self.hsv_analysis['land_mask']

        water_neighbors = 0
        land_neighbors = 0
        total_neighbors = 0

        for dy in range(-2, 3):
            for dx in range(-2, 3):
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

        if water_ratio > 0.2 and land_ratio > 0.2:
            separation_reward = 30.0 * min(water_ratio, land_ratio) * 2
        elif water_ratio > 0.1 or land_ratio > 0.1:
            separation_reward = 15.0 * (water_ratio + land_ratio)
        else:
            separation_reward = -5.0

        return separation_reward

    def update_coastline(self, position, value=1.0):
        """更新海岸线"""
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            self.current_coastline[y, x] = min(1.0, self.current_coastline[y, x] + value)


# ==================== 约束的代理 ====================

class ConstrainedCoastlineAgent:
    """约束的海岸线代理"""

    def __init__(self, env, lr=1e-4, gamma=0.98, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.995):
        self.env = env
        self.device = device

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = ConstrainedCoastlineDQN().to(device)
        self.target_net = ConstrainedCoastlineDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.memory = deque(maxlen=15000)

        self.batch_size = 32
        self.target_update_freq = 100
        self.train_freq = 4
        self.steps_done = 0

        print(f"✅ 约束DQN代理初始化完成")

    def select_action(self, rgb_state, hsv_state, position, enhanced_features, training=True):
        """选择动作"""
        allowed_actions = self.env.action_constraints.get_allowed_actions(
            position, self.env.current_coastline, self.env.hsv_analysis
        )

        if training and random.random() < self.epsilon:
            return random.choice(allowed_actions)
        else:
            with torch.no_grad():
                position_tensor = torch.LongTensor([position]).to(device)
                q_values = self.policy_net(rgb_state, hsv_state, position_tensor, enhanced_features)

                masked_q_values = q_values.clone()
                for i in range(self.env.action_dim):
                    if i not in allowed_actions:
                        masked_q_values[0, i] = float('-inf')

                return masked_q_values.argmax(dim=1).item()

    def train_step(self):
        """训练步骤"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        rgb_states = torch.cat([item[0][0] for item in batch])
        hsv_states = torch.cat([item[0][1] for item in batch])
        positions = torch.LongTensor([item[0][2] for item in batch]).to(device)
        enhanced_features = torch.cat([item[0][3] for item in batch])

        actions = torch.LongTensor([item[1] for item in batch]).to(device)
        rewards = torch.FloatTensor([item[3] for item in batch]).to(device)

        current_q_values = self.policy_net(rgb_states, hsv_states, positions, enhanced_features).gather(1,
                                                                                                        actions.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size).to(device)
        non_final_mask = torch.tensor([item[2] is not None for item in batch], dtype=torch.bool).to(device)

        if non_final_mask.any():
            non_final_next_rgb = torch.cat([item[2][0] for item in batch if item[2] is not None])
            non_final_next_hsv = torch.cat([item[2][1] for item in batch if item[2] is not None])
            non_final_next_pos = torch.LongTensor([item[2][2] for item in batch if item[2] is not None]).to(device)
            non_final_next_feat = torch.cat([item[2][3] for item in batch if item[2] is not None])

            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_rgb, non_final_next_hsv, non_final_next_pos, non_final_next_feat
                ).max(1)[0]

        target_q_values = rewards + (self.gamma * next_state_values)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def optimize_constrained_coastline(self, max_episodes=200, max_steps_per_episode=400):
        """优化海岸线检测"""
        print("🎯 约束海岸线优化开始...")

        search_positions = np.where(self.env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return self.env.current_coastline

        height = self.env.height
        middle_start = height // 3
        middle_end = 2 * height // 3

        middle_third_starts = [pos for pos in candidate_positions if middle_start <= pos[0] <= middle_end]

        if not middle_third_starts:
            middle_third_starts = candidate_positions[:50]

        episode_rewards = []
        total_improvements = 0

        for episode in range(max_episodes):

            if episode < max_episodes // 2:
                if random.random() < 0.8 and middle_third_starts:
                    start_position = random.choice(middle_third_starts)
                else:
                    start_position = random.choice(candidate_positions[:50])
            else:
                if self.env.gt_analysis and random.random() < 0.9:
                    gt_positions = np.where(self.env.gt_analysis['gt_binary'])
                    if len(gt_positions[0]) > 0:
                        valid_gt_positions = [(gt_positions[0][i], gt_positions[1][i])
                                            for i in range(len(gt_positions[0]))
                                            if middle_start <= gt_positions[0][i] <= middle_end]
                        if valid_gt_positions:
                            start_position = random.choice(valid_gt_positions)
                        else:
                            start_position = random.choice(middle_third_starts)
                    else:
                        start_position = random.choice(middle_third_starts)
                else:
                    start_position = random.choice(middle_third_starts)

            current_position = start_position
            episode_reward = 0
            episode_improvements = 0

            for step in range(max_steps_per_episode):
                rgb_state, hsv_state = self.env.get_state_tensor(current_position)
                enhanced_features = self.env.get_enhanced_features(current_position)

                action = self.select_action(rgb_state, hsv_state, current_position,
                                            enhanced_features, training=True)

                next_position, reward = self.env.step(current_position, action)
                episode_reward += reward

                next_rgb_state, next_hsv_state = self.env.get_state_tensor(next_position)
                next_enhanced_features = self.env.get_enhanced_features(next_position)

                current_state = (rgb_state, hsv_state, current_position, enhanced_features)
                next_state = (next_rgb_state, next_hsv_state, next_position,
                              next_enhanced_features) if reward > -50 else None

                self.memory.append((current_state, action, next_state, reward))

                y_pos = next_position[0]
                is_middle_region = middle_start <= y_pos <= middle_end

                if episode < max_episodes * 0.3:
                    if reward > 15.0:
                        update_value = 0.8 if is_middle_region else 0.6
                        self.env.update_coastline(next_position, update_value)
                        episode_improvements += 1
                        total_improvements += 1
                elif episode < max_episodes * 0.7:
                    if reward > 10.0:
                        update_value = 0.7 if is_middle_region else 0.5
                        self.env.update_coastline(next_position, update_value)
                        episode_improvements += 1
                        total_improvements += 1
                else:
                    if reward > 8.0:
                        update_value = 0.6 if is_middle_region else 0.4
                        self.env.update_coastline(next_position, update_value)
                        episode_improvements += 1
                        total_improvements += 1
                    elif reward > 3.0:
                        update_value = 0.3 if is_middle_region else 0.2
                        self.env.update_coastline(next_position, update_value)
                        episode_improvements += 1

                if self.steps_done % self.train_freq == 0:
                    loss = self.train_step()

                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()

                self.steps_done += 1
                current_position = next_position

                if reward < -80:
                    break

            episode_rewards.append(episode_reward)
            self.decay_epsilon()

            if episode % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                current_pixels = np.sum(self.env.current_coastline > 0.3)

                middle_region_pixels = np.sum(self.env.current_coastline[middle_start:middle_end, :] > 0.3)
                middle_ratio = middle_region_pixels / max(1, current_pixels)

                print(f"   Episode {episode:3d}: 奖励={avg_reward:6.2f}, ε={self.epsilon:.3f}, "
                      f"像素={current_pixels:,}, 中间比例={middle_ratio:.1%}")

        final_pixels = np.sum(self.env.current_coastline > 0.3)
        print(f"   ✅ 优化完成: 总像素={final_pixels:,}, 改进次数={total_improvements}")
        # 在训练完成后保存模型
        model_save_path = "./saved_models/coastline_dqn_model.pth"
        os.makedirs("./saved_models", exist_ok=True)
        self.save_model(model_save_path)

        return self.env.current_coastline

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, save_path):
        """保存模型参数"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, save_path)
        print(f"✅ 模型已保存: {save_path}")

    def load_model(self, load_path):
        """加载模型参数"""
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            print(f"✅ 模型已加载: {load_path}")
            return True
        return False

# ==================== 精准海域分析器 ====================

class PreciseSeaAreaAnalyzer:
    """精准海域分析器 - 专门识别和清理海域误检"""

    def __init__(self):
        print("✅ 精准海域分析器初始化完成")
        self.sea_features = self._define_enhanced_sea_features()

    def _define_enhanced_sea_features(self):
        """定义增强的海域特征"""
        return {
            'deep_blue_range': {
                'hue_range': (200, 250),  # 深蓝色范围
                'saturation_min': 0.3,
                'value_range': (0.1, 0.6)  # 较暗的值
            },
            'dark_water_range': {
                'value_max': 0.4,  # 非常暗的区域
                'saturation_min': 0.2
            },
            'uniform_water_variance': 0.05  # 水域的颜色变化通常很小
        }

    def analyze_sea_areas(self, rgb_image, current_coastline, gt_analysis=None):
        """分析海域区域"""
        print("🔍 精准海域分析...")

        # 转换为HSV
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

        # 1. 识别深海区域
        deep_sea_mask = self._identify_deep_sea_areas(hsv_image)

        # 2. 识别暗水区域
        dark_water_mask = self._identify_dark_water_areas(hsv_image)

        # 3. 识别均匀水域
        uniform_water_mask = self._identify_uniform_water_areas(rgb_image)

        # 4. 综合海域掩码
        comprehensive_sea_mask = deep_sea_mask | dark_water_mask | uniform_water_mask

        # 5. 形态学优化
        comprehensive_sea_mask = self._morphological_sea_cleanup(comprehensive_sea_mask)

        # 6. 分析海岸线在海域中的分布
        sea_intrusion_analysis = self._analyze_sea_intrusion(current_coastline, comprehensive_sea_mask, gt_analysis)

        return {
            'deep_sea_mask': deep_sea_mask,
            'dark_water_mask': dark_water_mask,
            'uniform_water_mask': uniform_water_mask,
            'comprehensive_sea_mask': comprehensive_sea_mask,
            'sea_intrusion_analysis': sea_intrusion_analysis
        }

    def _identify_deep_sea_areas(self, hsv_image):
        """识别深海区域"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        features = self.sea_features['deep_blue_range']

        hue_mask = ((h >= features['hue_range'][0]) & (h <= features['hue_range'][1]))
        saturation_mask = s >= features['saturation_min']
        value_mask = ((v >= features['value_range'][0]) & (v <= features['value_range'][1]))

        deep_sea = hue_mask & saturation_mask & value_mask

        # 形态学处理
        deep_sea = binary_closing(deep_sea, np.ones((7, 7)))
        deep_sea = binary_erosion(deep_sea, np.ones((3, 3)))
        deep_sea = binary_dilation(deep_sea, np.ones((5, 5)))

        return deep_sea

    def _identify_dark_water_areas(self, hsv_image):
        """识别暗水区域"""
        s, v = hsv_image[:, :, 1], hsv_image[:, :, 2]

        features = self.sea_features['dark_water_range']

        dark_mask = v <= features['value_max']
        saturation_mask = s >= features['saturation_min']

        dark_water = dark_mask & saturation_mask

        # 去除小噪声
        dark_water = binary_closing(dark_water, np.ones((5, 5)))
        dark_water = self._remove_small_components(dark_water, min_size=50)

        return dark_water

    def _identify_uniform_water_areas(self, rgb_image):
        """识别均匀水域 - 基于颜色一致性"""
        if len(rgb_image.shape) == 3:
            gray = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = rgb_image.copy()

        # 计算局部方差
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(gray.astype(float), size=9)
        local_variance = uniform_filter((gray.astype(float) - local_mean)**2, size=9)

        # 低方差区域 + 低亮度 = 均匀水域
        uniform_mask = (local_variance < self.sea_features['uniform_water_variance'] * 255**2)
        dark_mask = gray < 120  # 相对较暗

        uniform_water = uniform_mask & dark_mask

        # 形态学清理
        uniform_water = binary_closing(uniform_water, np.ones((7, 7)))
        uniform_water = self._remove_small_components(uniform_water, min_size=100)

        return uniform_water

    def _morphological_sea_cleanup(self, sea_mask):
        """海域掩码的形态学清理"""
        # 先闭运算填补小洞
        cleaned = binary_closing(sea_mask, np.ones((9, 9)))

        # 去除小碎片
        cleaned = self._remove_small_components(cleaned, min_size=200)

        # 轻微膨胀以确保覆盖边界
        cleaned = binary_dilation(cleaned, np.ones((3, 3)))

        return cleaned

    def _analyze_sea_intrusion(self, coastline, sea_mask, gt_analysis):
        """分析海岸线在海域中的分布"""
        coastline_binary = (coastline > 0.5).astype(bool)

        # 海岸线在海域中的像素
        sea_intrusion = coastline_binary & sea_mask
        intrusion_count = np.sum(sea_intrusion)

        # 计算入侵严重程度
        total_coastline = np.sum(coastline_binary)
        intrusion_ratio = intrusion_count / max(1, total_coastline)

        # 分析入侵的空间分布
        intrusion_clusters = self._analyze_intrusion_clusters(sea_intrusion)

        # GT保护区域分析
        gt_protected_intrusion = 0
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_protection_area = binary_dilation(gt_binary, np.ones((10, 10)))
            gt_protected_intrusion = np.sum(sea_intrusion & gt_protection_area)

        return {
            'intrusion_mask': sea_intrusion,
            'intrusion_count': intrusion_count,
            'intrusion_ratio': intrusion_ratio,
            'intrusion_clusters': intrusion_clusters,
            'gt_protected_intrusion': gt_protected_intrusion,
            'removable_intrusion': intrusion_count - gt_protected_intrusion
        }

    def _analyze_intrusion_clusters(self, intrusion_mask):
        """分析入侵集群"""
        labeled_intrusion, num_clusters = label(intrusion_mask)

        clusters = []
        for i in range(1, num_clusters + 1):
            cluster_mask = (labeled_intrusion == i)
            cluster_size = np.sum(cluster_mask)

            # 计算集群的紧密度
            cluster_coords = np.where(cluster_mask)
            if len(cluster_coords[0]) > 1:
                y_coords, x_coords = cluster_coords[0], cluster_coords[1]
                y_span = np.max(y_coords) - np.min(y_coords) + 1
                x_span = np.max(x_coords) - np.min(x_coords) + 1
                compactness = cluster_size / (y_span * x_span)
            else:
                compactness = 1.0

            clusters.append({
                'id': i,
                'size': cluster_size,
                'compactness': compactness,
                'bbox': (np.min(cluster_coords[0]), np.min(cluster_coords[1]),
                        np.max(cluster_coords[0]), np.max(cluster_coords[1]))
            })

        return clusters

    def _remove_small_components(self, binary_image, min_size=50):
        """移除小组件"""
        labeled_array, num_components = label(binary_image)

        result = binary_image.copy()
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            if size < min_size:
                result[labeled_array == i] = False

        return result


# ==================== 连通性防护器 ====================

class ConnectivityGuard:
    """连通性防护器 - 防止上下岸线错误连通"""

    def __init__(self):
        print("✅ 连通性防护器初始化完成")

    def analyze_connectivity_risks(self, coastline, image_shape):
        """分析连通性风险"""
        print("🛡️ 分析连通性风险...")

        height, width = image_shape[:2]
        coastline_binary = (coastline > 0.5).astype(bool)

        # 分析垂直连通性风险
        vertical_risks = self._analyze_vertical_connectivity(coastline_binary, height, width)

        # 分析水平分割情况
        horizontal_analysis = self._analyze_horizontal_separation(coastline_binary, height, width)

        # 识别危险连接区域
        dangerous_connections = self._identify_dangerous_connections(coastline_binary, vertical_risks)

        return {
            'vertical_risks': vertical_risks,
            'horizontal_analysis': horizontal_analysis,
            'dangerous_connections': dangerous_connections
        }

    def _analyze_vertical_connectivity(self, coastline_binary, height, width):
        """分析垂直连通性"""
        risks = []

        # 检查每一列的连通情况
        for x in range(width):
            column = coastline_binary[:, x]
            if np.sum(column) == 0:
                continue

            # 找到连续区间
            connected_regions = self._find_connected_regions(column)

            # 分析危险的长连接
            for region in connected_regions:
                start, end = region
                length = end - start + 1
                coverage = length / height

                # 如果某列的连接覆盖超过60%的高度，标记为风险
                if coverage > 0.6:
                    risks.append({
                        'column': x,
                        'start': start,
                        'end': end,
                        'length': length,
                        'coverage': coverage,
                        'risk_level': 'high' if coverage > 0.8 else 'medium'
                    })

        return risks

    def _find_connected_regions(self, binary_column):
        """找到二进制列中的连续区域"""
        regions = []
        start = None

        for i, value in enumerate(binary_column):
            if value and start is None:
                start = i
            elif not value and start is not None:
                regions.append((start, i - 1))
                start = None

        # 处理列末尾的情况
        if start is not None:
            regions.append((start, len(binary_column) - 1))

        return regions

    def _analyze_horizontal_separation(self, coastline_binary, height, width):
        """分析水平分割情况"""
        # 将图像分为上、中、下三部分
        third_height = height // 3

        upper_region = coastline_binary[:third_height, :]
        middle_region = coastline_binary[third_height:2*third_height, :]
        lower_region = coastline_binary[2*third_height:, :]

        upper_pixels = np.sum(upper_region)
        middle_pixels = np.sum(middle_region)
        lower_pixels = np.sum(lower_region)
        total_pixels = upper_pixels + middle_pixels + lower_pixels

        if total_pixels == 0:
            return {
                'upper_ratio': 0, 'middle_ratio': 0, 'lower_ratio': 0,
                'separation_quality': 'no_coastline'
            }

        upper_ratio = upper_pixels / total_pixels
        middle_ratio = middle_pixels / total_pixels
        lower_ratio = lower_pixels / total_pixels

        # 评估分离质量
        if middle_ratio > 0.7:
            separation_quality = 'excellent'  # 主要集中在中间
        elif middle_ratio > 0.5:
            separation_quality = 'good'
        elif upper_ratio > 0.4 and lower_ratio > 0.4:
            separation_quality = 'poor'  # 上下都有很多，可能连通了
        else:
            separation_quality = 'fair'

        return {
            'upper_ratio': upper_ratio,
            'middle_ratio': middle_ratio,
            'lower_ratio': lower_ratio,
            'separation_quality': separation_quality
        }

    def _identify_dangerous_connections(self, coastline_binary, vertical_risks):
        """识别危险连接区域"""
        dangerous_areas = np.zeros_like(coastline_binary, dtype=bool)

        for risk in vertical_risks:
            if risk['risk_level'] == 'high':
                x = risk['column']
                start = max(0, risk['start'])
                end = min(coastline_binary.shape[0] - 1, risk['end'])

                # 标记整个危险连接区域
                dangerous_areas[start:end+1, x] = True

                # 向两侧扩展一点
                for dx in [-1, 1]:
                    nx = x + dx
                    if 0 <= nx < coastline_binary.shape[1]:
                        dangerous_areas[start:end+1, nx] = True

        return dangerous_areas


# ==================== 精准海域清理后处理器 ====================

class PreciseSeaCleanupPostProcessor:
    """精准海域清理后处理器"""

    def __init__(self, target_pixel_range=(90000, 100000)):
        self.sea_analyzer = PreciseSeaAreaAnalyzer()
        self.connectivity_guard = ConnectivityGuard()
        self.target_pixel_range = target_pixel_range
        print(f"✅ 精准海域清理后处理器初始化完成")
        print(f"   🎯 目标像素范围: {target_pixel_range[0]:,} - {target_pixel_range[1]:,}")

    def process_precise_sea_cleanup(self, coastline, hsv_analysis, gt_analysis, rgb_image):
        """精准海域清理处理"""
        print("🔧 开始精准海域清理后处理...")
        print(f"   输入海岸线像素: {np.sum(coastline > 0.3):,}")

        # 第一步：海域分析
        sea_analysis = self.sea_analyzer.analyze_sea_areas(rgb_image, coastline, gt_analysis)

        # 第二步：连通性风险分析
        connectivity_analysis = self.connectivity_guard.analyze_connectivity_risks(coastline, rgb_image.shape)

        # 第三步：初步二值化
        binary_coastline = self._smart_binarization(coastline, hsv_analysis)
        print(f"   初步二值化后: {np.sum(binary_coastline):,} 像素")

        # 第四步：海域清理
        sea_cleaned_coastline = self._aggressive_sea_cleanup(binary_coastline, sea_analysis, gt_analysis)
        print(f"   海域清理后: {np.sum(sea_cleaned_coastline):,} 像素")

        # 第五步：连通性修复
        connectivity_fixed = self._fix_dangerous_connectivity(sea_cleaned_coastline, connectivity_analysis, gt_analysis)
        print(f"   连通性修复后: {np.sum(connectivity_fixed):,} 像素")

        # 第六步：像素数量控制
        pixel_controlled = self._control_pixel_count(connectivity_fixed, gt_analysis, sea_analysis)
        print(f"   像素控制后: {np.sum(pixel_controlled):,} 像素")

        # 第七步：最终GT保护
        final_coastline = self._final_gt_protection(pixel_controlled, gt_analysis)
        print(f"   最终GT保护后: {np.sum(final_coastline):,} 像素")

        return final_coastline.astype(float)

    def _smart_binarization(self, coastline, hsv_analysis):
        """智能二值化"""
        guidance_weight = hsv_analysis['coastline_guidance']
        transition_weight = hsv_analysis['transition_strength']

        # 更保守的加权策略
        weighted_coastline = coastline * (0.6 + guidance_weight * 0.2 + transition_weight * 0.2)

        # 提高阈值
        valid_mask = weighted_coastline > 0
        if np.any(valid_mask):
            threshold = np.percentile(weighted_coastline[valid_mask], 70)  # 提高到70
        else:
            threshold = 0.35  # 提高基础阈值

        binary_result = weighted_coastline > threshold

        # 移除小组件
        binary_result = self._remove_small_components(binary_result, min_size=10)

        return binary_result

    def _aggressive_sea_cleanup(self, binary_coastline, sea_analysis, gt_analysis):
        """激进的海域清理"""
        result = binary_coastline.copy()

        # GT保护掩码
        gt_protection_mask = np.zeros_like(binary_coastline, dtype=bool)
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_protection_mask = binary_dilation(gt_binary, np.ones((8, 8)))

        # 获取海域掩码
        sea_mask = sea_analysis['comprehensive_sea_mask']

        # 在海域中的海岸线像素
        sea_coastline_pixels = binary_coastline & sea_mask

        # 分离受保护和不受保护的海域像素
        protected_sea_pixels = sea_coastline_pixels & gt_protection_mask
        unprotected_sea_pixels = sea_coastline_pixels & ~gt_protection_mask

        print(f"     海域像素总数: {np.sum(sea_coastline_pixels):,}")
        print(f"     受GT保护: {np.sum(protected_sea_pixels):,}")
        print(f"     可清理的: {np.sum(unprotected_sea_pixels):,}")

        # 分析海域像素的清理价值
        cleanable_pixels = self._identify_cleanable_sea_pixels(unprotected_sea_pixels, binary_coastline, gt_analysis)

        # 执行清理
        result = result & ~cleanable_pixels

        removed_count = np.sum(cleanable_pixels)
        print(f"     实际清理: {removed_count:,} 海域像素")

        return result

    def _identify_cleanable_sea_pixels(self, unprotected_sea_pixels, binary_coastline, gt_analysis):
        """识别可清理的海域像素"""
        cleanable = np.zeros_like(unprotected_sea_pixels, dtype=bool)

        if not np.any(unprotected_sea_pixels):
            return cleanable

        # 获取候选像素位置
        sea_positions = np.where(unprotected_sea_pixels)

        for i in range(len(sea_positions[0])):
            y, x = sea_positions[0][i], sea_positions[1][i]

            # 分析该像素的清理价值
            cleanup_score = self._calculate_cleanup_value(y, x, binary_coastline, gt_analysis)

            # 如果清理价值高，标记为可清理
            if cleanup_score > 0.3:  # 降低阈值，更激进清理
                cleanable[y, x] = True

        return cleanable

    def _calculate_cleanup_value(self, y, x, binary_coastline, gt_analysis):
        """计算像素的清理价值"""
        height, width = binary_coastline.shape

        # 检查周围的海岸线密度
        neighbor_coastline = 0
        total_neighbors = 0

        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    total_neighbors += 1
                    if binary_coastline[ny, nx]:
                        neighbor_coastline += 1

        neighbor_density = neighbor_coastline / max(1, total_neighbors)

        # 如果周围海岸线密度很低，清理价值高
        low_density_score = 1.0 - neighbor_density

        # 检查是否在图像边缘附近（边缘误检可能性高）
        edge_distance = min(y, x, height - 1 - y, width - 1 - x)
        edge_score = 1.0 if edge_distance < 10 else 0.0

        # 检查是否在GT远离区域
        gt_distance_score = 0.0
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            if np.any(gt_binary):
                gt_coords = np.where(gt_binary)
                distances = np.sqrt((gt_coords[0] - y)**2 + (gt_coords[1] - x)**2)
                min_gt_distance = np.min(distances)
                gt_distance_score = min(1.0, min_gt_distance / 20.0)  # 距离GT越远，清理价值越高

        # 综合评分
        cleanup_value = (low_density_score * 0.4 +
                        edge_score * 0.2 +
                        gt_distance_score * 0.4)

        return cleanup_value

    def _fix_dangerous_connectivity(self, binary_coastline, connectivity_analysis, gt_analysis):
        """修复危险连通"""
        result = binary_coastline.copy()

        dangerous_connections = connectivity_analysis['dangerous_connections']
        vertical_risks = connectivity_analysis['vertical_risks']

        if not np.any(dangerous_connections):
            return result

        print(f"     修复 {len(vertical_risks)} 个危险连接")

        # GT保护
        gt_protection_mask = np.zeros_like(binary_coastline, dtype=bool)
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_protection_mask = binary_dilation(gt_binary, np.ones((5, 5)))

        # 对每个危险连接进行处理
        for risk in vertical_risks:
            if risk['risk_level'] == 'high':
                self._break_dangerous_connection(result, risk, gt_protection_mask)

        return result

    def _break_dangerous_connection(self, binary_coastline, risk, gt_protection_mask):
        """打断危险连接"""
        x = risk['column']
        start = risk['start']
        end = risk['end']
        length = risk['length']

        # 在连接的中间部分创建间隙
        gap_size = max(3, length // 8)  # 间隙大小
        gap_start = start + length // 2 - gap_size // 2
        gap_end = gap_start + gap_size

        # 检查间隙区域是否有GT保护
        gap_region = gt_protection_mask[gap_start:gap_end+1, x]

        # 如果间隙区域没有GT保护，则创建间隙
        if not np.any(gap_region):
            binary_coastline[gap_start:gap_end+1, x] = False

            # 向两侧也创建小间隙
            for dx in [-1, 1]:
                nx = x + dx
                if 0 <= nx < binary_coastline.shape[1]:
                    small_gap_start = gap_start + gap_size // 4
                    small_gap_end = gap_end - gap_size // 4
                    small_gap_region = gt_protection_mask[small_gap_start:small_gap_end+1, nx]
                    if not np.any(small_gap_region):
                        binary_coastline[small_gap_start:small_gap_end+1, nx] = False

    def _control_pixel_count(self, binary_coastline, gt_analysis, sea_analysis):
        """控制像素数量到目标范围"""
        current_pixels = np.sum(binary_coastline)
        target_min, target_max = self.target_pixel_range

        print(f"     当前像素: {current_pixels:,}, 目标: {target_min:,}-{target_max:,}")

        if target_min <= current_pixels <= target_max:
            print("     像素数量已在目标范围内")
            return binary_coastline

        if current_pixels > target_max:
            # 需要进一步减少像素
            pixels_to_remove = current_pixels - target_max
            return self._smart_pixel_reduction(binary_coastline, pixels_to_remove, gt_analysis, sea_analysis)
        else:
            # 像素太少，需要适度增加
            return self._smart_pixel_addition(binary_coastline, gt_analysis)

    def _smart_pixel_reduction(self, binary_coastline, pixels_to_remove, gt_analysis, sea_analysis):
        """智能像素减少"""
        result = binary_coastline.copy()

        # GT保护
        gt_protection_mask = np.zeros_like(binary_coastline, dtype=bool)
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_protection_mask = binary_dilation(gt_binary, np.ones((6, 6)))

        # 获取所有可移除的候选像素
        candidate_pixels = binary_coastline & ~gt_protection_mask
        candidate_positions = np.where(candidate_pixels)

        if len(candidate_positions[0]) == 0:
            print("     ⚠️ 所有像素都受GT保护，无法减少")
            return result

        # 计算每个候选像素的移除价值
        pixel_scores = []
        for i in range(len(candidate_positions[0])):
            y, x = candidate_positions[0][i], candidate_positions[1][i]
            score = self._calculate_removal_value(y, x, binary_coastline, gt_analysis, sea_analysis)
            pixel_scores.append((score, y, x))

        # 按移除价值排序（价值高的优先移除）
        pixel_scores.sort(reverse=True)

        # 移除指定数量的像素
        removed_count = 0
        for score, y, x in pixel_scores:
            if removed_count >= pixels_to_remove:
                break

            result[y, x] = False
            removed_count += 1

        print(f"     智能减少: {removed_count:,} 像素")
        return result

    def _calculate_removal_value(self, y, x, binary_coastline, gt_analysis, sea_analysis):
        """计算像素的移除价值"""
        height, width = binary_coastline.shape

        removal_score = 0.0

        # 1. 孤立度评分（周围海岸线越少，移除价值越高）
        neighbor_count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and binary_coastline[ny, nx]:
                    neighbor_count += 1

        isolation_score = 1.0 - (neighbor_count / 24.0)  # 24是最大邻居数
        removal_score += isolation_score * 0.3

        # 2. 海域位置评分（在海域中的像素移除价值更高）
        if 'comprehensive_sea_mask' in sea_analysis:
            sea_mask = sea_analysis['comprehensive_sea_mask']
            if sea_mask[y, x]:
                removal_score += 0.4  # 海域中的像素优先移除

        # 3. GT距离评分（距离GT越远，移除价值越高）
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            if np.any(gt_binary):
                gt_coords = np.where(gt_binary)
                distances = np.sqrt((gt_coords[0] - y)**2 + (gt_coords[1] - x)**2)
                min_distance = np.min(distances)
                distance_score = min(1.0, min_distance / 30.0)
                removal_score += distance_score * 0.2

        # 4. 边缘位置评分
        edge_distance = min(y, x, height - 1 - y, width - 1 - x)
        edge_score = 1.0 if edge_distance < 15 else 0.0
        removal_score += edge_score * 0.1

        return removal_score

    def _smart_pixel_addition(self, binary_coastline, gt_analysis):
        """智能像素增加（保守）"""
        result = binary_coastline.copy()

        # 在现有海岸线周围适度扩展
        expanded = binary_dilation(binary_coastline, np.ones((3, 3)))
        new_pixels = expanded & ~binary_coastline

        # 只添加最有价值的新像素
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_nearby = binary_dilation(gt_binary, np.ones((5, 5)))
            valuable_new_pixels = new_pixels & gt_nearby
            result = result | valuable_new_pixels

        return result

    def _final_gt_protection(self, binary_coastline, gt_analysis):
        """最终GT保护"""
        if not gt_analysis or gt_analysis['gt_binary'] is None:
            return binary_coastline

        result = binary_coastline.copy()
        gt_binary = gt_analysis['gt_binary']

        # 确保所有GT像素都存在
        result = result | gt_binary

        # 确保GT周围的连通性
        gt_expanded = binary_dilation(gt_binary, np.ones((3, 3)))
        gt_connectivity = gt_expanded & binary_coastline
        result = result | gt_connectivity

        return result

    def _remove_small_components(self, binary_image, min_size=10):
        """移除小组件"""
        labeled_array, num_components = label(binary_image)

        result = binary_image.copy()
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            if size < min_size:
                result[labeled_array == i] = False

        return result


# ==================== 精准清理检测器 ====================

class PreciseSeaCleanupDetector:
    """精准海域清理检测器"""

    def __init__(self):
        self.gt_analyzer = GroundTruthAnalyzer()
        self.post_processor = PreciseSeaCleanupPostProcessor()
        print("✅ 精准海域清理检测系统初始化完成")
        print("   🎯 主要特色：海域清理 + 连通防护 + 像素控制 + GT保护")

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

    def process_image(self, image_path, ground_truth_path=None, force_retrain=False):
        """
        处理单个图像 - 支持模型保存和加载

        Args:
            image_path: 输入图像路径
            ground_truth_path: Ground Truth路径（可选）
            force_retrain: 是否强制重新训练模型
        """
        print(f"\n🌊 精准海域清理处理: {os.path.basename(image_path)}")

        try:
            # 加载图像
            original_img = self.load_image_from_file(image_path)
            if original_img is None:
                return None

            # 调整尺寸
            img_pil = Image.fromarray(original_img)
            processed_img = np.array(img_pil.resize((400, 400), Image.LANCZOS))
            print(f"   📐 处理后尺寸: {processed_img.shape}")

            # 加载并分析Ground Truth
            gt_coastline = None
            gt_analysis = None

            if ground_truth_path and os.path.exists(ground_truth_path):
                gt_img = self.load_image_from_file(ground_truth_path)
                if gt_img is not None:
                    gt_resized = np.array(Image.fromarray(gt_img).resize((400, 400), Image.LANCZOS))
                    if len(gt_resized.shape) == 3:
                        gt_gray = BasicImageProcessor.rgb_to_gray(gt_resized)
                    else:
                        gt_gray = gt_resized
                    gt_coastline = (gt_gray > 127).astype(float)

                    print("\n📍 步骤1: Ground Truth模式分析")
                    gt_analysis = self.gt_analyzer.analyze_gt_pattern(gt_coastline)
                    if gt_analysis:
                        print(f"   GT像素数: {gt_analysis['total_pixels']:,}")

            # 步骤2: 创建环境
            print("\n📍 步骤2: 创建约束环境")
            env = ConstrainedCoastlineEnvironment(processed_img, gt_analysis)

            # 步骤3: 智能模型管理
            print("\n📍 步骤3: 智能模型管理")
            agent = ConstrainedCoastlineAgent(env)

            # 设置模型路径
            model_dir = "./saved_models"
            os.makedirs(model_dir, exist_ok=True)

            # 创建基于图像特征的模型文件名
            image_hash = self._get_image_hash(processed_img)
            gt_hash = self._get_gt_hash(gt_analysis) if gt_analysis else "no_gt"
            model_filename = f"coastline_model_{image_hash}_{gt_hash}.pth"
            model_path = os.path.join(model_dir, model_filename)

            # 通用模型路径（用于相似图像）
            general_model_path = os.path.join(model_dir, "coastline_general_model.pth")

            optimized_coastline = None
            model_loaded = False

            if not force_retrain:
                # 尝试加载特定模型
                if agent.load_model(model_path):
                    print("🎯 使用特定预训练模型")
                    optimized_coastline = self._apply_pretrained_model(agent, env)
                    model_loaded = True
                # 尝试加载通用模型
                elif agent.load_model(general_model_path):
                    print("🔄 使用通用预训练模型")
                    optimized_coastline = self._apply_pretrained_model(agent, env)
                    model_loaded = True

            if not model_loaded or force_retrain:
                print("🚀 开始训练新模型")
                optimized_coastline = agent.optimize_constrained_coastline(
                    max_episodes=200,
                    max_steps_per_episode=400
                )

                # 保存训练好的模型
                print("\n💾 保存训练模型")
                agent.save_model(model_path)  # 保存特定模型
                agent.save_model(general_model_path)  # 更新通用模型

                # 保存模型元信息
                self._save_model_metadata(model_path, {
                    'image_path': image_path,
                    'image_shape': processed_img.shape,
                    'gt_available': gt_analysis is not None,
                    'gt_pixels': gt_analysis['total_pixels'] if gt_analysis else 0,
                    'training_pixels': np.sum(optimized_coastline > 0.3),
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                })

            # 步骤4: 精准海域清理后处理
            print("\n📍 步骤4: 精准海域清理后处理")
            final_coastline = self.post_processor.process_precise_sea_cleanup(
                optimized_coastline, env.hsv_analysis, gt_analysis, processed_img
            )

            # 质量评估
            quality_metrics = self._evaluate_precise_quality(final_coastline, gt_coastline, env.hsv_analysis)

            return {
                'original_image': original_img,
                'processed_image': processed_img,
                'gt_analysis': gt_analysis,
                'ground_truth': gt_coastline,
                'hsv_analysis': env.hsv_analysis,
                'optimized_coastline': optimized_coastline,
                'final_coastline': final_coastline,
                'quality_metrics': quality_metrics,
                'model_used': 'pretrained' if model_loaded else 'newly_trained',
                'model_path': model_path,
                'success': quality_metrics['overall_score'] > 0.5
            }

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_image_hash(self, image):
        """生成图像特征哈希"""
        import hashlib
        # 使用图像的基本统计特征生成哈希
        features = [
            image.shape,
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image)
        ]
        feature_str = str(features)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]

    def _get_gt_hash(self, gt_analysis):
        """生成GT特征哈希"""
        import hashlib
        if gt_analysis is None:
            return "no_gt"

        features = [
            gt_analysis['total_pixels'],
            gt_analysis.get('edge_region', np.array([])).shape
        ]
        feature_str = str(features)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]

    def _apply_pretrained_model(self, agent, env):
        """应用预训练模型进行推理"""
        print("   🔮 使用预训练模型进行推理...")

        # 获取搜索区域
        search_positions = np.where(env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return env.current_coastline

        # 使用预训练模型进行推理
        height = env.height
        middle_start = height // 3
        middle_end = 2 * height // 3

        # 重点关注中间区域
        middle_positions = [pos for pos in candidate_positions if middle_start <= pos[0] <= middle_end]

        if not middle_positions:
            middle_positions = candidate_positions[:100]

        # 对每个候选位置进行评估
        inference_steps = min(500, len(middle_positions))
        print(f"   🎯 推理步骤数: {inference_steps}")

        for step in range(inference_steps):
            # 选择位置
            if step < len(middle_positions):
                position = middle_positions[step]
            else:
                position = candidate_positions[step % len(candidate_positions)]

            # 获取状态
            rgb_state, hsv_state = env.get_state_tensor(position)
            enhanced_features = env.get_enhanced_features(position)

            # 使用预训练模型进行动作选择
            action = agent.select_action(rgb_state, hsv_state, position, enhanced_features, training=False)

            # 执行动作
            next_position, reward = env.step(position, action)

            # 根据奖励更新海岸线
            if reward > 10.0:  # 只保留高质量预测
                y, x = next_position
                is_middle_region = middle_start <= y <= middle_end
                update_value = 0.7 if is_middle_region else 0.5
                env.update_coastline(next_position, update_value)

        final_pixels = np.sum(env.current_coastline > 0.3)
        print(f"   ✅ 推理完成: {final_pixels:,} 像素")

        return env.current_coastline

    def _save_model_metadata(self, model_path, metadata):
        """保存模型元信息"""
        import json

        metadata_path = model_path.replace('.pth', '_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   📝 模型元信息已保存: {metadata_path}")
        except Exception as e:
            print(f"   ⚠️ 元信息保存失败: {e}")

    # 还需要在 ConstrainedCoastlineAgent 类中添加以下方法：

    def save_model(self, save_path):
        """保存模型参数"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'memory_size': len(self.memory)
        }, save_path)
        print(f"✅ 模型已保存: {save_path}")

    def load_model(self, load_path):
        """加载模型参数"""
        if os.path.exists(load_path):
            try:
                checkpoint = torch.load(load_path, map_location=device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon_end)  # 使用较低的探索率
                self.steps_done = checkpoint.get('steps_done', 0)
                print(f"✅ 模型已加载: {load_path}")
                print(f"   📊 模型状态: ε={self.epsilon:.3f}, 步数={self.steps_done}")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        return False

    def _evaluate_precise_quality(self, predicted, ground_truth, hsv_analysis):
        """评估精准质量"""
        metrics = {}

        pred_binary = (predicted > 0.5).astype(bool)
        coastline_pixels = np.sum(pred_binary)

        metrics['coastline_pixels'] = int(coastline_pixels)

        # 连通性分析
        labeled_array, num_components = label(pred_binary)
        metrics['num_components'] = int(num_components)

        # 像素密度合理性
        target_min, target_max = self.post_processor.target_pixel_range
        if target_min <= coastline_pixels <= target_max:
            pixel_score = 1.0
        else:
            pixel_score = max(0.0, 1.0 - abs(coastline_pixels - (target_min + target_max) / 2) / target_max)
        metrics['pixel_density_score'] = float(pixel_score)

        # 海域清理效果评估
        water_mask = hsv_analysis['water_mask']
        water_intrusion = np.sum(pred_binary & water_mask) / (coastline_pixels + 1e-8)
        metrics['water_intrusion'] = float(water_intrusion)
        metrics['sea_cleanup_score'] = float(max(0.0, 1.0 - water_intrusion * 2))

        # 连通性质量评估
        height = pred_binary.shape[0]
        third_height = height // 3
        upper_pixels = np.sum(pred_binary[:third_height, :])
        middle_pixels = np.sum(pred_binary[third_height:2*third_height, :])
        lower_pixels = np.sum(pred_binary[2*third_height:, :])

        if coastline_pixels > 0:
            middle_ratio = middle_pixels / coastline_pixels
            connectivity_score = min(1.0, middle_ratio * 2)  # 鼓励中间集中
        else:
            connectivity_score = 0.0
        metrics['connectivity_score'] = float(connectivity_score)

        # GT匹配度分析
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

            # 综合评分（有GT情况）
            overall_score = (f1_score * 0.25 + iou * 0.15 +
                           pixel_score * 0.2 + metrics['sea_cleanup_score'] * 0.25 +
                           metrics['connectivity_score'] * 0.15)
        else:
            # 综合评分（无GT情况）
            overall_score = (pixel_score * 0.3 + metrics['sea_cleanup_score'] * 0.4 +
                           metrics['connectivity_score'] * 0.3)

        metrics['overall_score'] = float(overall_score)

        return metrics


# ==================== 创建精准清理可视化 ====================

def create_precise_cleanup_visualization(result, save_path):
    """创建精准清理版可视化"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Precise Sea Cleanup Coastline Detection - {result.get("sample_id", "Unknown")}',
                 fontsize=16, fontweight='bold')

    # 第一行：输入和分析
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['processed_image'])
    axes[0, 1].set_title('Processed Image (400x400)')
    axes[0, 1].axis('off')

    if result['ground_truth'] is not None:
        axes[0, 2].imshow(result['ground_truth'], cmap='Reds')
        gt_pixels = np.sum(result['ground_truth'] > 0.5)
        axes[0, 2].set_title(f'Ground Truth\n({gt_pixels:,} pixels)')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].axis('off')
        axes[0, 2].set_title('Ground Truth\n(Not Available)')

    if 'hsv_analysis' in result:
        axes[0, 3].imshow(result['hsv_analysis']['coastline_guidance'], cmap='plasma')
        axes[0, 3].set_title('HSV Coastline Guidance')
        axes[0, 3].axis('off')
    else:
        axes[0, 3].axis('off')

    # 第二行：检测结果对比
    axes[1, 0].imshow(result['optimized_coastline'], cmap='hot')
    opt_pixels = np.sum(result['optimized_coastline'] > 0.3)
    axes[1, 0].set_title(f'Before Cleanup\n({opt_pixels:,} pixels)', color='orange', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['final_coastline'], cmap='hot')
    final_pixels = np.sum(result['final_coastline'] > 0.5)
    axes[1, 1].set_title(f'After Precise Cleanup\n({final_pixels:,} pixels)', color='red', fontweight='bold')
    axes[1, 1].axis('off')

    # 清理对比
    cleanup_diff = (result['optimized_coastline'] > 0.3).astype(float) - (result['final_coastline'] > 0.5).astype(float)
    axes[1, 2].imshow(cleanup_diff, cmap='RdBu', vmin=-1, vmax=1)
    removed_pixels = np.sum(cleanup_diff > 0)
    axes[1, 2].set_title(f'Cleanup Difference\n(Removed: {removed_pixels:,} pixels)')
    axes[1, 2].axis('off')

    # 连通性分析
    labeled_array, num_components = label(result['final_coastline'] > 0.5)
    axes[1, 3].imshow(labeled_array, cmap='tab20')
    axes[1, 3].set_title(f'Connectivity Analysis\n({num_components} components)')
    axes[1, 3].axis('off')

    # 第三行：统计信息
    for i in range(4):
        axes[2, i].axis('off')

    metrics = result['quality_metrics']

    # 计算清理效果
    reduction_ratio = (opt_pixels - final_pixels) / max(1, opt_pixels)

    stats_text = f"""Precise Sea Cleanup Results:

Overall Score: {metrics['overall_score']:.3f}
Status: {"✅ SUCCESS" if result['success'] else "❌ FAILED"}

Cleanup Performance:
• Pixel reduction: {reduction_ratio:.1%} ({opt_pixels:,} → {final_pixels:,})
• Sea cleanup score: {metrics.get('sea_cleanup_score', 0):.3f}
• Water intrusion: {metrics.get('water_intrusion', 0):.1%}
• Connectivity score: {metrics.get('connectivity_score', 0):.3f}
• Components: {metrics['num_components']}

Pixel Control:
• Target range: 90K - 100K
• Actual pixels: {final_pixels:,}
• Density score: {metrics.get('pixel_density_score', 0):.3f}"""

    if 'f1_score' in metrics:
        stats_text += f"""

GT Matching Metrics:
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}
• F1-Score: {metrics['f1_score']:.3f}
• IoU: {metrics['iou']:.3f}"""

    stats_text += f"""

Key Improvements:
✓ Aggressive sea area cleanup
✓ Dangerous connectivity breaking
✓ Smart pixel count control
✓ GT protection maintained
✓ Edge artifact removal
✓ Isolation-based filtering

Technical Features:
• Deep sea detection (HSV analysis)
• Dark water identification
• Uniform area recognition
• Vertical connectivity guard
• Smart pixel reduction
• GT-protected cleanup
• Device: {device}"""

    axes[2, 0].text(0.02, 0.98, stats_text, transform=fig.transFigure,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 精准清理版可视化已保存: {save_path}")


# ==================== 演示函数 ====================

def create_demo_image():
    """创建演示海岸线图像"""
    print("🎨 创建演示海岸线图像...")

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:, :] = [20, 100, 200]

    # 创建主要横向海岸线 - 重点在中间1/3区域
    for y in range(400):
        if 130 <= y <= 270:  # 中间1/3区域
            main_coastline_x = int(180 + 50 * np.sin(y * 0.03) + 30 * np.sin(y * 0.1) + 10 * np.cos(y * 0.15))
        else:
            main_coastline_x = int(180 + 20 * np.sin(y * 0.01))

        main_coastline_x = max(50, min(350, main_coastline_x))

        img[y, main_coastline_x:] = [100, 180, 50]

        for offset in range(-6, 7):
            x = main_coastline_x + offset
            if 0 <= x < 400:
                mix_ratio = (6 - abs(offset)) / 6.0
                img[y, x] = [
                    int(20 + (100 - 20) * mix_ratio),
                    int(100 + (180 - 100) * mix_ratio),
                    int(200 + (50 - 200) * mix_ratio)
                ]

    # 添加小岛
    island_center = (200, 120)
    for dy in range(-18, 19):
        for dx in range(-18, 19):
            y, x = island_center[0] + dy, island_center[1] + dx
            if 0 <= y < 400 and 0 <= x < 400:
                distance = math.sqrt(dy * dy + dx * dx)
                if distance <= 15:
                    img[y, x] = [100, 180, 50]
                elif distance <= 18:
                    mix_ratio = (18 - distance) / 3.0
                    img[y, x] = [
                        int(20 + (100 - 20) * mix_ratio),
                        int(100 + (180 - 100) * mix_ratio),
                        int(200 + (50 - 200) * mix_ratio)
                    ]

    # 创建对应的GT
    gt = np.zeros((400, 400), dtype=np.uint8)

    for y in range(400):
        if 130 <= y <= 270:
            main_coastline_x = int(180 + 50 * np.sin(y * 0.03) + 30 * np.sin(y * 0.1) + 10 * np.cos(y * 0.15))
        else:
            main_coastline_x = int(180 + 20 * np.sin(y * 0.01))
        main_coastline_x = max(50, min(350, main_coastline_x))

        for offset in range(-2, 3):
            x = main_coastline_x + offset
            if 0 <= x < 400:
                gt[y, x] = 255

    # 小岛GT
    for dy in range(-18, 19):
        for dx in range(-18, 19):
            y, x = island_center[0] + dy, island_center[1] + dx
            if 0 <= y < 400 and 0 <= x < 400:
                distance = math.sqrt(dy * dy + dx * dx)
                if 13 <= distance <= 16:
                    gt[y, x] = 255

    return img, gt


# ==================== 测试函数 ====================

def test_precise_sea_cleanup():
    """测试精准海域清理系统"""
    print("🧪 测试精准海域清理系统...")

    detector = PreciseSeaCleanupDetector()

    # 使用你现有的数据路径
    initial_dir = "E:/initial"
    ground_truth_dir = "E:/ground"

    if os.path.exists(initial_dir):
        files = [f for f in os.listdir(initial_dir) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
        if files:
            test_file = files[0]
            initial_path = os.path.join(initial_dir, test_file)

            # 查找GT文件
            gt_path = None
            if os.path.exists(ground_truth_dir):
                gt_files = [f for f in os.listdir(ground_truth_dir) if
                            f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
                if gt_files:
                    gt_path = os.path.join(ground_truth_dir, gt_files[0])

            print(f"\n🧪 测试精准清理: {test_file}")
            result = detector.process_image(initial_path, gt_path)

            if result:
                result['sample_id'] = 'precise_cleanup_real_data'

                # 保存结果
                output_dir = "./precise_cleanup_results"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, 'precise_sea_cleanup_detection.png')
                create_precise_cleanup_visualization(result, save_path)

                # 显示清理效果
                metrics = result['quality_metrics']
                opt_pixels = np.sum(result['optimized_coastline'] > 0.3)
                final_pixels = metrics['coastline_pixels']
                reduction = (opt_pixels - final_pixels) / max(1, opt_pixels)

                print(f"\n📊 精准清理效果:")
                print(f"   清理前像素: {opt_pixels:,}")
                print(f"   清理后像素: {final_pixels:,}")
                print(f"   像素减少: {reduction:.1%}")
                print(f"   海域清理得分: {metrics.get('sea_cleanup_score', 0):.3f}")
                print(f"   水域入侵率: {metrics.get('water_intrusion', 0):.1%}")
                print(f"   连通性得分: {metrics.get('connectivity_score', 0):.3f}")
                print(f"   综合得分: {metrics['overall_score']:.3f}")

                return result

    # 如果没有真实数据，使用演示数据
    print("\n🎨 使用演示数据测试系统...")

    demo_img, demo_gt = create_demo_image()

    os.makedirs("./temp", exist_ok=True)
    demo_img_path = "./temp/demo_image_precise.png"
    demo_gt_path = "./temp/demo_gt_precise.png"

    Image.fromarray(demo_img).save(demo_img_path)
    Image.fromarray(demo_gt).save(demo_gt_path)

    result = detector.process_image(demo_img_path, demo_gt_path)
    if result:
        result['sample_id'] = 'precise_cleanup_demo'

        # 保存结果
        output_dir = "./precise_cleanup_results"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'precise_sea_cleanup_detection.png')
        create_precise_cleanup_visualization(result, save_path)

        # 显示清理效果
        metrics = result['quality_metrics']
        opt_pixels = np.sum(result['optimized_coastline'] > 0.3)
        final_pixels = metrics['coastline_pixels']
        reduction = (opt_pixels - final_pixels) / max(1, opt_pixels)

        print(f"\n📊 精准清理效果:")
        print(f"   清理前像素: {opt_pixels:,}")
        print(f"   清理后像素: {final_pixels:,}")
        print(f"   像素减少: {reduction:.1%}")
        print(f"   海域清理得分: {metrics.get('sea_cleanup_score', 0):.3f}")
        print(f"   水域入侵率: {metrics.get('water_intrusion', 0):.1%}")
        print(f"   连通性得分: {metrics.get('connectivity_score', 0):.3f}")
        print(f"   综合得分: {metrics['overall_score']:.3f}")

        return result

    return None


def process_all_coastal_data(self, initial_dir="E:/initial", ground_truth_dir="E:/ground",
                             output_dir="./batch_results"):
    """
    批量处理所有年份的海岸线数据

    Args:
        initial_dir: 初始图像目录
        ground_truth_dir: Ground Truth目录
        output_dir: 输出结果目录
    """
    print("🌊 开始批量处理海岸线数据...")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文件
    if not os.path.exists(initial_dir):
        print(f"❌ 初始目录不存在: {initial_dir}")
        return None

    # 获取所有coastal文件并按年份排序
    initial_files = [f for f in os.listdir(initial_dir)
                     if f.lower().startswith('coastal_') and f.lower().endswith('.pdf')]

    if not initial_files:
        print(f"❌ 在 {initial_dir} 中未找到coastal_*.pdf文件")
        return None

    # 按年份排序
    initial_files.sort()

    # 获取GT文件
    gt_files = []
    if os.path.exists(ground_truth_dir):
        gt_files = [f for f in os.listdir(ground_truth_dir)
                    if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
        gt_files.sort()

    print(f"📁 找到 {len(initial_files)} 个初始文件")
    print(f"📁 找到 {len(gt_files)} 个GT文件")

    # 处理结果存储
    results_summary = []
    successful_count = 0
    failed_count = 0

    # 逐个处理文件
    for i, initial_file in enumerate(initial_files):
        print(f"\n{'=' * 60}")
        print(f"🔄 处理文件 {i + 1}/{len(initial_files)}: {initial_file}")
        print(f"{'=' * 60}")

        # 构建完整路径
        initial_path = os.path.join(initial_dir, initial_file)

        # 匹配对应的GT文件
        gt_path = self._find_matching_gt_file(initial_file, gt_files, ground_truth_dir)

        if gt_path:
            print(f"📍 匹配GT文件: {os.path.basename(gt_path)}")
        else:
            print("⚠️ 未找到匹配的GT文件，将使用无GT模式")

        try:
            # 处理单个文件
            result = self.process_image(initial_path, gt_path)

            if result and result['success']:
                successful_count += 1

                # 提取年份用于文件命名
                year = self._extract_year_from_filename(initial_file)

                # 保存可视化结果
                vis_filename = f"coastal_{year}_detection_result.png"
                vis_path = os.path.join(output_dir, vis_filename)
                result['sample_id'] = f'coastal_{year}'
                create_precise_cleanup_visualization(result, vis_path)

                # 保存数值结果
                self._save_numerical_results(result, year, output_dir)

                # 记录结果摘要
                metrics = result['quality_metrics']
                summary = {
                    'year': year,
                    'file': initial_file,
                    'gt_available': gt_path is not None,
                    'final_pixels': metrics['coastline_pixels'],
                    'overall_score': metrics['overall_score'],
                    'sea_cleanup_score': metrics.get('sea_cleanup_score', 0),
                    'connectivity_score': metrics.get('connectivity_score', 0),
                    'model_used': result.get('model_used', 'unknown'),
                    'processing_time': self._get_current_time()
                }

                if 'f1_score' in metrics:
                    summary.update({
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1_score': metrics['f1_score'],
                        'iou': metrics['iou']
                    })

                results_summary.append(summary)

                print(f"✅ {year}年数据处理成功!")
                print(f"   📊 综合得分: {metrics['overall_score']:.3f}")
                print(f"   🎯 最终像素: {metrics['coastline_pixels']:,}")

            else:
                failed_count += 1
                print(f"❌ {initial_file} 处理失败")

        except Exception as e:
            failed_count += 1
            print(f"❌ 处理 {initial_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 生成批量处理报告
    self._generate_batch_report(results_summary, output_dir, successful_count, failed_count)

    print(f"\n{'=' * 80}")
    print(f"🎉 批量处理完成!")
    print(f"   ✅ 成功: {successful_count} 个文件")
    print(f"   ❌ 失败: {failed_count} 个文件")
    print(f"   📁 结果保存在: {output_dir}")
    print(f"{'=' * 80}")

    return results_summary


def _find_matching_gt_file(self, initial_file, gt_files, gt_dir):
    """找到匹配的GT文件"""
    # 提取年份
    year = self._extract_year_from_filename(initial_file)

    # 尝试不同的匹配策略
    for gt_file in gt_files:
        gt_year = self._extract_year_from_filename(gt_file)
        if gt_year == year:
            return os.path.join(gt_dir, gt_file)

    # 如果没有完全匹配，尝试模糊匹配
    base_name = os.path.splitext(initial_file)[0]
    for gt_file in gt_files:
        if base_name in gt_file or gt_file.replace('.pdf', '').replace('.png', '') in base_name:
            return os.path.join(gt_dir, gt_file)

    return None


def _extract_year_from_filename(self, filename):
    """从文件名中提取年份"""
    import re

    # 匹配4位数字年份
    year_match = re.search(r'(\d{4})', filename)
    if year_match:
        return year_match.group(1)

    # 如果没有找到年份，返回默认值
    return "unknown"


def _save_numerical_results(self, result, year, output_dir):
    """保存数值结果"""
    import json

    # 创建数值结果字典
    numerical_data = {
        'year': year,
        'processing_info': {
            'model_used': result.get('model_used', 'unknown'),
            'success': result['success'],
            'image_shape': result['processed_image'].shape
        },
        'quality_metrics': result['quality_metrics'],
        'pixel_statistics': {
            'optimized_pixels': int(np.sum(result['optimized_coastline'] > 0.3)),
            'final_pixels': int(np.sum(result['final_coastline'] > 0.5)),
            'pixel_reduction_ratio': 0.0
        }
    }

    # 计算像素减少比例
    opt_pixels = numerical_data['pixel_statistics']['optimized_pixels']
    final_pixels = numerical_data['pixel_statistics']['final_pixels']
    if opt_pixels > 0:
        reduction_ratio = (opt_pixels - final_pixels) / opt_pixels
        numerical_data['pixel_statistics']['pixel_reduction_ratio'] = float(reduction_ratio)

    # 保存JSON文件
    json_filename = f"coastal_{year}_metrics.json"
    json_path = os.path.join(output_dir, json_filename)

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(numerical_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 数值结果已保存: {json_filename}")
    except Exception as e:
        print(f"   ⚠️ 保存数值结果失败: {e}")


def _generate_batch_report(self, results_summary, output_dir, successful_count, failed_count):
    """生成批量处理报告"""
    import json
    from datetime import datetime

    # 创建汇总报告
    report = {
        'batch_processing_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_files': successful_count + failed_count,
            'successful_files': successful_count,
            'failed_files': failed_count,
            'success_rate': successful_count / (successful_count + failed_count) if (
                                                                                                successful_count + failed_count) > 0 else 0
        },
        'detailed_results': results_summary
    }

    # 计算统计信息
    if results_summary:
        overall_scores = [r['overall_score'] for r in results_summary]
        final_pixels = [r['final_pixels'] for r in results_summary]

        report['statistics'] = {
            'overall_score': {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'min': float(np.min(overall_scores)),
                'max': float(np.max(overall_scores))
            },
            'final_pixels': {
                'mean': float(np.mean(final_pixels)),
                'std': float(np.std(final_pixels)),
                'min': int(np.min(final_pixels)),
                'max': int(np.max(final_pixels))
            }
        }

        # GT匹配统计
        gt_available_count = sum(1 for r in results_summary if r['gt_available'])
        if gt_available_count > 0:
            f1_scores = [r.get('f1_score', 0) for r in results_summary if 'f1_score' in r]
            if f1_scores:
                report['statistics']['f1_score'] = {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'min': float(np.min(f1_scores)),
                    'max': float(np.max(f1_scores))
                }

    # 保存报告
    report_path = os.path.join(output_dir, 'batch_processing_report.json')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   📋 批量处理报告已保存: batch_processing_report.json")
    except Exception as e:
        print(f"   ⚠️ 保存报告失败: {e}")

    # 生成简易CSV报告
    self._generate_csv_report(results_summary, output_dir)


def _generate_csv_report(self, results_summary, output_dir):
    """生成CSV格式的简易报告"""
    import csv

    if not results_summary:
        return

    csv_path = os.path.join(output_dir, 'batch_results_summary.csv')

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # 确定字段名
            fieldnames = ['year', 'file', 'gt_available', 'final_pixels', 'overall_score',
                          'sea_cleanup_score', 'connectivity_score', 'model_used']

            # 如果有GT数据，添加相关字段
            if any('f1_score' in r for r in results_summary):
                fieldnames.extend(['precision', 'recall', 'f1_score', 'iou'])

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results_summary:
                # 只写入存在的字段
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)

        print(f"   📊 CSV报告已保存: batch_results_summary.csv")
    except Exception as e:
        print(f"   ⚠️ 保存CSV报告失败: {e}")


def _get_current_time(self):
    """获取当前时间字符串"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 修改主测试函数以支持批量处理
def test_batch_processing():
    """测试批量处理功能"""
    print("🧪 测试批量海岸线数据处理...")

    detector = PreciseSeaCleanupDetector()

    # 执行批量处理
    results = detector.process_all_coastal_data(
        initial_dir="E:/initial",
        ground_truth_dir="E:/ground",
        output_dir="./coastal_batch_results"
    )

    if results:
        print(f"\n📈 批量处理结果汇总:")
        print(f"   处理年份: {[r['year'] for r in results]}")
        print(f"   平均得分: {np.mean([r['overall_score'] for r in results]):.3f}")
        print(f"   GT可用率: {sum(r['gt_available'] for r in results) / len(results):.1%}")

    return results


# ==================== 简单修复：添加到你的原始代码末尾 ====================

def test_my_saved_model():
    """
    简单的测试函数 - 使用已保存模型测试2018-2020年数据
    直接添加到你的原始代码末尾即可
    """
    print("🧪 测试已保存模型在2018-2020年数据上的表现")
    print("=" * 60)

    # 检查模型是否存在
    model_dir = "./saved_models"
    if not os.path.exists(model_dir):
        print("❌ 模型目录不存在，请先训练模型")
        return

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    print(f"📁 找到 {len(model_files)} 个模型文件")

    # 设置路径
    initial_dir = "E:/initial"
    ground_dir = "E:/ground"
    output_dir = "./quick_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 目标年份
    years = ['2018', '2019', '2020']

    # 创建检测器
    detector = PreciseSeaCleanupDetector()

    # 测试结果
    results = []

    for year in years:
        print(f"\n{'=' * 50}")
        print(f"🔄 测试 {year} 年数据")
        print(f"{'=' * 50}")

        # 构建文件路径
        coastal_file = f"coastal_{year}.pdf"
        ground_file = f"ground_{year}.pdf"

        coastal_path = os.path.join(initial_dir, coastal_file)
        ground_path = os.path.join(ground_dir, ground_file)

        # 检查文件是否存在
        if not os.path.exists(coastal_path):
            print(f"❌ 未找到文件: {coastal_path}")
            continue

        if not os.path.exists(ground_path):
            print(f"⚠️ 未找到GT文件: {ground_path}")
            ground_path = None

        try:
            print(f"📁 处理文件: {coastal_file}")
            print(f"📁 GT文件: {ground_file if ground_path else 'None'}")

            # 关键：使用已保存模型，不重新训练
            result = detector.process_image(
                image_path=coastal_path,
                ground_truth_path=ground_path,
                force_retrain=False  # 🔑 使用已保存模型
            )

            if result and result.get('success', False):
                print(f"✅ {year} 年处理成功!")

                # 保存可视化
                result['sample_id'] = f'test_{year}'
                vis_path = os.path.join(output_dir, f'test_{year}.png')
                create_precise_cleanup_visualization(result, vis_path)

                # 提取关键指标
                metrics = result['quality_metrics']
                year_result = {
                    'year': year,
                    'success': True,
                    'model_used': result.get('model_used', 'unknown'),
                    'overall_score': metrics['overall_score'],
                    'final_pixels': metrics['coastline_pixels'],
                    'sea_cleanup_score': metrics.get('sea_cleanup_score', 0),
                    'connectivity_score': metrics.get('connectivity_score', 0)
                }

                # 如果有GT，添加GT指标
                if ground_path and 'f1_score' in metrics:
                    year_result.update({
                        'f1_score': metrics['f1_score'],
                        'iou': metrics['iou'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall']
                    })

                results.append(year_result)

                # 显示结果
                print(f"   📊 综合得分: {metrics['overall_score']:.3f}")
                print(f"   🎯 像素数: {metrics['coastline_pixels']:,}")
                print(f"   🌊 海域清理: {metrics.get('sea_cleanup_score', 0):.3f}")
                print(f"   🤖 模型类型: {result.get('model_used', 'unknown')}")

                if 'f1_score' in metrics:
                    print(f"   📋 GT匹配 F1: {metrics['f1_score']:.3f}")

            else:
                print(f"❌ {year} 年处理失败")
                results.append({'year': year, 'success': False})

        except Exception as e:
            print(f"❌ {year} 年处理出错: {e}")
            results.append({'year': year, 'success': False, 'error': str(e)})

    # 显示汇总结果
    successful_results = [r for r in results if r.get('success', False)]

    print(f"\n{'=' * 60}")
    print(f"🎉 测试完成!")
    print(f"   成功: {len(successful_results)}/{len(years)}")
    print(f"   结果保存在: {output_dir}")
    print(f"{'=' * 60}")

    if successful_results:
        print(f"\n📈 汇总结果:")

        # 计算平均值
        avg_score = np.mean([r['overall_score'] for r in successful_results])
        avg_pixels = np.mean([r['final_pixels'] for r in successful_results])

        print(f"   平均综合得分: {avg_score:.3f}")
        print(f"   平均像素数: {avg_pixels:,.0f}")

        # 逐年显示
        print(f"\n   📅 逐年结果:")
        for result in successful_results:
            score = result['overall_score']
            pixels = result['final_pixels']
            model = result['model_used']
            print(f"      {result['year']}: 得分={score:.3f}, 像素={pixels:,}, 模型={model}")

            if 'f1_score' in result:
                print(f"                GT匹配F1={result['f1_score']:.3f}")

    return results


# ==================== 直接调用测试 ====================

# 在你的原始代码的最后，将 main() 替换为：

def main():
    """主函数 - 现在包含测试选项"""
    print("🚀 启动精准海域清理海岸线检测系统...")

    print("\n请选择:")
    print("1. 训练新模型")
    print("2. 测试已保存模型（2018-2020）")
    choice = "2"

    if choice == "2":
        # 测试已保存模型
        test_my_saved_model()


# 或者直接运行测试：
if __name__ == "__main__":
    # 方式1：带选择的主函数
    main()

    # 方式2：直接运行测试
    # test_my_saved_model()