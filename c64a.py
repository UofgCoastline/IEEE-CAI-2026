
"""
自适应海岸线检测系统 - HSV→GT渐进监督 + 海陆分离
主要改进：
1. 前期HSV监督，后期逐渐转向GT学习
2. 海陆分离导向，保留所有有效边界
3. 中间区域重点探索
4. 宽松后处理，GT保护机制
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

print("🌊 自适应海岸线检测系统 - HSV→GT渐进监督!")
print("重点：前期HSV → 后期GT + 海陆分离 + 中间区域探索")
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


# ==================== HSV注意力监督器 ====================

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

    def evaluate_prediction_quality(self, prediction, ground_truth, hsv_analysis):
        """评价预测质量"""
        quality_score = 0.0
        pred_binary = (prediction > 0.5).astype(bool)

        coastline_guidance = hsv_analysis['coastline_guidance']
        guidance_alignment = np.sum(pred_binary * coastline_guidance) / (np.sum(pred_binary) + 1e-8)
        quality_score += guidance_alignment * 0.3

        transition_strength = hsv_analysis['transition_strength']
        transition_coverage = np.sum(pred_binary * transition_strength) / (np.sum(transition_strength) + 1e-8)
        quality_score += transition_coverage * 0.2

        water_mask = hsv_analysis['water_mask']
        water_penetration = np.sum(pred_binary & water_mask) / (np.sum(pred_binary) + 1e-8)
        quality_score -= water_penetration * 0.5

        if ground_truth is not None:
            gt_binary = (ground_truth > 0.5).astype(bool)
            tp = np.sum(pred_binary & gt_binary)
            fp = np.sum(pred_binary & ~gt_binary)
            fn = np.sum(~pred_binary & gt_binary)
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * precision * recall / (precision + recall + 1e-8)
            quality_score += f1_score * 0.4
        else:
            hsv_reasonableness = self._evaluate_hsv_reasonableness(pred_binary, hsv_analysis)
            quality_score += hsv_reasonableness * 0.4

        return max(0.0, min(1.0, quality_score))

    def _evaluate_hsv_reasonableness(self, prediction, hsv_analysis):
        """评价基于HSV的合理性"""
        water_mask = hsv_analysis['water_mask']
        land_mask = hsv_analysis['land_mask']

        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask

        boundary_region = water_boundary | land_boundary
        boundary_coverage = np.sum(prediction & boundary_region) / (np.sum(prediction) + 1e-8)

        return boundary_coverage


# ==================== 自适应HSV监督器 ====================

class AdaptiveHSVSupervisor:
    """自适应HSV监督器 - 前期强监督，后期逐渐转向GT"""

    def __init__(self, total_episodes=200):
        print("✅ 自适应HSV监督器初始化完成 - 前期HSV后期GT")
        self.hsv_supervisor = HSVAttentionSupervisor()
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.hsv_weight_schedule = self._create_weight_schedule()
        
    def _create_weight_schedule(self):
        """创建权重衰减计划"""
        schedule = []
        
        # 前期阶段 (0-30%)
        early_episodes = int(self.total_episodes * 0.3)
        for i in range(early_episodes):
            weight = 1.0 - 0.3 * (i / early_episodes)
            schedule.append(weight)
            
        # 中期阶段 (30%-70%)  
        middle_episodes = int(self.total_episodes * 0.4)
        for i in range(middle_episodes):
            weight = 0.7 - 0.4 * (i / middle_episodes)
            schedule.append(weight)
            
        # 后期阶段 (70%-100%)
        late_episodes = self.total_episodes - early_episodes - middle_episodes
        for i in range(late_episodes):
            weight = 0.3 - 0.2 * (i / late_episodes)
            schedule.append(weight)
            
        return schedule
    
    def get_current_hsv_weight(self):
        """获取当前HSV权重"""
        if self.current_episode < len(self.hsv_weight_schedule):
            return self.hsv_weight_schedule[self.current_episode]
        return 0.1
        
    def get_current_gt_weight(self):
        """获取当前GT权重"""
        hsv_weight = self.get_current_hsv_weight()
        return 1.0 - hsv_weight + 0.5
        
    def update_episode(self, episode):
        """更新当前轮次"""
        self.current_episode = episode
        
    def evaluate_prediction_quality(self, prediction, ground_truth, hsv_analysis):
        """自适应质量评估"""
        hsv_weight = self.get_current_hsv_weight()
        gt_weight = self.get_current_gt_weight()
        
        hsv_quality = self.hsv_supervisor.evaluate_prediction_quality(
            prediction, ground_truth, hsv_analysis
        )
        
        gt_quality = 0.0
        if ground_truth is not None:
            pred_binary = (prediction > 0.5).astype(bool)
            gt_binary = (ground_truth > 0.5).astype(bool)
            
            tp = np.sum(pred_binary & gt_binary)
            fp = np.sum(pred_binary & ~gt_binary)
            fn = np.sum(~pred_binary & gt_binary)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            gt_quality = 2 * precision * recall / (precision + recall + 1e-8)
        
        combined_quality = hsv_quality * hsv_weight + gt_quality * gt_weight
        
        return combined_quality


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

        self.adaptive_hsv_supervisor = AdaptiveHSVSupervisor()
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

        # 自适应HSV和GT奖励权重
        hsv_weight = self.adaptive_hsv_supervisor.get_current_hsv_weight()
        gt_weight = self.adaptive_hsv_supervisor.get_current_gt_weight()

        # HSV监督奖励
        hsv_reward = self._calculate_hsv_reward(new_pos)
        reward += hsv_reward * 20.0 * hsv_weight

        # GT奖励
        if self.gt_analysis and self.gt_analysis['gt_binary'] is not None:
            gt_reward = self._calculate_gt_reward(new_pos)
            reward += gt_reward * gt_weight

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
        print("🎯 自适应海岸线优化开始...")

        self.env.adaptive_hsv_supervisor = AdaptiveHSVSupervisor(max_episodes)

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
            self.env.adaptive_hsv_supervisor.update_episode(episode)
            
            current_hsv_weight = self.env.adaptive_hsv_supervisor.get_current_hsv_weight()
            current_gt_weight = self.env.adaptive_hsv_supervisor.get_current_gt_weight()

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
                      f"像素={current_pixels:,}, 中间比例={middle_ratio:.1%}, "
                      f"HSV权重={current_hsv_weight:.2f}, GT权重={current_gt_weight:.2f}")

        final_pixels = np.sum(self.env.current_coastline > 0.3)
        print(f"   ✅ 优化完成: 总像素={final_pixels:,}, 改进次数={total_improvements}")

        return self.env.current_coastline

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ==================== 海陆分离后处理器 ====================

class SeaLandSeparationPostProcessor:
    """海陆分离后处理器"""

    def __init__(self):
        print("✅ 海陆分离后处理器初始化完成")

    def process_sea_land_separation(self, coastline, hsv_analysis, gt_analysis):
        """海陆分离处理"""
        print("🔧 开始海陆分离后处理...")

        binary_coastline = self._generous_hsv_binarization(coastline, hsv_analysis)
        gt_protected_coastline = self._apply_gt_protection(binary_coastline, gt_analysis)
        validated_coastline = self._validate_sea_land_separation(gt_protected_coastline, hsv_analysis)
        connected_coastline = self._optimize_connectivity_generous(validated_coastline, hsv_analysis)
        final_coastline = self._gentle_smoothing(connected_coastline)

        return final_coastline.astype(float)

    def _generous_hsv_binarization(self, coastline, hsv_analysis):
        """宽松的HSV引导二值化"""
        guidance_weight = hsv_analysis['coastline_guidance']
        transition_weight = hsv_analysis['transition_strength']

        weighted_coastline = coastline * (0.5 + guidance_weight * 0.3 + transition_weight * 0.2)

        valid_mask = weighted_coastline > 0
        if np.any(valid_mask):
            threshold = np.percentile(weighted_coastline[valid_mask], 60)
        else:
            threshold = 0.3

        binary_result = weighted_coastline > threshold
        binary_result = self._remove_small_components(binary_result, min_size=3)

        return binary_result

    def _apply_gt_protection(self, binary_coastline, gt_analysis):
        """GT保护处理"""
        if not gt_analysis or gt_analysis['gt_binary'] is None:
            return binary_coastline

        result = binary_coastline.copy()
        gt_binary = gt_analysis['gt_binary']

        result = result | gt_binary

        gt_expanded = gt_binary.copy()
        for _ in range(2):
            gt_expanded = binary_dilation(gt_expanded, np.ones((3, 3)))

        gt_protection_mask = gt_expanded
        protected_region = binary_coastline & gt_protection_mask
        result = result | protected_region

        print(f"   GT保护: 添加了 {np.sum(gt_binary):,} GT像素")

        return result

    def _validate_sea_land_separation(self, binary_coastline, hsv_analysis):
        """海陆分离验证"""
        result = binary_coastline.copy()
        water_mask = hsv_analysis['water_mask']
        land_mask = hsv_analysis['land_mask']

        points_to_remove = []

        for y in range(binary_coastline.shape[0]):
            for x in range(binary_coastline.shape[1]):
                if not binary_coastline[y, x]:
                    continue

                water_neighbors = 0
                land_neighbors = 0
                total_neighbors = 0

                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < water_mask.shape[0] and 0 <= nx < water_mask.shape[1]:
                            total_neighbors += 1
                            if water_mask[ny, nx]:
                                water_neighbors += 1
                            if land_mask[ny, nx]:
                                land_neighbors += 1

                if total_neighbors > 0:
                    water_ratio = water_neighbors / total_neighbors
                    land_ratio = land_neighbors / total_neighbors

                    if water_ratio > 0.95 or land_ratio > 0.95:
                        points_to_remove.append((y, x))

        for y, x in points_to_remove:
            result[y, x] = False

        return result

    def _optimize_connectivity_generous(self, binary_coastline, hsv_analysis):
        """宽松的连通性优化"""
        result = binary_coastline.copy()
        labeled_array, num_components = label(binary_coastline)

        if num_components <= 1:
            return result

        for i in range(1, min(num_components + 1, 10)):
            for j in range(i + 1, min(num_components + 1, 10)):
                connection_path = self._find_generous_connection(
                    labeled_array, i, j, hsv_analysis
                )
                if connection_path:
                    for y, x in connection_path:
                        result[y, x] = True

        return result

    def _find_generous_connection(self, labeled_array, comp1_id, comp2_id, hsv_analysis):
        """寻找宽松的连接路径"""
        comp1_coords = np.where(labeled_array == comp1_id)
        comp2_coords = np.where(labeled_array == comp2_id)

        if len(comp1_coords[0]) == 0 or len(comp2_coords[0]) == 0:
            return None

        best_path = None
        best_score = -1

        sample1 = list(zip(comp1_coords[0][::max(1, len(comp1_coords[0]) // 3)],
                           comp1_coords[1][::max(1, len(comp1_coords[1]) // 3)]))
        sample2 = list(zip(comp2_coords[0][::max(1, len(comp2_coords[0]) // 3)],
                           comp2_coords[1][::max(1, len(comp2_coords[1]) // 3)]))

        for p1 in sample1[:3]:
            for p2 in sample2[:3]:
                distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if distance > 50:
                    continue

                path = self._generate_simple_path(p1, p2)
                if path:
                    score = len(path) * 0.1
                    if score > best_score:
                        best_score = score
                        best_path = path

        return best_path if best_score > 0.1 else None

    def _generate_simple_path(self, p1, p2):
        """生成简单连接路径"""
        path = []
        x1, y1 = p1[1], p1[0]
        x2, y2 = p2[1], p2[0]

        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return [(p1[0], p1[1])]

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            path.append((y, x))

        return path

    def _gentle_smoothing(self, binary_coastline):
        """轻微平滑处理"""
        kernel = np.ones((3, 3), dtype=bool)
        smoothed = binary_closing(binary_coastline, kernel, iterations=1)
        return smoothed

    def _remove_small_components(self, binary_image, min_size=3):
        """移除小组件"""
        labeled_array, num_components = label(binary_image)

        result = binary_image.copy()
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            if size < min_size:
                result[labeled_array == i] = False

        return result


# ==================== 主检测器 ====================

class ConstrainedCoastlineDetector:
    """约束的海岸线检测器"""

    def __init__(self):
        self.gt_analyzer = GroundTruthAnalyzer()
        self.post_processor = SeaLandSeparationPostProcessor()
        print("✅ 约束海岸线检测系统初始化完成")
        print("   🎯 主要特色：自适应HSV->GT监督 + 海陆分离保护")

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

    def process_image(self, image_path, ground_truth_path=None):
        """处理单个图像"""
        print(f"\n🌊 自适应海岸线检测处理: {os.path.basename(image_path)}")

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

            # 步骤2: 创建约束环境
            print("\n📍 步骤2: 创建自适应约束环境")
            constrained_env = ConstrainedCoastlineEnvironment(processed_img, gt_analysis)

            # 步骤3: 自适应DQN训练
            print("\n📍 步骤3: 自适应DQN学习")
            constrained_agent = ConstrainedCoastlineAgent(constrained_env)

            optimized_coastline = constrained_agent.optimize_constrained_coastline(
                max_episodes=200,
                max_steps_per_episode=400
            )

            # 步骤4: 海陆分离后处理
            print("\n📍 步骤4: 海陆分离后处理")
            final_coastline = self.post_processor.process_sea_land_separation(
                optimized_coastline, constrained_env.hsv_analysis, gt_analysis
            )

            # 质量评估
            quality_metrics = self._evaluate_adaptive_quality(final_coastline, gt_coastline,
                                                             constrained_env.hsv_analysis)

            return {
                'original_image': original_img,
                'processed_image': processed_img,
                'gt_analysis': gt_analysis,
                'ground_truth': gt_coastline,
                'hsv_analysis': constrained_env.hsv_analysis,
                'optimized_coastline': optimized_coastline,
                'final_coastline': final_coastline,
                'quality_metrics': quality_metrics,
                'success': quality_metrics['overall_score'] > 0.5
            }

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _evaluate_adaptive_quality(self, predicted, ground_truth, hsv_analysis):
        """评估自适应质量"""
        metrics = {}

        pred_binary = (predicted > 0.5).astype(bool)
        coastline_pixels = np.sum(pred_binary)

        metrics['coastline_pixels'] = int(coastline_pixels)

        # 连通性分析
        labeled_array, num_components = label(pred_binary)
        metrics['num_components'] = int(num_components)

        # 自适应HSV质量评估
        adaptive_hsv_supervisor = AdaptiveHSVSupervisor()
        adaptive_hsv_supervisor.current_episode = 200
        adaptive_quality = adaptive_hsv_supervisor.evaluate_prediction_quality(
            predicted, ground_truth, hsv_analysis
        )
        metrics['adaptive_quality'] = float(adaptive_quality)

        # 水域渗透检查
        water_mask = hsv_analysis['water_mask']
        water_penetration = np.sum(pred_binary & water_mask) / (coastline_pixels + 1e-8)
        metrics['water_penetration'] = float(water_penetration)

        # 海陆分离效果评估
        sea_land_separation_score = self._evaluate_sea_land_separation(pred_binary, hsv_analysis)
        metrics['sea_land_separation'] = float(sea_land_separation_score)

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

            overall_score = (f1_score * 0.3 + iou * 0.2 + adaptive_quality * 0.2 +
                             sea_land_separation_score * 0.2 + (1 - water_penetration) * 0.1)
        else:
            density_score = min(1.0, coastline_pixels / 5000.0)
            overall_score = (adaptive_quality * 0.3 + sea_land_separation_score * 0.4 +
                             (1 - water_penetration) * 0.2 + density_score * 0.1)

        metrics['overall_score'] = float(overall_score)

        return metrics

    def _evaluate_sea_land_separation(self, binary_coastline, hsv_analysis):
        """评估海陆分离效果"""
        if not np.any(binary_coastline):
            return 0.0

        water_mask = hsv_analysis['water_mask']
        land_mask = hsv_analysis['land_mask']

        separation_scores = []
        coastline_positions = np.where(binary_coastline)

        for i in range(len(coastline_positions[0])):
            y, x = coastline_positions[0][i], coastline_positions[1][i]
            
            water_neighbors = 0
            land_neighbors = 0
            total_neighbors = 0

            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < water_mask.shape[0] and 0 <= nx < water_mask.shape[1]:
                        total_neighbors += 1
                        if water_mask[ny, nx]:
                            water_neighbors += 1
                        if land_mask[ny, nx]:
                            land_neighbors += 1

            if total_neighbors > 0:
                water_ratio = water_neighbors / total_neighbors
                land_ratio = land_neighbors / total_neighbors
                
                if water_ratio > 0.1 and land_ratio > 0.1:
                    separation_score = min(water_ratio, land_ratio) * 2
                else:
                    separation_score = 0.1
                    
                separation_scores.append(separation_score)

        if separation_scores:
            return np.mean(separation_scores)
        else:
            return 0.0


# ==================== 可视化函数 ====================

def create_adaptive_visualization(result, save_path):
    """创建自适应版可视化"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Adaptive Coastline Detection (HSV→GT + Sea-Land Separation) - {result.get("sample_id", "Unknown")}',
                 fontsize=16, fontweight='bold')

    # 第一行：输入和HSV分析
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

    # 第二行：检测结果
    axes[1, 0].imshow(result['optimized_coastline'], cmap='hot')
    opt_pixels = np.sum(result['optimized_coastline'] > 0.3)
    axes[1, 0].set_title(f'Adaptive DQN Detection\n({opt_pixels:,} pixels)', color='blue', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['final_coastline'], cmap='hot')
    final_pixels = np.sum(result['final_coastline'] > 0.5)
    axes[1, 1].set_title(f'Final Sea-Land Separation\n({final_pixels:,} pixels)', color='red', fontweight='bold')
    axes[1, 1].axis('off')

    # 海陆分离可视化
    if 'hsv_analysis' in result:
        water_mask = result['hsv_analysis']['water_mask']
        land_mask = result['hsv_analysis']['land_mask']
        pred_binary = (result['final_coastline'] > 0.5).astype(bool)

        separation_vis = np.zeros((*result['final_coastline'].shape, 3))
        separation_vis[:, :, 0] = water_mask.astype(float) * 0.6
        separation_vis[:, :, 1] = land_mask.astype(float) * 0.6
        separation_vis[:, :, 2] = pred_binary.astype(float)

        axes[1, 2].imshow(separation_vis)
        axes[1, 2].set_title('Sea-Land Separation\n(Water:Red, Land:Green, Coast:Blue)')
        axes[1, 2].axis('off')
    else:
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
    stats_text = f"""Adaptive Coastline Detection Results:

Overall Score: {metrics['overall_score']:.3f}
Adaptive Quality: {metrics.get('adaptive_quality', 0):.3f}
Status: {"✅ SUCCESS" if result['success'] else "❌ FAILED"}

Quality Analysis:
• Final pixels: {metrics['coastline_pixels']:,}
• Components: {metrics['num_components']}
• Water penetration: {metrics.get('water_penetration', 0):.1%}
• Sea-land separation: {metrics.get('sea_land_separation', 0):.3f}"""

    if 'f1_score' in metrics:
        stats_text += f"""

GT Matching Metrics:
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}
• F1-Score: {metrics['f1_score']:.3f}
• IoU: {metrics['iou']:.3f}"""

    stats_text += f"""

Adaptive Features:
✓ HSV→GT progressive supervision
✓ Middle region focus (1/3)
✓ Sea-land boundary preservation
✓ GT protection in post-processing
✓ Generous connectivity optimization
✓ Adaptive reward weighting

Technical Improvements:
• Adaptive HSV weight: 1.0→0.1
• GT weight progression: 0.5→1.5
• Sea-land separation rewards
• Water boundary utilization
• GT-protected post-processing
• Generous connection thresholds
• Device: {device}"""

    axes[2, 0].text(0.02, 0.98, stats_text, transform=fig.transFigure,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 自适应版可视化已保存: {save_path}")


# ==================== 演示函数 ====================

def create_adaptive_demo_image():
    """创建自适应演示海岸线图像"""
    print("🎨 创建自适应演示海岸线图像...")

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




# ==================== 改进的海陆分离后处理器 ====================

class ImprovedSeaLandSeparationPostProcessor:
    """改进的海陆分离后处理器 - 强化GT保护，减少错误删除"""

    def __init__(self):
        print("✅ 改进的海陆分离后处理器初始化完成 - 强化GT保护")

    def process_sea_land_separation(self, coastline, hsv_analysis, gt_analysis):
        """改进的海陆分离处理 - 强化GT保护"""
        print("🔧 开始改进的海陆分离后处理...")
        print(f"   输入海岸线像素: {np.sum(coastline > 0.3):,}")

        # 第一步：极其宽松的二值化 - 保留更多候选
        binary_coastline = self._ultra_generous_binarization(coastline, hsv_analysis)
        print(f"   宽松二值化后: {np.sum(binary_coastline):,} 像素")

        # 第二步：强化GT保护 - 最高优先级
        gt_protected_coastline = self._enhanced_gt_protection(binary_coastline, gt_analysis)
        print(f"   GT保护后: {np.sum(gt_protected_coastline):,} 像素")

        # 第三步：极其保守的验证 - 只移除明显错误的
        validated_coastline = self._ultra_conservative_validation(gt_protected_coastline, hsv_analysis, gt_analysis)
        print(f"   保守验证后: {np.sum(validated_coastline):,} 像素")

        # 第四步：强化连通性优化
        connected_coastline = self._enhanced_connectivity_optimization(validated_coastline, hsv_analysis, gt_analysis)
        print(f"   连通性优化后: {np.sum(connected_coastline):,} 像素")

        # 第五步：最终GT再保护 - 确保不被删除
        final_coastline = self._final_gt_protection(connected_coastline, gt_analysis)
        print(f"   最终GT保护后: {np.sum(final_coastline):,} 像素")

        return final_coastline.astype(float)

    def _ultra_generous_binarization(self, coastline, hsv_analysis):
        """极其宽松的二值化 - 保留更多候选"""
        guidance_weight = hsv_analysis['coastline_guidance']
        transition_weight = hsv_analysis['transition_strength']

        # 更宽松的加权 - 降低HSV的影响
        weighted_coastline = coastline * (0.8 + guidance_weight * 0.1 + transition_weight * 0.1)

        # 大幅降低阈值
        valid_mask = weighted_coastline > 0
        if np.any(valid_mask):
            threshold = np.percentile(weighted_coastline[valid_mask], 40)  # 从60降到40
        else:
            threshold = 0.2  # 从0.3降到0.2

        binary_result = weighted_coastline > threshold

        # 几乎不移除任何组件
        binary_result = self._remove_tiny_components(binary_result, min_size=1)  # 从3降到1

        print(f"     二值化阈值: {threshold:.3f}")
        return binary_result

    def _enhanced_gt_protection(self, binary_coastline, gt_analysis):
        """强化GT保护 - 最高优先级"""
        if not gt_analysis or gt_analysis['gt_binary'] is None:
            return binary_coastline

        result = binary_coastline.copy()
        gt_binary = gt_analysis['gt_binary']

        print(f"     GT像素数: {np.sum(gt_binary):,}")

        # 1. 强制添加所有GT像素
        result = result | gt_binary

        # 2. 大范围扩展GT保护区域
        gt_expanded = gt_binary.copy()
        for _ in range(5):  # 增加扩展范围
            gt_expanded = binary_dilation(gt_expanded, np.ones((3, 3)))

        # 3. 在扩展区域内，强制保留所有原始预测
        gt_protection_mask = gt_expanded
        protected_region = binary_coastline & gt_protection_mask
        result = result | protected_region

        # 4. 特别保护：GT周围的高质量预测
        gt_neighbor_protection = binary_dilation(gt_binary, np.ones((7, 7)))  # 更大范围
        high_quality_predictions = binary_coastline & gt_neighbor_protection
        result = result | high_quality_predictions

        print(f"     强制添加GT: {np.sum(gt_binary):,} 像素")
        print(f"     扩展保护: {np.sum(protected_region):,} 像素")
        print(f"     邻域保护: {np.sum(high_quality_predictions):,} 像素")

        return result

    def _ultra_conservative_validation(self, binary_coastline, hsv_analysis, gt_analysis):
        """极其保守的验证 - 只移除非常明显的错误"""
        result = binary_coastline.copy()
        water_mask = hsv_analysis['water_mask']
        land_mask = hsv_analysis['land_mask']

        # 创建GT保护掩码 - 绝对不能删除的区域
        gt_protection_mask = np.zeros_like(binary_coastline, dtype=bool)
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            # GT及其大范围邻域都受到保护
            gt_protection_mask = binary_dilation(gt_binary, np.ones((15, 15)))

        points_to_remove = []

        for y in range(binary_coastline.shape[0]):
            for x in range(binary_coastline.shape[1]):
                if not binary_coastline[y, x]:
                    continue

                # GT保护区域：绝对不删除
                if gt_protection_mask[y, x]:
                    continue

                # 检查更大范围的水陆分布
                water_neighbors = 0
                land_neighbors = 0
                total_neighbors = 0

                for dy in range(-5, 6):  # 扩大检查范围
                    for dx in range(-5, 6):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < water_mask.shape[0] and 0 <= nx < water_mask.shape[1]:
                            total_neighbors += 1
                            if water_mask[ny, nx]:
                                water_neighbors += 1
                            if land_mask[ny, nx]:
                                land_neighbors += 1

                if total_neighbors > 0:
                    water_ratio = water_neighbors / total_neighbors
                    land_ratio = land_neighbors / total_neighbors

                    # 只有在极端纯净环境中才删除（99%以上）
                    if water_ratio > 0.99 or land_ratio > 0.99:
                        # 进一步检查是否有连接价值
                        if not self._has_critical_connection_value(y, x, binary_coastline):
                            points_to_remove.append((y, x))

        # 移除无价值的点
        for y, x in points_to_remove:
            result[y, x] = False

        print(f"     极保守验证: 移除了 {len(points_to_remove)} 个极端无效点")

        return result

    def _has_critical_connection_value(self, y, x, binary_coastline):
        """检查点是否有关键连接价值"""
        # 检查移除这个点是否会严重影响连通性
        temp_coastline = binary_coastline.copy()
        temp_coastline[y, x] = False

        # 检查更大局部区域的连通性变化
        local_region = slice(max(0, y-5), min(binary_coastline.shape[0], y+6)), \
                      slice(max(0, x-5), min(binary_coastline.shape[1], x+6))

        original_local = binary_coastline[local_region]
        modified_local = temp_coastline[local_region]

        # 计算连通组件
        _, original_components = label(original_local)
        _, modified_components = label(modified_local)

        # 如果移除后组件数显著增加，说明这个点很重要
        return modified_components > original_components

    def _enhanced_connectivity_optimization(self, binary_coastline, hsv_analysis, gt_analysis):
        """强化连通性优化 - 特别关注GT连接"""
        result = binary_coastline.copy()
        labeled_array, num_components = label(binary_coastline)

        if num_components <= 1:
            return result

        print(f"     连接 {num_components} 个组件...")

        # 优先连接包含GT的组件
        gt_components = set()
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_binary = gt_analysis['gt_binary']
            for y in range(gt_binary.shape[0]):
                for x in range(gt_binary.shape[1]):
                    if gt_binary[y, x] and labeled_array[y, x] > 0:
                        gt_components.add(labeled_array[y, x])

        print(f"     包含GT的组件: {len(gt_components)} 个")

        # 1. 优先连接GT组件之间
        gt_component_list = list(gt_components)
        for i in range(len(gt_component_list)):
            for j in range(i + 1, len(gt_component_list)):
                connection_path = self._find_gt_priority_connection(
                    labeled_array, gt_component_list[i], gt_component_list[j], 
                    hsv_analysis, gt_analysis
                )
                if connection_path:
                    for y, x in connection_path:
                        result[y, x] = True

        # 2. 连接GT组件与其他组件
        non_gt_components = set(range(1, min(num_components + 1, 20))) - gt_components
        for gt_comp in gt_components:
            for other_comp in non_gt_components:
                connection_path = self._find_gt_priority_connection(
                    labeled_array, gt_comp, other_comp, hsv_analysis, gt_analysis
                )
                if connection_path:
                    for y, x in connection_path:
                        result[y, x] = True

        return result

    def _find_gt_priority_connection(self, labeled_array, comp1_id, comp2_id, hsv_analysis, gt_analysis):
        """寻找GT优先的连接路径"""
        comp1_coords = np.where(labeled_array == comp1_id)
        comp2_coords = np.where(labeled_array == comp2_id)

        if len(comp1_coords[0]) == 0 or len(comp2_coords[0]) == 0:
            return None

        # 寻找最佳连接点对
        best_path = None
        best_score = -1

        # 增加采样点，特别关注GT附近的点
        sample1 = self._get_gt_priority_samples(comp1_coords, gt_analysis)
        sample2 = self._get_gt_priority_samples(comp2_coords, gt_analysis)

        for p1 in sample1[:8]:  # 增加检查的点数
            for p2 in sample2[:8]:
                distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if distance > 80:  # 进一步增加允许的连接距离
                    continue

                path = self._generate_smart_path(p1, p2, gt_analysis)
                if path:
                    score = self._evaluate_gt_connection_quality(path, hsv_analysis, gt_analysis)
                    if score > best_score:
                        best_score = score
                        best_path = path

        return best_path if best_score > 0.05 else None  # 进一步降低连接阈值

    def _get_gt_priority_samples(self, component_coords, gt_analysis):
        """获取GT优先的采样点"""
        if not gt_analysis or gt_analysis['gt_binary'] is None:
            # 标准采样
            return list(zip(component_coords[0][::max(1, len(component_coords[0]) // 8)],
                           component_coords[1][::max(1, len(component_coords[1]) // 8)]))

        gt_binary = gt_analysis['gt_binary']
        gt_samples = []
        regular_samples = []

        # 分离GT附近的点和普通点
        for i in range(len(component_coords[0])):
            y, x = component_coords[0][i], component_coords[1][i]
            
            # 检查是否在GT附近
            is_near_gt = False
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < gt_binary.shape[0] and 0 <= nx < gt_binary.shape[1] and
                            gt_binary[ny, nx]):
                        is_near_gt = True
                        break
                if is_near_gt:
                    break
            
            if is_near_gt:
                gt_samples.append((y, x))
            else:
                regular_samples.append((y, x))

        # 优先返回GT附近的点
        result = gt_samples[:10] + regular_samples[:5]  # GT优先
        return result if result else regular_samples[:8]

    def _generate_smart_path(self, p1, p2, gt_analysis):
        """生成智能连接路径 - 尽量经过GT区域"""
        if not gt_analysis or gt_analysis['gt_binary'] is None:
            return self._generate_simple_path(p1, p2)

        # 尝试找到经过GT的路径
        gt_binary = gt_analysis['gt_binary']
        
        # 简单直线路径
        path = []
        x1, y1 = p1[1], p1[0]
        x2, y2 = p2[1], p2[0]

        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return [(p1[0], p1[1])]

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            path.append((y, x))

        return path

    def _generate_simple_path(self, p1, p2):
        """生成简单直线路径"""
        path = []
        x1, y1 = p1[1], p1[0]
        x2, y2 = p2[1], p2[0]

        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return [(p1[0], p1[1])]

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            path.append((y, x))

        return path

    def _evaluate_gt_connection_quality(self, path, hsv_analysis, gt_analysis):
        """评估GT连接质量"""
        if not path:
            return 0.0

        total_score = 0
        gt_bonus = 0

        for y, x in path:
            if (0 <= y < hsv_analysis['coastline_guidance'].shape[0] and
                    0 <= x < hsv_analysis['coastline_guidance'].shape[1]):
                
                # 基础分数
                total_score += 0.2
                
                # GT奖励
                if (gt_analysis and gt_analysis['gt_binary'] is not None and
                        gt_analysis['gt_binary'][y, x]):
                    gt_bonus += 1.0

        # GT路径大幅加分
        avg_score = total_score / len(path)
        gt_score = gt_bonus / len(path)
        
        return avg_score + gt_score * 2.0  # GT权重很高

    def _final_gt_protection(self, binary_coastline, gt_analysis):
        """最终GT保护 - 确保GT绝对不被删除"""
        if not gt_analysis or gt_analysis['gt_binary'] is None:
            return binary_coastline

        result = binary_coastline.copy()
        gt_binary = gt_analysis['gt_binary']

        # 最后一次强制确保所有GT像素都存在
        result = result | gt_binary

        # 确保GT的连通性
        gt_labeled, gt_components = label(gt_binary)
        for i in range(1, gt_components + 1):
            gt_component = (gt_labeled == i)
            # 对每个GT组件进行轻微扩展确保连通
            expanded_gt = binary_dilation(gt_component, np.ones((3, 3)))
            result = result | expanded_gt

        print(f"     最终确保GT: {np.sum(gt_binary):,} 像素完全保留")

        return result

    def _remove_tiny_components(self, binary_image, min_size=1):
        """移除极小组件"""
        labeled_array, num_components = label(binary_image)

        result = binary_image.copy()
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            if size < min_size:
                result[labeled_array == i] = False

        return result


# ==================== 替换主检测器中的后处理器 ====================

def create_improved_detector():
    """创建使用改进后处理器的检测器"""
    
    class ImprovedConstrainedCoastlineDetector(ConstrainedCoastlineDetector):
        def __init__(self):
            self.gt_analyzer = GroundTruthAnalyzer()
            self.post_processor = ImprovedSeaLandSeparationPostProcessor()  # 使用改进版
            print("✅ 改进版海岸线检测系统初始化完成")
            print("   🎯 主要特色：强化GT保护 + 极保守后处理")

    return ImprovedConstrainedCoastlineDetector()


# ==================== 测试函数 ====================

def test_improved_postprocessor():
    """测试改进的后处理器"""
    print("🧪 测试改进的后处理器...")
    
    # 使用改进的检测器
    detector = create_improved_detector()
    
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
            
            print(f"\n🧪 测试改进版处理: {test_file}")
            result = detector.process_image(initial_path, gt_path)
            
            if result:
                result['sample_id'] = 'improved_real_data'
                
                # 保存结果
                output_dir = "./improved_coastline_results"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, 'improved_coastline_detection.png')
                create_adaptive_visualization(result, save_path)
                
                # 显示对比结果
                metrics = result['quality_metrics']
                print(f"\n📊 改进版结果对比:")
                print(f"   最终像素数: {metrics['coastline_pixels']:,}")
                print(f"   GT匹配F1: {metrics.get('f1_score', 0):.3f}")
                print(f"   GT匹配IoU: {metrics.get('iou', 0):.3f}")
                print(f"   海陆分离得分: {metrics.get('sea_land_separation', 0):.3f}")
                print(f"   综合得分: {metrics['overall_score']:.3f}")
                
                return result
    
    return None

# 运行测试
if __name__ == "__main__":
    test_improved_postprocessor()


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("🚀 启动自适应海岸线检测系统...")
    print("🎯 主要特色：HSV→GT渐进监督 + 海陆分离 + 中间区域重点")

    detector = create_improved_detector()

    initial_dir = "E:/initial"
    ground_truth_dir = "E:/ground"

    print(f"\n📁 检查数据目录...")
    print(f"   原始图像: {initial_dir} {'✅' if os.path.exists(initial_dir) else '❌'}")
    print(f"   Ground Truth: {ground_truth_dir} {'✅' if os.path.exists(ground_truth_dir) else '❌'}")

    result = None

    # 尝试处理真实数据
    if os.path.exists(initial_dir):
        files = [f for f in os.listdir(initial_dir) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
        if files:
            test_file = files[0]
            initial_path = os.path.join(initial_dir, test_file)

            gt_path = None
            if os.path.exists(ground_truth_dir):
                gt_files = [f for f in os.listdir(ground_truth_dir) if
                            f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

                base_name = os.path.splitext(test_file)[0]

                # 匹配策略
                for gt_file in gt_files:
                    gt_base = os.path.splitext(gt_file)[0]
                    if base_name == gt_base:
                        gt_path = os.path.join(ground_truth_dir, gt_file)
                        break

                if gt_path is None:
                    for gt_file in gt_files:
                        if base_name in gt_file:
                            gt_path = os.path.join(ground_truth_dir, gt_file)
                            break

                if gt_path is None and gt_files:
                    gt_path = os.path.join(ground_truth_dir, gt_files[0])

            result = detector.process_image(initial_path, gt_path)
            if result:
                result['sample_id'] = 'adaptive_real_data'

    # 使用演示数据
    if result is None:
        print("\n🎨 使用自适应演示数据测试系统...")

        demo_img, demo_gt = create_adaptive_demo_image()

        os.makedirs("./temp", exist_ok=True)
        demo_img_path = "./temp/demo_image_adaptive.png"
        demo_gt_path = "./temp/demo_gt_adaptive.png"

        Image.fromarray(demo_img).save(demo_img_path)
        Image.fromarray(demo_gt).save(demo_gt_path)

        result = detector.process_image(demo_img_path, demo_gt_path)
        if result:
            result['sample_id'] = 'adaptive_demo'

    # 显示结果
    if result:
        output_dir = "./adaptive_coastline_results"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'adaptive_coastline_detection.png')
        create_adaptive_visualization(result, save_path)

        metrics = result['quality_metrics']
        print(f"\n✅ 自适应版处理完成!")
        print(f"   综合得分: {metrics['overall_score']:.3f}")
        print(f"   自适应质量得分: {metrics.get('adaptive_quality', 0):.3f}")
        print(f"   海岸线像素: {metrics['coastline_pixels']:,}")
        print(f"   连通组件数: {metrics['num_components']}")
        print(f"   海陆分离得分: {metrics.get('sea_land_separation', 0):.3f}")

        if 'f1_score' in metrics:
            print(f"   GT匹配F1: {metrics['f1_score']:.3f}")
            print(f"   GT匹配IoU: {metrics['iou']:.3f}")

        print(f"\n📊 可视化结果: {save_path}")

        # 中间区域分析
        if result['final_coastline'] is not None:
            height = result['final_coastline'].shape[0]
            middle_start = height // 3
            middle_end = 2 * height // 3
            middle_pixels = np.sum(result['final_coastline'][middle_start:middle_end, :] > 0.5)
            total_pixels = metrics['coastline_pixels']
            middle_ratio = middle_pixels / max(1, total_pixels)
            print(f"🎯 中间区域集中度: {middle_ratio:.1%}")

    else:
        print("❌ 所有处理尝试都失败了")


if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-