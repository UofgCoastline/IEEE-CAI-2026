def optimize_constrained_coastline(self, max_episodes=200, max_steps_per_episode=400):
    """优化约束海岸线检测 - 重点关注中间区域"""
    print("🎯 约束海岸线优化开始 - HSV监督 + 分支约束 + 中间区域重点...")

    search_positions = np.where(self.env.search_region)
    candidate_positions = list(zip(search_positions[0], search_positions[1]))

    if not candidate_positions:
        print("   ⚠️ 未找到搜索区域")
        return self.env.current_coastline

    # 智能起始点选择 - 优先选择中间区域
    middle_third_starts = []
    hsv_guided_starts = []

    height = self.env.height
    middle_start = height // 3
    middle_end = 2 * height // 3

    for pos in candidate_positions:
        y, x = pos

        # 中间1/3区域的点
        if middle_start <= y <= middle_end:
            guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
            transition_score = self.env.hsv_analysis['transition_strength'][y, x]

            if guidance_score > 0.3 or transition_score > 0.4:
                middle_third_starts.append(pos)

        # HSV高质量点（所有区域）
        guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
        transition_score = self.env.hsv_analysis['transition_strength'][y, x]
        if guidance_score > 0.4 or transition_score > 0.5:
            hsv_guided_starts.append(pos)

    # 如果中间区域起始点太少，补充一些
    if len(middle_third_starts) < 20:
        for pos in candidate_positions[::2]:  # 每2个取1个
            y, x = pos
            if middle_start <= y <= middle_end and pos not in middle_third_starts:
                middle_third_starts.append(pos)
                if len(middle_third_starts) >= 20:
                    break

    if not middle_third_starts:
        middle_third_starts = [pos for pos in candidate_positions if middle_start <= pos[0] <= middle_end]

    if not hsv_guided_starts:
        hsv_guided_starts = candidate_positions[:50]

    print(f"   中间区域起始点: {len(middle_third_starts)}")
    print(f"   HSV引导起始点: {len(hsv_guided_starts)}")

    episode_rewards = []
    total_improvements = 0
    hsv_quality_scores = []

    for episode in range(max_episodes):
        # 智能起始点策略 - 大幅提高中间区域的选择概率
        if episode < max_episodes // 2:
            # 前50%：80%概率从中间区域开始
            if random.random() < 0.8 and middle_third_starts:
                start_position = random.choice(middle_third_starts)
            else:
                start_position = random.choice(hsv_guided_starts)
        elif episode < 3 * max_episodes // 4:
            # 50%-75%：60%概率从中间区域开始
            if random.random() < 0.6 and middle_third_starts:
                start_position = random.choice(middle_third_starts)
            elif self.env.gt_analysis and random.random() < 0.7:
                # GT引导，但限制在中间区域
                gt_positions = np.where(self.env.gt_analysis['gt_binary'])
                if len(gt_positions[0]) > 0:
                    # 过滤GT位置，只选择中间区域的
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
                start_position = random.choice(hsv_guided_starts)
        else:
            # 后25%：连通性断点，但仍优先中间区域
            start_position = self._find_connectivity_break_start(candidate_positions,
                                                                 prefer_middle_region=True)
            if start_position is None:
                if middle_third_starts:
                    start_position = random.choice(middle_third_starts)
                else:
                    start_position = random.choice(hsv_guided_starts)

        current_position = start_position
        episode_reward = 0
        episode_improvements = 0

        for step in range(max_steps_per_episode):
            # 获取状态
            rgb_state, hsv_state = self.env.get_state_tensor(current_position)
            enhanced_features = self.env.get_enhanced_features(current_position)

            action = self.select_action(rgb_state, hsv_state, current_position,
                                        enhanced_features, training=True)

            next_position, reward = self.env.step(current_position, action)
            episode_reward += reward

            # 获取下一状态
            next_rgb_state, next_hsv_state = self.env.get_state_tensor(next_position)
            next_enhanced_features = self.env.get_enhanced_features(next_position)

            # 存储经验
            current_state = (rgb_state, hsv_state, current_position, enhanced_features)
            next_state = (next_rgb_state, next_hsv_state, next_position,
                          next_enhanced_features) if reward > -50 else None  # 调整阈值

            self.memory.append((current_state, action, next_state, reward))

            # 自适应海岸线更新 - 提高中间区域的更新阈值
            y_pos = next_position[0]
            is_middle_region = middle_start <= y_pos <= middle_end

            if reward > 20.0:  # 高质量检测
                update_value = 0.9 if is_middle_region else 0.7
                self.env.update_coastline(next_position, update_value)
                episode_improvements += 1
                total_improvements += 1
            elif reward > 10.0:  # 中等质量检测
                update_value = 0.6 if is_middle_region else 0.4
                self.env.update_coastline(next_position, update_value)
                episode_improvements += 1
            elif reward > 5.0 and is_middle_region:  # 中间区域降低阈值
                self.env.update_coastline(next_position, 0.3)

            # 训练
            if self.steps_done % self.train_freq == 0:
                loss = self.train_step()

            # 更新目标网络
            if self.steps_done % self.target_update_freq == 0:
                self.update_target_network()

            self.steps_done += 1
            current_position = next_position

            # 早停条件 - 更严格的边缘区域惩罚
            if reward < -80:  # 严重违规（如进入边缘区域）
                break

        episode_rewards.append(episode_reward)
        self.decay_epsilon()

        # HSV质量评估
        if episode % 20 == 0:
            hsv_quality = self.env.hsv_supervisor.evaluate_prediction_quality(
                self.env.current_coastline,
                self.env.gt_analysis['gt_binary'] if self.env.gt_analysis else None,
                self.env.hsv_analysis
            )
            hsv_quality_scores.append(hsv_quality)

            avg_reward = np.mean(episode_rewards[-20:])
            current_pixels = np.sum(self.env.current_coastline > 0.3)

            # 统计中间区域的像素分布
            middle_region_pixels = np.sum(self.env.current_coastline[middle_start:middle_end, :] > 0.3)
            middle_ratio = middle_region_pixels / max(1, current_pixels)

            print(f"   Episode {episode:3d}: 平均奖励={avg_reward:6.2f}, ε={self.epsilon:.3f}, "
                  f"海岸线像素={current_pixels:,}, 中间区域比例={middle_ratio:.1%}, "
                  f"HSV质量={hsv_quality:.3f}, 本轮改进={episode_improvements}")

    final_pixels = np.sum(self.env.current_coastline > 0.3)
    middle_final_pixels = np.sum(self.env.current_coastline[middle_start:middle_end, :] > 0.3)
    final_middle_ratio = middle_final_pixels / max(1, final_pixels)

    final_hsv_quality = self.env.hsv_supervisor.evaluate_prediction_quality(
        self.env.current_coastline,
        self.env.gt_analysis['gt_binary'] if self.env.gt_analysis else None,
        self.env.hsv_analysis
    )

    print(f"   ✅ 约束优化完成")
    print(f"   总改进次数: {total_improvements}")
    print(f"   最终海岸线像素: {final_pixels:,}")
    print(f"   中间区域像素: {middle_final_pixels:,} ({final_middle_ratio:.1%})")
    print(f"   最终HSV质量得分: {final_hsv_quality:.3f}")

    return self.env.current_coastline


def _find_connectivity_break_start(self, candidate_positions, prefer_middle_region=True):
    """寻找连通性断点的起始位置 - 优先选择中间区域"""
    current_coastline = self.env.current_coastline > 0.3
    labeled_array, num_components = label(current_coastline)

    if num_components <= 1:
        return None

    # 寻找组件间的潜在连接点
    connection_candidates = []

    height = self.env.height
    middle_start = height // 3
    middle_end = 2 * height // 3

    for pos in candidate_positions[::8]:  # 采样
        y, x = pos
        if not current_coastline[y, x]:  # 不在现有海岸线上

            # 检查HSV引导
            guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
            if guidance_score < 0.3:
                continue

            # 检查周围的组件
            nearby_components = set()
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.env.height and 0 <= nx < self.env.width and
                            labeled_array[ny, nx] > 0):
                        nearby_components.add(labeled_array[ny, nx])

            if len(nearby_components) >= 2:
                # 计算连接价值
                connection_value = guidance_score + len(nearby_components) * 0.1

                # 中间区域加权
                if prefer_middle_region and middle_start <= y <= middle_end:
                    connection_value *= 2.0  # 中间区域连接点优先级更高

                connection_candidates.append((pos, connection_value))

    if connection_candidates:
        # 选择最有价值的连接点
        connection_candidates.sort(key=lambda x: x[1], reverse=True)
        return connection_candidates[0][0]

    return None  # -*- coding: utf-8 -*-


"""
约束分支海岸线检测系统 - 限制蒙特卡洛树分支 + HSV注意力监督
主要改进：
1. 横向主干分叉 + 纵向极窄范围允许分支
2. HSV作为注意力监督器评价结果
3. 好奇心机制加强探索
4. 方向性约束防止海域渗透
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

print("🌊 约束分支海岸线检测系统 - HSV注意力监督!")
print("重点：横向主干 + 纵向窄分支 + HSV监督 + 好奇心探索")
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
    """HSV注意力监督器 - 评价结果与GT的比较"""

    def __init__(self):
        print("✅ HSV注意力监督器初始化完成")
        self.water_hsv_range = self._define_water_hsv_range()
        self.land_hsv_range = self._define_land_hsv_range()

    def _define_water_hsv_range(self):
        """定义水体的HSV范围"""
        return {
            'hue_range': (180, 240),  # 蓝色调范围
            'saturation_min': 0.2,  # 降低最小饱和度
            'value_min': 0.1  # 降低最小明度
        }

    def _define_land_hsv_range(self):
        """定义陆地的HSV范围"""
        return {
            'hue_range': (60, 120),  # 绿色调范围
            'saturation_min': 0.1,  # 降低最小饱和度
            'value_min': 0.2  # 降低最小明度
        }

    def analyze_image_hsv(self, rgb_image, gt_analysis=None):
        """分析图像的HSV特征 - 结合GT信息"""
        if len(rgb_image.shape) == 3:
            # 转换为HSV
            rgb_normalized = rgb_image.astype(float) / 255.0
            hsv_image = np.zeros_like(rgb_normalized)

            for i in range(rgb_image.shape[0]):
                for j in range(rgb_image.shape[1]):
                    r, g, b = rgb_normalized[i, j]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_image[i, j] = [h * 360, s, v]  # H范围0-360度
        else:
            # 灰度图像，假设为单通道
            hsv_image = np.stack([np.zeros_like(rgb_image),
                                  np.zeros_like(rgb_image),
                                  rgb_image / 255.0], axis=2)

        # 识别水体和陆地区域
        water_mask = self._detect_water_regions(hsv_image)
        land_mask = self._detect_land_regions(hsv_image)

        # 如果有GT，使用GT信息改进水陆分割
        if gt_analysis is not None:
            water_mask, land_mask = self._refine_with_gt(
                water_mask, land_mask, gt_analysis, hsv_image
            )

        # 生成海岸线引导
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

        # 蓝色调检测
        hue_mask = ((h >= self.water_hsv_range['hue_range'][0]) &
                    (h <= self.water_hsv_range['hue_range'][1]))

        # 饱和度和明度约束
        saturation_mask = s >= self.water_hsv_range['saturation_min']
        value_mask = v >= self.water_hsv_range['value_min']

        water_mask = hue_mask & saturation_mask & value_mask

        # 形态学处理去噪
        water_mask = binary_closing(water_mask, np.ones((5, 5)))
        water_mask = binary_erosion(water_mask, np.ones((3, 3)))
        water_mask = binary_dilation(water_mask, np.ones((3, 3)))

        return water_mask

    def _detect_land_regions(self, hsv_image):
        """检测陆地区域"""
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # 绿色调检测 + 其他陆地色调
        green_hue_mask = ((h >= self.land_hsv_range['hue_range'][0]) &
                          (h <= self.land_hsv_range['hue_range'][1]))

        # 棕色/土色调检测
        brown_hue_mask = ((h >= 20) & (h <= 50))

        # 灰色/岩石色调检测
        gray_mask = (s <= 0.2) & (v >= 0.4)

        hue_mask = green_hue_mask | brown_hue_mask | gray_mask

        # 饱和度和明度约束
        saturation_mask = s >= self.land_hsv_range['saturation_min']
        value_mask = v >= self.land_hsv_range['value_min']

        land_mask = hue_mask & (saturation_mask | gray_mask) & value_mask

        # 形态学处理
        land_mask = binary_closing(land_mask, np.ones((5, 5)))
        land_mask = binary_erosion(land_mask, np.ones((2, 2)))
        land_mask = binary_dilation(land_mask, np.ones((3, 3)))

        return land_mask

    def _refine_with_gt(self, water_mask, land_mask, gt_analysis, hsv_image):
        """使用GT信息改进水陆分割 - 优化版本"""
        print("   🎯 使用GT信息改进HSV水陆分割...")

        gt_binary = gt_analysis['gt_binary']
        gt_edge_region = gt_analysis['edge_region']

        # 快速采样策略 - 大幅减少计算量
        edge_positions = np.where(gt_edge_region)
        if len(edge_positions[0]) == 0:
            return water_mask, land_mask

        # 大幅减少采样点数
        sample_step = max(1, len(edge_positions[0]) // 50)  # 最多50个采样点
        sample_indices = range(0, len(edge_positions[0]), sample_step)

        water_samples = []
        land_samples = []

        print(f"     采样点数: {len(sample_indices)}")

        # 快速采样
        for idx in sample_indices:
            y, x = edge_positions[0][idx], edge_positions[1][idx]

            # 只检查直接邻居，减少计算
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < hsv_image.shape[0] and 0 <= nx < hsv_image.shape[1]:
                    pixel_hsv = hsv_image[ny, nx]

                    # 简化分类逻辑
                    if pixel_hsv[2] < 0.35:  # 暗色 -> 水域
                        water_samples.append(pixel_hsv)
                    elif pixel_hsv[2] > 0.4:  # 亮色 -> 陆地
                        land_samples.append(pixel_hsv)

        # 检查样本数量
        if len(water_samples) < 5 or len(land_samples) < 5:
            print(f"     样本不足，使用原始HSV分割")
            return water_mask, land_mask

        print(f"     水域样本: {len(water_samples)}, 陆地样本: {len(land_samples)}")

        # 计算样本中心点而不是重新分类所有像素
        water_samples = np.array(water_samples)
        land_samples = np.array(land_samples)

        water_center = np.mean(water_samples, axis=0)
        land_center = np.mean(land_samples, axis=0)

        print(f"     水域中心HSV: [{water_center[0]:.1f}, {water_center[1]:.2f}, {water_center[2]:.2f}]")
        print(f"     陆地中心HSV: [{land_center[0]:.1f}, {land_center[1]:.2f}, {land_center[2]:.2f}]")

        # 基于中心点快速重新分类 - 只处理GT附近区域
        refined_water_mask = water_mask.copy()
        refined_land_mask = land_mask.copy()

        # 只在GT扩展区域内重新分类
        search_region = binary_dilation(gt_edge_region, np.ones((10, 10)))
        search_positions = np.where(search_region)

        for i in range(len(search_positions[0])):
            y, x = search_positions[0][i], search_positions[1][i]
            pixel = hsv_image[y, x]

            # 计算到中心点的距离
            water_dist = np.linalg.norm(pixel - water_center)
            land_dist = np.linalg.norm(pixel - land_center)

            # 重新分类
            if water_dist < land_dist * 0.9:  # 偏向水域
                refined_water_mask[y, x] = True
                refined_land_mask[y, x] = False
            elif land_dist < water_dist * 0.9:  # 偏向陆地
                refined_land_mask[y, x] = True
                refined_water_mask[y, x] = False

        # 快速形态学处理
        kernel = np.ones((3, 3))
        refined_water_mask = binary_closing(refined_water_mask, kernel)
        refined_land_mask = binary_closing(refined_land_mask, kernel)

        print(f"     改进后水域像素: {np.sum(refined_water_mask):,}")
        print(f"     改进后陆地像素: {np.sum(refined_land_mask):,}")

        return refined_water_mask, refined_land_mask

    def _generate_coastline_guidance(self, water_mask, land_mask, gt_analysis=None):
        """生成海岸线引导图 - 结合GT信息"""
        # 计算水体和陆地的边界
        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask

        # 海岸线是水体和陆地边界的交集区域
        coastline_candidates = water_boundary | land_boundary  # 改为并集，扩大候选区域

        # 如果有GT，强化GT附近的引导
        if gt_analysis is not None:
            gt_binary = gt_analysis['gt_binary']
            gt_edge_region = gt_analysis['edge_region']

            # GT区域的强引导
            gt_guidance = binary_dilation(gt_binary, np.ones((5, 5)))
            coastline_candidates = coastline_candidates | gt_guidance

            print(f"     GT增强后的引导区域: {np.sum(coastline_candidates):,} 像素")

        # 扩展海岸线候选区域
        coastline_guidance = coastline_candidates.copy()
        for _ in range(3):  # 增加扩展次数
            coastline_guidance = binary_dilation(coastline_guidance, np.ones((3, 3)))

        # 计算引导强度
        from scipy.ndimage import distance_transform_edt

        if np.any(water_mask):
            water_dist = distance_transform_edt(~water_mask)
        else:
            water_dist = np.ones_like(water_mask, dtype=float) * 10

        if np.any(land_mask):
            land_dist = distance_transform_edt(~land_mask)
        else:
            land_dist = np.ones_like(land_mask, dtype=float) * 10

        # 海岸线引导强度：距离水陆边界都近的区域强度高
        guidance_strength = np.exp(-0.05 * (water_dist + land_dist))  # 减小衰减系数

        # 如果有GT，在GT附近给予额外强度
        if gt_analysis is not None:
            gt_dist = distance_transform_edt(~gt_analysis['gt_binary'])
            gt_bonus = np.exp(-0.1 * gt_dist)
            guidance_strength = guidance_strength + gt_bonus * 0.8

        guidance_strength = coastline_guidance * guidance_strength

        # 归一化
        if guidance_strength.max() > 0:
            guidance_strength = guidance_strength / guidance_strength.max()

        return guidance_strength

    def _calculate_transition_strength(self, hsv_image, water_mask, land_mask, gt_analysis=None):
        """计算过渡区域强度 - 结合GT信息"""
        # 计算HSV梯度
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # 色调梯度（需要考虑环形特性）
        h_grad = np.abs(np.gradient(h)[0]) + np.abs(np.gradient(h)[1])
        s_grad = np.abs(np.gradient(s)[0]) + np.abs(np.gradient(s)[1])
        v_grad = np.abs(np.gradient(v)[0]) + np.abs(np.gradient(v)[1])

        # 综合梯度强度
        transition_strength = (h_grad * 0.4 + s_grad * 0.3 + v_grad * 0.3)

        if transition_strength.max() > transition_strength.min():
            transition_strength = (transition_strength - transition_strength.min()) / (
                    transition_strength.max() - transition_strength.min() + 1e-8)

        # 在水陆边界附近的过渡强度更重要
        boundary_mask = binary_dilation(water_mask, np.ones((5, 5))) | binary_dilation(land_mask, np.ones((5, 5)))
        transition_strength = transition_strength * (1 + boundary_mask * 1.5)

        # 如果有GT，在GT附近增强过渡强度
        if gt_analysis is not None:
            gt_edge_region = gt_analysis['edge_region']
            transition_strength = transition_strength * (1 + gt_edge_region * 2.0)

        return transition_strength

    def evaluate_prediction_quality(self, prediction, ground_truth, hsv_analysis):
        """评价预测质量 - HSV监督"""
        quality_score = 0.0

        pred_binary = (prediction > 0.5).astype(bool)

        # 1. 与HSV引导的一致性
        coastline_guidance = hsv_analysis['coastline_guidance']
        guidance_alignment = np.sum(pred_binary * coastline_guidance) / (np.sum(pred_binary) + 1e-8)
        quality_score += guidance_alignment * 0.3

        # 2. 过渡区域的覆盖质量
        transition_strength = hsv_analysis['transition_strength']
        transition_coverage = np.sum(pred_binary * transition_strength) / (np.sum(transition_strength) + 1e-8)
        quality_score += transition_coverage * 0.2

        # 3. 避免水域渗透
        water_mask = hsv_analysis['water_mask']
        water_penetration = np.sum(pred_binary & water_mask) / (np.sum(pred_binary) + 1e-8)
        quality_score -= water_penetration * 0.5  # 惩罚水域渗透

        # 4. GT一致性（如果有GT）
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
            # 无GT时，基于HSV的合理性评分
            hsv_reasonableness = self._evaluate_hsv_reasonableness(pred_binary, hsv_analysis)
            quality_score += hsv_reasonableness * 0.4

        return max(0.0, min(1.0, quality_score))

    def _evaluate_hsv_reasonableness(self, prediction, hsv_analysis):
        """评价基于HSV的合理性"""
        water_mask = hsv_analysis['water_mask']
        land_mask = hsv_analysis['land_mask']

        # 预测的海岸线应该在水陆边界附近
        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask

        boundary_region = water_boundary | land_boundary
        boundary_coverage = np.sum(prediction & boundary_region) / (np.sum(prediction) + 1e-8)

        return boundary_coverage


# ==================== 约束的动作空间 ====================

class ConstrainedActionSpace:
    """约束的动作空间 - 限制分支方向"""

    def __init__(self):
        # 基础8方向动作
        self.base_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                             (0, 1), (1, -1), (1, 0), (1, 1)]

        # 主要横向动作（海岸线通常是横向的）
        self.primary_horizontal = [(0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 限制的纵向动作（只允许很小的纵向移动）
        self.limited_vertical = [(-1, 0), (1, 0)]

        print("✅ 约束动作空间初始化完成 - 主横向 + 限纵向")

    def get_allowed_actions(self, current_position, coastline_state, hsv_analysis):
        """获取当前位置允许的动作"""
        y, x = current_position
        allowed_actions = []

        # 分析当前位置的上下文
        context = self._analyze_position_context(current_position, coastline_state, hsv_analysis)

        for i, action in enumerate(self.base_actions):
            if self._is_action_allowed(action, context, current_position, hsv_analysis):
                allowed_actions.append(i)

        return allowed_actions if allowed_actions else [0, 1, 3, 4]  # 至少允许基本移动

    def _analyze_position_context(self, position, coastline_state, hsv_analysis):
        """分析位置上下文"""
        y, x = position

        # 检查周围的海岸线密度
        y_start, y_end = max(0, y - 3), min(coastline_state.shape[0], y + 4)
        x_start, x_end = max(0, x - 3), min(coastline_state.shape[1], x + 4)

        local_coastline = coastline_state[y_start:y_end, x_start:x_end]
        coastline_density = np.mean(local_coastline > 0.3)

        # 检查是否在水域附近
        if hsv_analysis:
            water_mask = hsv_analysis['water_mask']
            near_water = water_mask[y, x] or np.any(water_mask[y_start:y_end, x_start:x_end])
        else:
            near_water = False

        # 计算主要海岸线方向
        main_direction = self._estimate_main_coastline_direction(position, coastline_state)

        return {
            'coastline_density': coastline_density,
            'near_water': near_water,
            'main_direction': main_direction,
            'vertical_constraint_level': 'high' if near_water else 'medium'
        }

    def _estimate_main_coastline_direction(self, position, coastline_state):
        """估计主要海岸线方向"""
        y, x = position

        # 检查水平和垂直方向的海岸线连续性
        horizontal_score = 0
        vertical_score = 0

        # 水平方向检查
        for dx in [-5, -3, -1, 1, 3, 5]:
            if 0 <= x + dx < coastline_state.shape[1]:
                if coastline_state[y, x + dx] > 0.3:
                    horizontal_score += 1

        # 垂直方向检查
        for dy in [-5, -3, -1, 1, 3, 5]:
            if 0 <= y + dy < coastline_state.shape[0]:
                if coastline_state[y + dy, x] > 0.3:
                    vertical_score += 1

        if horizontal_score > vertical_score * 1.5:
            return 'horizontal'
        elif vertical_score > horizontal_score * 1.5:
            return 'vertical'
        else:
            return 'mixed'

    def _is_action_allowed(self, action, context, current_position, hsv_analysis):
        """判断动作是否被允许"""
        dy, dx = action

        # 强制约束：如果在水域附近，严格限制纵向移动
        if context['near_water'] and abs(dy) > 0:
            # 只允许非常小的纵向移动
            if abs(dy) > 1 or (abs(dy) == 1 and abs(dx) == 0):
                return False

        # 主干方向约束
        if context['main_direction'] == 'horizontal':
            # 主要海岸线是横向的，限制纵向移动
            if abs(dy) > 1:
                return False
            # 纵向移动必须伴随横向移动
            if abs(dy) == 1 and dx == 0:
                return False

        # 高密度区域约束
        if context['coastline_density'] > 0.7:
            # 在高密度海岸线区域，避免大的移动
            if abs(dy) + abs(dx) > 2:
                return False

        # 水域渗透检查
        if hsv_analysis:
            y, x = current_position
            new_y, new_x = y + dy, x + dx

            if (0 <= new_y < hsv_analysis['water_mask'].shape[0] and
                    0 <= new_x < hsv_analysis['water_mask'].shape[1]):

                # 检查目标位置是否深入水域
                if hsv_analysis['water_mask'][new_y, new_x]:
                    # 检查周围是否也都是水域（深入水域的标志）
                    water_neighbors = 0
                    for check_dy in [-1, 0, 1]:
                        for check_dx in [-1, 0, 1]:
                            check_y, check_x = new_y + check_dy, new_x + check_dx
                            if (0 <= check_y < hsv_analysis['water_mask'].shape[0] and
                                    0 <= check_x < hsv_analysis['water_mask'].shape[1]):
                                if hsv_analysis['water_mask'][check_y, check_x]:
                                    water_neighbors += 1

                    # 如果周围大部分都是水域，不允许这个动作
                    if water_neighbors > 6:
                        return False

        return True


# ==================== 好奇心驱动探索 ====================

class CuriosityDrivenExploration:
    """好奇心驱动的探索机制"""

    def __init__(self, exploration_decay=0.995):
        self.visit_history = {}
        self.exploration_bonus = {}
        self.exploration_decay = exploration_decay
        self.step_count = 0
        print("✅ 好奇心驱动探索机制初始化完成")

    def get_curiosity_bonus(self, position, hsv_analysis, current_coastline):
        """获取好奇心奖励"""
        y, x = position
        pos_key = f"{y}_{x}"

        # 访问次数奖励
        visit_count = self.visit_history.get(pos_key, 0)
        visit_bonus = max(0, 10.0 - visit_count * 2.0)

        # HSV引导的探索奖励
        hsv_bonus = 0.0
        if hsv_analysis:
            # 在HSV引导区域探索给额外奖励
            coastline_guidance = hsv_analysis['coastline_guidance']
            if coastline_guidance[y, x] > 0.3:
                hsv_bonus = coastline_guidance[y, x] * 15.0

            # 在高过渡强度区域探索
            transition_strength = hsv_analysis['transition_strength']
            if transition_strength[y, x] > 0.5:
                hsv_bonus += transition_strength[y, x] * 10.0

        # 连接性探索奖励
        connectivity_bonus = self._calculate_connectivity_exploration_bonus(
            position, current_coastline
        )

        # 边界探索奖励
        boundary_bonus = self._calculate_boundary_exploration_bonus(
            position, hsv_analysis
        )

        total_bonus = visit_bonus + hsv_bonus + connectivity_bonus + boundary_bonus

        # 记录访问
        self.visit_history[pos_key] = visit_count + 1
        self.step_count += 1

        # 定期衰减探索奖励
        if self.step_count % 100 == 0:
            self._decay_exploration_bonuses()

        return total_bonus

    def _calculate_connectivity_exploration_bonus(self, position, current_coastline):
        """计算连接性探索奖励"""
        y, x = position

        # 寻找孤立的海岸线组件
        labeled_array, num_components = label(current_coastline > 0.3)

        if num_components <= 1:
            return 0.0

        # 如果当前位置能连接不同组件，给予奖励
        nearby_components = set()
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = y + dy, x + dx
                if (0 <= ny < current_coastline.shape[0] and
                        0 <= nx < current_coastline.shape[1]):
                    component_id = labeled_array[ny, nx]
                    if component_id > 0:
                        nearby_components.add(component_id)

        # 如果附近有多个组件，说明这里是连接的关键位置
        if len(nearby_components) >= 2:
            return 20.0 * len(nearby_components)

        return 0.0

    def _calculate_boundary_exploration_bonus(self, position, hsv_analysis):
        """计算边界探索奖励"""
        if not hsv_analysis:
            return 0.0

        y, x = position
        water_mask = hsv_analysis['water_mask']
        land_mask = hsv_analysis['land_mask']

        # 检查是否在水陆边界
        is_near_water_boundary = False
        is_near_land_boundary = False

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if (0 <= ny < water_mask.shape[0] and 0 <= nx < water_mask.shape[1]):
                    if water_mask[ny, nx]:
                        is_near_water_boundary = True
                    if land_mask[ny, nx]:
                        is_near_land_boundary = True

        # 如果同时靠近水和陆地，这是很好的海岸线位置
        if is_near_water_boundary and is_near_land_boundary:
            return 15.0
        elif is_near_water_boundary or is_near_land_boundary:
            return 8.0

        return 0.0

    def _decay_exploration_bonuses(self):
        """衰减探索奖励"""
        for key in list(self.exploration_bonus.keys()):
            self.exploration_bonus[key] *= self.exploration_decay
            if self.exploration_bonus[key] < 0.1:
                del self.exploration_bonus[key]


# ==================== 约束的DQN网络 ====================

class ConstrainedCoastlineDQN(nn.Module):
    """约束的海岸线DQN网络 - HSV监督"""

    def __init__(self, input_channels=3, hidden_dim=256, action_dim=8):
        super(ConstrainedCoastlineDQN, self).__init__()

        # RGB特征提取器
        self.rgb_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # HSV监督特征提取器
        self.hsv_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.feature_dim = 128 * 8 * 8 + 64 * 8 * 8

        # Q值网络
        self.q_network = nn.Sequential(
            nn.Linear(self.feature_dim + 2 + 25, hidden_dim),  # 25个增强特征
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, action_dim)
        )

        # 动作掩码网络
        self.action_mask_network = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_state, hsv_state, position, enhanced_features):
        # 特征提取
        rgb_features = self.rgb_extractor(rgb_state)
        hsv_features = self.hsv_extractor(hsv_state)

        # 展平
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        hsv_features = hsv_features.view(hsv_features.size(0), -1)

        # 位置特征
        position_norm = position.float() / 400.0

        # 特征融合
        combined = torch.cat([rgb_features, hsv_features, position_norm, enhanced_features], dim=1)

        # Q值计算
        q_values = self.q_network(combined)

        # 动作掩码
        action_mask = self.action_mask_network(enhanced_features)

        # 应用掩码
        masked_q_values = q_values * action_mask - (1 - action_mask) * 1e6

        return masked_q_values


# ==================== 约束环境 ====================

class ConstrainedCoastlineEnvironment:
    """约束的海岸线环境 - 限制分支 + HSV监督"""

    def __init__(self, image, gt_analysis):
        self.image = image
        self.gt_analysis = gt_analysis
        self.current_coastline = np.zeros(image.shape[:2], dtype=float)
        self.height, self.width = image.shape[:2]

        # HSV监督器
        self.hsv_supervisor = HSVAttentionSupervisor()
        self.hsv_analysis = self.hsv_supervisor.analyze_image_hsv(image, gt_analysis)

        # 约束动作空间
        self.action_constraints = ConstrainedActionSpace()
        self.base_actions = self.action_constraints.base_actions
        self.action_dim = len(self.base_actions)

        # 好奇心探索
        self.curiosity_explorer = CuriosityDrivenExploration()

        # 边缘检测（简化版）
        self.edge_map = self._detect_edges()

        # 设置搜索区域
        self._setup_constrained_search_region()

        print(f"✅ 约束海岸线环境初始化完成")
        print(f"   HSV水域像素: {np.sum(self.hsv_analysis['water_mask']):,}")
        print(f"   HSV陆地像素: {np.sum(self.hsv_analysis['land_mask']):,}")
        print(f"   海岸线引导区域: {np.sum(self.hsv_analysis['coastline_guidance'] > 0.3):,}")

    def _detect_edges(self):
        """简化的边缘检测"""
        if len(self.image.shape) == 3:
            gray = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = self.image.copy()

        # Sobel边缘检测
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = ndimage.convolve(gray, sobel_x, mode='constant')
        grad_y = ndimage.convolve(gray, sobel_y, mode='constant')

        edge_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-8)

        return edge_magnitude

    def _setup_constrained_search_region(self):
        """设置约束的搜索区域 - 重点关注中间1/3区域"""
        print("🎯 设置智能搜索区域 - 重点关注中间1/3...")

        # 第一步：确定图像的有效海岸线区域
        effective_region = self._identify_effective_coastline_region()

        # 第二步：基于HSV分析的搜索区域
        coastline_guidance = self.hsv_analysis['coastline_guidance']
        transition_strength = self.hsv_analysis['transition_strength']

        # 主要搜索区域：HSV引导 + 过渡区域
        primary_region = (coastline_guidance > 0.2) | (transition_strength > 0.4)

        # 第三步：应用有效区域限制
        self.search_region = primary_region & effective_region

        # 扩展搜索区域，但限制在有效区域内
        for _ in range(2):
            expanded = binary_dilation(self.search_region, np.ones((3, 3), dtype=bool))
            self.search_region = expanded & effective_region  # 确保不超出有效区域

        # 排除深水区域
        deep_water = self.hsv_analysis['water_mask']
        for _ in range(5):  # 深水区域向内收缩
            deep_water = binary_erosion(deep_water, np.ones((3, 3), dtype=bool))

        self.search_region = self.search_region & ~deep_water

        # 如果有GT，进一步优化，但仍限制在有效区域内
        if self.gt_analysis:
            gt_region = self.gt_analysis['edge_region'] & effective_region
            self.search_region = self.search_region | gt_region

        # 统计信息
        total_pixels = self.height * self.width
        effective_pixels = np.sum(effective_region)
        search_pixels = np.sum(self.search_region)

        print(f"   有效区域像素: {effective_pixels:,} ({effective_pixels / total_pixels:.1%})")
        print(f"   搜索区域像素: {search_pixels:,} ({search_pixels / total_pixels:.1%})")

    def _identify_effective_coastline_region(self):
        """智能识别有效的海岸线区域 - 重点关注中间1/3"""
        height, width = self.height, self.width

        # 方法1：基于GT分布分析
        if self.gt_analysis and self.gt_analysis['gt_binary'] is not None:
            gt_binary = self.gt_analysis['gt_binary']
            gt_positions = np.where(gt_binary)

            if len(gt_positions[0]) > 0:
                # 分析GT的垂直分布
                y_coords = gt_positions[0]
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                y_center = (y_min + y_max) // 2
                y_range = y_max - y_min

                # 扩展有效区域：GT范围 + 适当边界
                margin = max(20, y_range // 4)  # 至少20像素边界
                effective_y_min = max(0, y_min - margin)
                effective_y_max = min(height, y_max + margin)

                print(f"   基于GT分析 - Y范围: {effective_y_min}-{effective_y_max} (GT: {y_min}-{y_max})")
            else:
                # GT为空，使用默认中间1/3
                effective_y_min = height // 3
                effective_y_max = 2 * height // 3
                print(f"   GT为空，使用中间1/3 - Y范围: {effective_y_min}-{effective_y_max}")
        else:
            # 方法2：基于HSV引导分析
            coastline_guidance = self.hsv_analysis['coastline_guidance']

            # 分析每行的引导强度
            row_guidance = np.mean(coastline_guidance, axis=1)

            # 找到引导强度较高的区域
            high_guidance_rows = np.where(row_guidance > np.percentile(row_guidance, 70))[0]

            if len(high_guidance_rows) > 0:
                y_min_guidance = np.min(high_guidance_rows)
                y_max_guidance = np.max(high_guidance_rows)
                y_range_guidance = y_max_guidance - y_min_guidance

                # 扩展有效区域
                margin = max(30, y_range_guidance // 3)
                effective_y_min = max(0, y_min_guidance - margin)
                effective_y_max = min(height, y_max_guidance + margin)

                print(
                    f"   基于HSV引导分析 - Y范围: {effective_y_min}-{effective_y_max} (引导: {y_min_guidance}-{y_max_guidance})")
            else:
                # 方法3：默认中间1/3策略
                effective_y_min = height // 3
                effective_y_max = 2 * height // 3
                print(f"   使用默认中间1/3策略 - Y范围: {effective_y_min}-{effective_y_max}")

        # 创建有效区域掩码
        effective_region = np.zeros((height, width), dtype=bool)
        effective_region[effective_y_min:effective_y_max, :] = True

        # 额外优化：基于水陆分布进一步细化
        effective_region = self._refine_effective_region(effective_region)

        return effective_region

    def _refine_effective_region(self, initial_region):
        """细化有效区域 - 基于水陆分布"""
        water_mask = self.hsv_analysis['water_mask']
        land_mask = self.hsv_analysis['land_mask']

        # 分析每行的水陆比例
        refined_region = initial_region.copy()

        for y in range(self.height):
            if not initial_region[y, 0]:  # 不在初始有效区域内
                continue

            row_water = np.mean(water_mask[y, :])
            row_land = np.mean(land_mask[y, :])

            # 如果某行几乎全是水或全是陆地，降低其重要性
            if row_water > 0.9 or row_land > 0.9:
                # 减少这一行在有效区域中的权重
                refined_region[y, :] = False
            elif row_water > 0.1 and row_land > 0.1:
                # 有水有陆地的行更重要，保持
                refined_region[y, :] = True

        return refined_region

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

        # HSV监督状态
        hsv_state = np.zeros((3, window_size, window_size), dtype=np.float32)

        # HSV引导
        guidance_window = self.hsv_analysis['coastline_guidance'][y_start:y_end, x_start:x_end]
        hsv_state[0, :actual_h, :actual_w] = guidance_window

        # 过渡强度
        transition_window = self.hsv_analysis['transition_strength'][y_start:y_end, x_start:x_end]
        hsv_state[1, :actual_h, :actual_w] = transition_window

        # 水域掩码
        water_window = self.hsv_analysis['water_mask'][y_start:y_end, x_start:x_end].astype(float)
        hsv_state[2, :actual_h, :actual_w] = water_window

        rgb_tensor = torch.FloatTensor(rgb_state).unsqueeze(0).to(device)
        hsv_tensor = torch.FloatTensor(hsv_state).unsqueeze(0).to(device)

        return rgb_tensor, hsv_tensor

    def get_enhanced_features(self, position):
        """获取增强特征 - 包含HSV和约束信息"""
        y, x = position

        if not (0 <= y < self.height and 0 <= x < self.width):
            return torch.zeros(25, dtype=torch.float32, device=device).unsqueeze(0)

        features = np.zeros(25, dtype=np.float32)

        # 基础特征
        features[0] = self.edge_map[y, x]

        # HSV监督特征
        features[1] = self.hsv_analysis['coastline_guidance'][y, x]
        features[2] = self.hsv_analysis['transition_strength'][y, x]
        features[3] = 1.0 if self.hsv_analysis['water_mask'][y, x] else 0.0
        features[4] = 1.0 if self.hsv_analysis['land_mask'][y, x] else 0.0

        # 局部HSV统计
        y_start, y_end = max(0, y - 3), min(self.height, y + 4)
        x_start, x_end = max(0, x - 3), min(self.width, x + 4)

        local_guidance = self.hsv_analysis['coastline_guidance'][y_start:y_end, x_start:x_end]
        local_transition = self.hsv_analysis['transition_strength'][y_start:y_end, x_start:x_end]
        local_water = self.hsv_analysis['water_mask'][y_start:y_end, x_start:x_end]

        if local_guidance.size > 0:
            features[5] = np.mean(local_guidance)
            features[6] = np.max(local_guidance)
            features[7] = np.std(local_guidance)

        if local_transition.size > 0:
            features[8] = np.mean(local_transition)
            features[9] = np.max(local_transition)

        features[10] = np.mean(local_water.astype(float))

        # GT特征
        if self.gt_analysis:
            try:
                features[11] = 1.0 if self.gt_analysis['gt_binary'][y, x] else 0.0

                if np.any(self.gt_analysis['gt_binary']):
                    gt_coords = np.where(self.gt_analysis['gt_binary'])
                    if len(gt_coords[0]) > 0:
                        distances = np.sqrt((gt_coords[0] - y) ** 2 + (gt_coords[1] - x) ** 2)
                        min_dist = np.min(distances)
                        features[12] = min(1.0, min_dist / 20.0)

                features[13] = self.gt_analysis['density_map'][y, x]
            except (IndexError, KeyError):
                pass

        # 约束和探索特征
        allowed_actions = self.action_constraints.get_allowed_actions(
            position, self.current_coastline, self.hsv_analysis
        )
        features[14] = len(allowed_actions) / 8.0  # 动作自由度

        # 好奇心特征
        curiosity_bonus = self.curiosity_explorer.get_curiosity_bonus(
            position, self.hsv_analysis, self.current_coastline
        )
        features[15] = min(1.0, curiosity_bonus / 50.0)  # 归一化好奇心奖励

        # 当前海岸线特征
        local_coastline = self.current_coastline[y_start:y_end, x_start:x_end]
        if local_coastline.size > 0:
            features[16] = np.mean(local_coastline)
            features[17] = np.sum(local_coastline > 0.5) / max(1, local_coastline.size)
            features[18] = np.sum(local_coastline > 0.3) / max(1, local_coastline.size)

        # 位置特征
        features[19] = y / self.height
        features[20] = x / self.width

        # 方向约束特征
        context = self.action_constraints._analyze_position_context(
            position, self.current_coastline, self.hsv_analysis
        )
        features[21] = context['coastline_density']
        features[22] = 1.0 if context['near_water'] else 0.0
        features[23] = {'horizontal': 0.0, 'vertical': 1.0, 'mixed': 0.5}.get(context['main_direction'], 0.5)

        # 边界距离特征
        water_boundary_dist = self._calculate_boundary_distance(position, self.hsv_analysis['water_mask'])
        features[24] = min(1.0, water_boundary_dist / 10.0)

        return torch.FloatTensor(features).unsqueeze(0).to(device)

    def _calculate_boundary_distance(self, position, mask):
        """计算到边界的距离"""
        y, x = position

        # 简化的边界距离计算
        min_dist = float('inf')
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                ny, nx = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if mask[ny, nx] != mask[y, x]:  # 边界
                        dist = math.sqrt(dy * dy + dx * dx)
                        min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 10.0

    def step(self, position, action_idx):
        """执行约束的动作"""
        # 检查动作是否被允许
        allowed_actions = self.action_constraints.get_allowed_actions(
            position, self.current_coastline, self.hsv_analysis
        )

        if action_idx not in allowed_actions:
            # 如果动作不被允许，选择最近的允许动作
            action_idx = allowed_actions[0] if allowed_actions else 0

        y, x = position
        dy, dx = self.base_actions[action_idx]

        new_y = np.clip(y + dy, 0, self.height - 1)
        new_x = np.clip(x + dx, 0, self.width - 1)

        new_position = (new_y, new_x)
        reward = self._calculate_constrained_reward(position, new_position, action_idx)

        return new_position, reward

    def _calculate_constrained_reward(self, old_pos, new_pos, action_idx):
        """计算约束的奖励函数 - 重点奖励中间区域"""
        y, x = new_pos
        reward = 0.0

        # 边界检查
        if not (0 <= y < self.height and 0 <= x < self.width):
            return -50.0

        # 搜索区域限制 - 强化惩罚
        if not self.search_region[y, x]:
            # 检查是否在无效区域（上1/3或下1/3的边缘区域）
            if y < self.height // 3 or y > 2 * self.height // 3:
                return -100.0  # 强烈惩罚在边缘区域的探索
            else:
                return -30.0

        # 区域位置奖励 - 新增
        region_bonus = self._calculate_region_position_bonus(y)
        reward += region_bonus

        # HSV监督奖励
        hsv_reward = self._calculate_hsv_reward(new_pos)
        reward += hsv_reward * 30.0

        # 好奇心奖励
        curiosity_reward = self.curiosity_explorer.get_curiosity_bonus(
            new_pos, self.hsv_analysis, self.current_coastline
        )
        reward += curiosity_reward

        # GT奖励（如果有）- 增强中间区域的GT奖励
        if self.gt_analysis and self.gt_analysis['gt_binary'] is not None:
            if self.gt_analysis['gt_binary'][y, x]:
                base_gt_reward = 40.0
                # 在中间区域的GT匹配给予额外奖励
                if self.height // 3 <= y <= 2 * self.height // 3:
                    base_gt_reward *= 1.5
                reward += base_gt_reward
            else:
                gt_coords = np.where(self.gt_analysis['gt_binary'])
                if len(gt_coords[0]) > 0:
                    distances = np.sqrt((gt_coords[0] - y) ** 2 + (gt_coords[1] - x) ** 2)
                    min_dist = np.min(distances)

                    if min_dist <= 3:
                        gt_proximity_reward = 30.0 - min_dist * 5.0
                        # 中间区域GT接近度奖励加成
                        if self.height // 3 <= y <= 2 * self.height // 3:
                            gt_proximity_reward *= 1.3
                        reward += gt_proximity_reward
                    elif min_dist <= 8:
                        gt_proximity_reward = 20.0 - min_dist * 2.0
                        if self.height // 3 <= y <= 2 * self.height // 3:
                            gt_proximity_reward *= 1.2
                        reward += gt_proximity_reward

        # 水域渗透惩罚（强化）
        if self.hsv_analysis['water_mask'][y, x]:
            # 检查是否深入水域
            water_neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and
                            self.hsv_analysis['water_mask'][ny, nx]):
                        water_neighbors += 1

            if water_neighbors > 6:  # 深入水域
                reward -= 80.0
            else:
                reward -= 40.0  # 轻微水域渗透

        # 动作约束奖励
        allowed_actions = self.action_constraints.get_allowed_actions(
            new_pos, self.current_coastline, self.hsv_analysis
        )
        if action_idx in allowed_actions:
            reward += 5.0  # 符合约束的动作
        else:
            reward -= 10.0  # 违反约束的动作

        # 连通性奖励
        connectivity_reward = self._calculate_connectivity_reward(new_pos)
        reward += connectivity_reward * 10.0

        # 边缘区域强烈惩罚 - 新增
        edge_penalty = self._calculate_edge_region_penalty(y)
        reward += edge_penalty

        return reward

    def _calculate_region_position_bonus(self, y):
        """计算区域位置奖励 - 中间区域高奖励"""
        height = self.height

        # 定义中间核心区域 (中间40%)
        core_start = int(height * 0.3)
        core_end = int(height * 0.7)

        # 定义过渡区域
        transition_start = int(height * 0.25)
        transition_end = int(height * 0.75)

        if core_start <= y <= core_end:
            # 核心中间区域：最高奖励
            return 25.0
        elif transition_start <= y <= transition_end:
            # 过渡区域：中等奖励
            return 10.0
        elif height // 6 <= y <= 5 * height // 6:
            # 外围可接受区域：低奖励
            return 2.0
        else:
            # 边缘区域：无奖励
            return 0.0

    def _calculate_edge_region_penalty(self, y):
        """计算边缘区域惩罚"""
        height = self.height

        # 上边缘惩罚
        if y < height // 4:
            distance_from_top = y
            penalty = -50.0 * (1.0 - distance_from_top / (height // 4))
            return penalty

        # 下边缘惩罚
        elif y > 3 * height // 4:
            distance_from_bottom = height - 1 - y
            penalty = -50.0 * (1.0 - distance_from_bottom / (height // 4))
            return penalty

        return 0.0

    def _calculate_hsv_reward(self, position):
        """计算HSV监督奖励"""
        y, x = position

        # HSV引导奖励
        guidance_score = self.hsv_analysis['coastline_guidance'][y, x]

        # 过渡区域奖励
        transition_score = self.hsv_analysis['transition_strength'][y, x]

        # 边界位置奖励
        water_mask = self.hsv_analysis['water_mask']
        land_mask = self.hsv_analysis['land_mask']

        near_water = False
        near_land = False

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if water_mask[ny, nx]:
                        near_water = True
                    if land_mask[ny, nx]:
                        near_land = True

        boundary_bonus = 0.0
        if near_water and near_land:
            boundary_bonus = 1.0  # 最佳位置：水陆边界
        elif near_water or near_land:
            boundary_bonus = 0.5  # 次佳位置：接近边界

        return guidance_score + transition_score + boundary_bonus

    def _calculate_connectivity_reward(self, position):
        """计算连通性奖励"""
        y, x = position

        # 检查周围海岸线连接性
        coastline_neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.current_coastline[ny, nx] > 0.3:
                        coastline_neighbors += 1

        # 理想的连接性：2个邻居（线性连接）
        if coastline_neighbors == 2:
            return 3.0
        elif coastline_neighbors == 1:
            return 2.0  # 延续现有路径
        elif coastline_neighbors == 3:
            return 1.0  # 可接受的分支
        elif coastline_neighbors >= 4:
            return -1.0  # 过度分支

        return 0.0

    def update_coastline(self, position, value=1.0):
        """更新海岸线"""
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            self.current_coastline[y, x] = min(1.0, self.current_coastline[y, x] + value)


# ==================== 约束的代理 ====================

class ConstrainedCoastlineAgent:
    """约束的海岸线代理 - HSV监督 + 分支约束"""

    def __init__(self, env, lr=1e-4, gamma=0.98, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay=0.995):
        self.env = env
        self.device = device

        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 网络
        self.policy_net = ConstrainedCoastlineDQN().to(device)
        self.target_net = ConstrainedCoastlineDQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)

        # 经验回放
        self.memory = deque(maxlen=15000)

        # 训练参数
        self.batch_size = 32
        self.target_update_freq = 100
        self.train_freq = 4
        self.steps_done = 0

        print(f"✅ 约束DQN代理初始化完成")

    def select_action(self, rgb_state, hsv_state, position, enhanced_features, training=True):
        """选择约束的动作"""
        # 获取允许的动作
        allowed_actions = self.env.action_constraints.get_allowed_actions(
            position, self.env.current_coastline, self.env.hsv_analysis
        )

        if training and random.random() < self.epsilon:
            return random.choice(allowed_actions)
        else:
            with torch.no_grad():
                position_tensor = torch.LongTensor([position]).to(device)
                q_values = self.policy_net(rgb_state, hsv_state, position_tensor, enhanced_features)

                # 只考虑允许的动作
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

        # 解包批次数据
        rgb_states = torch.cat([item[0][0] for item in batch])
        hsv_states = torch.cat([item[0][1] for item in batch])
        positions = torch.LongTensor([item[0][2] for item in batch]).to(device)
        enhanced_features = torch.cat([item[0][3] for item in batch])

        actions = torch.LongTensor([item[1] for item in batch]).to(device)
        rewards = torch.FloatTensor([item[3] for item in batch]).to(device)

        current_q_values = self.policy_net(rgb_states, hsv_states, positions, enhanced_features).gather(1,
                                                                                                        actions.unsqueeze(
                                                                                                            1))

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

        # Huber损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def optimize_constrained_coastline(self, max_episodes=200, max_steps_per_episode=400):
        """优化约束海岸线检测"""
        print("🎯 约束海岸线优化开始 - HSV监督 + 分支约束...")

        search_positions = np.where(self.env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return self.env.current_coastline

        # 基于HSV的智能起始点选择
        hsv_guided_starts = []
        for pos in candidate_positions[::3]:
            y, x = pos
            guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
            transition_score = self.env.hsv_analysis['transition_strength'][y, x]

            if guidance_score > 0.4 or transition_score > 0.5:
                hsv_guided_starts.append(pos)

        if not hsv_guided_starts:
            hsv_guided_starts = candidate_positions[:50]

        episode_rewards = []
        total_improvements = 0
        hsv_quality_scores = []

        for episode in range(max_episodes):
            # 智能起始点策略
            if episode < max_episodes // 4:
                # 前1/4：从HSV高质量点开始
                start_position = random.choice(hsv_guided_starts)
            elif episode < max_episodes // 2:
                # 2/4：从GT附近开始（如果有）
                if self.env.gt_analysis and random.random() < 0.8:
                    gt_positions = np.where(self.env.gt_analysis['gt_binary'])
                    if len(gt_positions[0]) > 0:
                        idx = random.randint(0, len(gt_positions[0]) - 1)
                        start_position = (gt_positions[0][idx], gt_positions[1][idx])
                    else:
                        start_position = random.choice(hsv_guided_starts)
                else:
                    start_position = random.choice(hsv_guided_starts)
            elif episode < 3 * max_episodes // 4:
                # 3/4：从连通性断点开始
                start_position = self._find_connectivity_break_start(candidate_positions)
                if start_position is None:
                    start_position = random.choice(hsv_guided_starts)
            else:
                # 后1/4：随机探索剩余区域
                start_position = random.choice(candidate_positions)

            current_position = start_position
            episode_reward = 0
            episode_improvements = 0

            for step in range(max_steps_per_episode):
                # 获取状态
                rgb_state, hsv_state = self.env.get_state_tensor(current_position)
                enhanced_features = self.env.get_enhanced_features(current_position)

                action = self.select_action(rgb_state, hsv_state, current_position,
                                            enhanced_features, training=True)

                next_position, reward = self.env.step(current_position, action)
                episode_reward += reward

                # 获取下一状态
                next_rgb_state, next_hsv_state = self.env.get_state_tensor(next_position)
                next_enhanced_features = self.env.get_enhanced_features(next_position)

                # 存储经验
                current_state = (rgb_state, hsv_state, current_position, enhanced_features)
                next_state = (next_rgb_state, next_hsv_state, next_position,
                              next_enhanced_features) if reward > -30 else None

                self.memory.append((current_state, action, next_state, reward))

                # 自适应海岸线更新
                if reward > 20.0:  # 高质量检测
                    self.env.update_coastline(next_position, 0.9)
                    episode_improvements += 1
                    total_improvements += 1
                elif reward > 10.0:  # 中等质量检测
                    self.env.update_coastline(next_position, 0.6)
                    episode_improvements += 1
                elif reward > 5.0:  # 低质量但可接受
                    self.env.update_coastline(next_position, 0.3)

                # 训练
                if self.steps_done % self.train_freq == 0:
                    loss = self.train_step()

                # 更新目标网络
                if self.steps_done % self.target_update_freq == 0:
                    self.update_target_network()

                self.steps_done += 1
                current_position = next_position

                # 早停条件
                if reward < -40:
                    break

            episode_rewards.append(episode_reward)
            self.decay_epsilon()

            # HSV质量评估
            if episode % 20 == 0:
                hsv_quality = self.env.hsv_supervisor.evaluate_prediction_quality(
                    self.env.current_coastline,
                    self.env.gt_analysis['gt_binary'] if self.env.gt_analysis else None,
                    self.env.hsv_analysis
                )
                hsv_quality_scores.append(hsv_quality)

                avg_reward = np.mean(episode_rewards[-20:])
                current_pixels = np.sum(self.env.current_coastline > 0.3)

                print(f"   Episode {episode:3d}: 平均奖励={avg_reward:6.2f}, ε={self.epsilon:.3f}, "
                      f"海岸线像素={current_pixels:,}, HSV质量={hsv_quality:.3f}, 本轮改进={episode_improvements}")

        final_pixels = np.sum(self.env.current_coastline > 0.3)
        final_hsv_quality = self.env.hsv_supervisor.evaluate_prediction_quality(
            self.env.current_coastline,
            self.env.gt_analysis['gt_binary'] if self.env.gt_analysis else None,
            self.env.hsv_analysis
        )

        print(f"   ✅ 约束优化完成")
        print(f"   总改进次数: {total_improvements}")
        print(f"   最终海岸线像素: {final_pixels:,}")
        print(f"   最终HSV质量得分: {final_hsv_quality:.3f}")

        return self.env.current_coastline

    def _find_connectivity_break_start(self, candidate_positions):
        """寻找连通性断点的起始位置"""
        current_coastline = self.env.current_coastline > 0.3
        labeled_array, num_components = label(current_coastline)

        if num_components <= 1:
            return None

        # 寻找组件间的潜在连接点
        connection_candidates = []

        for pos in candidate_positions[::8]:  # 采样
            y, x = pos
            if not current_coastline[y, x]:  # 不在现有海岸线上

                # 检查HSV引导
                guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
                if guidance_score < 0.3:
                    continue

                # 检查周围的组件
                nearby_components = set()
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.env.height and 0 <= nx < self.env.width and
                                labeled_array[ny, nx] > 0):
                            nearby_components.add(labeled_array[ny, nx])

                if len(nearby_components) >= 2:
                    # 计算连接价值
                    connection_value = guidance_score + len(nearby_components) * 0.1
                    connection_candidates.append((pos, connection_value))

        if connection_candidates:
            # 选择最有价值的连接点
            connection_candidates.sort(key=lambda x: x[1], reverse=True)
            return connection_candidates[0][0]

        return None

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ==================== 后处理器 ====================

class ConstrainedPostProcessor:
    """约束的后处理器 - 防止过度分支"""

    def __init__(self):
        print("✅ 约束后处理器初始化完成")

    def process_constrained_coastline(self, coastline, hsv_analysis):
        """约束的海岸线后处理"""
        print("🔧 开始约束后处理...")

        # 第一步：HSV引导的智能二值化
        binary_coastline = self._hsv_guided_binarization(coastline, hsv_analysis)

        # 第二步：分支约束处理
        constrained_coastline = self._apply_branch_constraints(binary_coastline, hsv_analysis)

        # 第三步：水域渗透修复
        cleaned_coastline = self._remove_water_penetration(constrained_coastline, hsv_analysis)

        # 第四步：连通性优化
        connected_coastline = self._optimize_connectivity(cleaned_coastline, hsv_analysis)

        # 第五步：最终平滑
        final_coastline = self._final_smoothing(connected_coastline)

        return final_coastline.astype(float)

    def _hsv_guided_binarization(self, coastline, hsv_analysis):
        """HSV引导的二值化"""
        # 结合HSV引导和过渡强度的自适应阈值
        guidance_weight = hsv_analysis['coastline_guidance']
        transition_weight = hsv_analysis['transition_strength']

        # 加权海岸线
        weighted_coastline = coastline * (1 + guidance_weight + transition_weight)

        # 自适应阈值
        valid_mask = weighted_coastline > 0
        if np.any(valid_mask):
            threshold = np.percentile(weighted_coastline[valid_mask], 75)
        else:
            threshold = 0.5

        binary_result = weighted_coastline > threshold

        # 移除孤立噪点
        binary_result = self._remove_small_components(binary_result, min_size=5)

        return binary_result

    def _apply_branch_constraints(self, binary_coastline, hsv_analysis):
        """应用分支约束"""
        result = binary_coastline.copy()

        # 检测过度分支
        over_branched_points = self._detect_over_branching(binary_coastline)

        # 移除不合理的分支
        for point in over_branched_points:
            y, x = point

            # 检查是否在水域内（如果是，优先移除）
            if hsv_analysis['water_mask'][y, x]:
                result[y, x] = False
                continue

            # 检查分支质量
            branch_quality = self._evaluate_branch_quality(point, binary_coastline, hsv_analysis)
            if branch_quality < 0.3:
                result[y, x] = False

        return result

    def _detect_over_branching(self, binary_coastline):
        """检测过度分支点"""
        over_branched = []

        for y in range(1, binary_coastline.shape[0] - 1):
            for x in range(1, binary_coastline.shape[1] - 1):
                if binary_coastline[y, x]:
                    # 计算连接的分支数
                    neighbors = binary_coastline[y - 1:y + 2, x - 1:x + 2].astype(int)
                    neighbors[1, 1] = 0  # 排除自己

                    # 使用8连通性分析
                    labeled_neighbors, num_branches = label(neighbors)

                    # 如果分支数超过3个，认为是过度分支
                    if num_branches > 3:
                        over_branched.append((y, x))

                    # 检查是否形成密集团块
                    neighbor_count = np.sum(neighbors)
                    if neighbor_count > 6:  # 8邻域中超过6个都是海岸线
                        over_branched.append((y, x))

        return over_branched

    def _evaluate_branch_quality(self, point, binary_coastline, hsv_analysis):
        """评估分支质量"""
        y, x = point

        # HSV支持度
        guidance_score = hsv_analysis['coastline_guidance'][y, x]
        transition_score = hsv_analysis['transition_strength'][y, x]

        # 水域惩罚
        water_penalty = 1.0 if hsv_analysis['water_mask'][y, x] else 0.0

        # 局部连续性
        continuity_score = self._calculate_local_continuity(point, binary_coastline)

        quality = (guidance_score * 0.4 + transition_score * 0.3 +
                   continuity_score * 0.3 - water_penalty * 0.8)

        return max(0.0, quality)

    def _calculate_local_continuity(self, point, binary_coastline):
        """计算局部连续性"""
        y, x = point

        # 检查是否形成合理的线性连接
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        connected_directions = []
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if (0 <= ny < binary_coastline.shape[0] and
                    0 <= nx < binary_coastline.shape[1] and
                    binary_coastline[ny, nx]):
                connected_directions.append((dy, dx))

        # 理想情况：2个方向连接（形成线）
        if len(connected_directions) == 2:
            # 检查是否形成直线或平滑曲线
            dir1, dir2 = connected_directions
            if (dir1[0] + dir2[0] == 0 and dir1[1] + dir2[1] == 0):  # 直线
                return 1.0
            else:  # 曲线
                return 0.8
        elif len(connected_directions) == 1:
            return 0.6  # 端点
        elif len(connected_directions) == 3:
            return 0.4  # 轻微分支
        else:
            return 0.2  # 过度分支或孤立点

    def _remove_water_penetration(self, binary_coastline, hsv_analysis):
        """移除水域渗透"""
        result = binary_coastline.copy()
        water_mask = hsv_analysis['water_mask']

        # 检测深入水域的点
        water_penetration_points = []

        for y in range(binary_coastline.shape[0]):
            for x in range(binary_coastline.shape[1]):
                if binary_coastline[y, x] and water_mask[y, x]:
                    # 检查周围是否大部分都是水域
                    water_neighbors = 0
                    total_neighbors = 0

                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < water_mask.shape[0] and 0 <= nx < water_mask.shape[1]:
                                total_neighbors += 1
                                if water_mask[ny, nx]:
                                    water_neighbors += 1

                    # 如果周围大部分是水域，认为是渗透
                    if water_neighbors / total_neighbors > 0.7:
                        water_penetration_points.append((y, x))

        # 移除水域渗透点
        for y, x in water_penetration_points:
            result[y, x] = False

        print(f"   移除了 {len(water_penetration_points)} 个水域渗透点")

        return result

    def _optimize_connectivity(self, binary_coastline, hsv_analysis):
        """优化连通性"""
        result = binary_coastline.copy()

        # 连接近距离的组件
        labeled_array, num_components = label(binary_coastline)

        if num_components <= 1:
            return result

        print(f"   连接 {num_components} 个组件...")

        # 找到组件间的最佳连接
        for i in range(1, min(num_components + 1, 10)):  # 限制组件数
            for j in range(i + 1, min(num_components + 1, 10)):
                connection_path = self._find_hsv_guided_connection(
                    labeled_array, i, j, hsv_analysis
                )
                if connection_path:
                    for y, x in connection_path:
                        result[y, x] = True

        return result

    def _find_hsv_guided_connection(self, labeled_array, comp1_id, comp2_id, hsv_analysis):
        """寻找HSV引导的连接路径"""
        comp1_coords = np.where(labeled_array == comp1_id)
        comp2_coords = np.where(labeled_array == comp2_id)

        if len(comp1_coords[0]) == 0 or len(comp2_coords[0]) == 0:
            return None

        # 寻找最佳连接点对
        best_path = None
        best_score = -1

        # 采样减少计算
        sample1 = list(zip(comp1_coords[0][::max(1, len(comp1_coords[0]) // 3)],
                           comp1_coords[1][::max(1, len(comp1_coords[1]) // 3)]))
        sample2 = list(zip(comp2_coords[0][::max(1, len(comp2_coords[0]) // 3)],
                           comp2_coords[1][::max(1, len(comp2_coords[1]) // 3)]))

        for p1 in sample1[:3]:
            for p2 in sample2[:3]:
                distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if distance > 30:  # 距离太远不连接
                    continue

                path = self._generate_hsv_guided_path(p1, p2, hsv_analysis)
                if path:
                    score = self._evaluate_path_quality(path, hsv_analysis)
                    if score > best_score:
                        best_score = score
                        best_path = path

        return best_path if best_score > 0.3 else None

    def _generate_hsv_guided_path(self, p1, p2, hsv_analysis):
        """生成HSV引导的路径"""
        path = []

        # 简单的直线路径（后续可以改进为A*算法）
        x1, y1 = p1[1], p1[0]
        x2, y2 = p2[1], p2[0]

        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return [(p1[0], p1[1])]

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if (0 <= y < hsv_analysis['coastline_guidance'].shape[0] and
                    0 <= x < hsv_analysis['coastline_guidance'].shape[1]):
                path.append((y, x))

        return path

    def _evaluate_path_quality(self, path, hsv_analysis):
        """评估路径质量"""
        if not path:
            return 0.0

        total_guidance = 0
        total_transition = 0
        water_penalty = 0

        for y, x in path:
            total_guidance += hsv_analysis['coastline_guidance'][y, x]
            total_transition += hsv_analysis['transition_strength'][y, x]
            if hsv_analysis['water_mask'][y, x]:
                water_penalty += 1

        avg_guidance = total_guidance / len(path)
        avg_transition = total_transition / len(path)
        water_ratio = water_penalty / len(path)

        quality = avg_guidance * 0.5 + avg_transition * 0.3 - water_ratio * 0.4
        return max(0.0, quality)

    def _final_smoothing(self, binary_coastline):
        """最终平滑处理"""
        # 轻微的形态学处理
        kernel = np.ones((3, 3), dtype=bool)

        # 闭操作填充小间隙
        smoothed = binary_closing(binary_coastline, kernel, iterations=1)

        # 轻微腐蚀去除毛刺
        smoothed = binary_erosion(smoothed, kernel, iterations=1)

        # 轻微膨胀恢复
        smoothed = binary_dilation(smoothed, kernel, iterations=1)

        return smoothed

    def _remove_small_components(self, binary_image, min_size=10):
        """移除小组件"""
        labeled_array, num_components = label(binary_image)

        # 计算每个组件的大小
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled_array == i)
            component_sizes.append(size)

        # 移除小组件
        result = binary_image.copy()
        for i, size in enumerate(component_sizes, 1):
            if size < min_size:
                result[labeled_array == i] = False

        return result


# ==================== 主检测器 ====================

class ConstrainedCoastlineDetector:
    """约束的海岸线检测器"""

    def __init__(self):
        self.gt_analyzer = GroundTruthAnalyzer()
        self.post_processor = ConstrainedPostProcessor()
        print("✅ 约束海岸线检测系统初始化完成")
        print("   🎯 主要特色：HSV监督 + 分支约束 + 好奇心探索")
        print("   📦 防止海域渗透，主横向分支，极限纵向扩展")

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
        """处理单个图像（约束版）"""
        print(f"\n🌊 约束海岸线检测处理: {os.path.basename(image_path)}")

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
            print("\n📍 步骤2: 创建约束环境（HSV监督 + 分支限制）")
            constrained_env = ConstrainedCoastlineEnvironment(processed_img, gt_analysis)

            # 步骤3: 约束DQN训练
            print("\n📍 步骤3: 约束DQN学习（防海域渗透 + 好奇心探索）")
            constrained_agent = ConstrainedCoastlineAgent(constrained_env)

            optimized_coastline = constrained_agent.optimize_constrained_coastline(
                max_episodes=200,
                max_steps_per_episode=400
            )

            # 步骤4: 约束后处理
            print("\n📍 步骤4: 约束后处理（分支限制 + 水域清理）")
            final_coastline = self.post_processor.process_constrained_coastline(
                optimized_coastline, constrained_env.hsv_analysis
            )

            # 质量评估
            quality_metrics = self._evaluate_constrained_quality(final_coastline, gt_coastline,
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

    def _evaluate_constrained_quality(self, predicted, ground_truth, hsv_analysis):
        """评估约束质量"""
        metrics = {}

        pred_binary = (predicted > 0.5).astype(bool)
        coastline_pixels = np.sum(pred_binary)

        metrics['coastline_pixels'] = int(coastline_pixels)

        # 连通性分析
        labeled_array, num_components = label(pred_binary)
        metrics['num_components'] = int(num_components)

        # HSV质量评估
        hsv_supervisor = HSVAttentionSupervisor()
        hsv_quality = hsv_supervisor.evaluate_prediction_quality(
            predicted, ground_truth, hsv_analysis
        )
        metrics['hsv_quality'] = float(hsv_quality)

        # 水域渗透检查
        water_mask = hsv_analysis['water_mask']
        water_penetration = np.sum(pred_binary & water_mask) / (coastline_pixels + 1e-8)
        metrics['water_penetration'] = float(water_penetration)

        # 分支控制评估
        branch_score = self._evaluate_branch_control(pred_binary)
        metrics['branch_control'] = float(branch_score)

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

            # 综合质量得分
            overall_score = (f1_score * 0.25 + iou * 0.25 + hsv_quality * 0.2 +
                             branch_score * 0.15 + (1 - water_penetration) * 0.15)
        else:
            # 无GT时的评分
            density_score = min(1.0, coastline_pixels / 2000.0)
            overall_score = (hsv_quality * 0.4 + branch_score * 0.3 +
                             (1 - water_penetration) * 0.2 + density_score * 0.1)

        metrics['overall_score'] = float(overall_score)

        return metrics

    def _evaluate_branch_control(self, binary_coastline):
        """评估分支控制质量"""
        if not np.any(binary_coastline):
            return 0.0

        # 计算过度分支点
        over_branched_count = 0
        total_points = np.sum(binary_coastline)

        for y in range(1, binary_coastline.shape[0] - 1):
            for x in range(1, binary_coastline.shape[1] - 1):
                if binary_coastline[y, x]:
                    neighbors = np.sum(binary_coastline[y - 1:y + 2, x - 1:x + 2]) - 1
                    if neighbors > 4:  # 过度连接
                        over_branched_count += 1

        branch_control_score = 1.0 - (over_branched_count / total_points)
        return max(0.0, branch_control_score)


# ==================== 可视化函数 ====================

def create_constrained_visualization(result, save_path):
    """创建约束版可视化"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Constrained Coastline Detection with HSV Supervision - {result.get("sample_id", "Unknown")}',
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

    # HSV引导图
    if 'hsv_analysis' in result:
        axes[0, 3].imshow(result['hsv_analysis']['coastline_guidance'], cmap='plasma')
        axes[0, 3].set_title('HSV Coastline Guidance')
        axes[0, 3].axis('off')
    else:
        axes[0, 3].axis('off')
        axes[0, 3].set_title('HSV Guidance\n(Not Available)')

    # 第二行：检测结果
    axes[1, 0].imshow(result['optimized_coastline'], cmap='hot')
    opt_pixels = np.sum(result['optimized_coastline'] > 0.3)
    axes[1, 0].set_title(f'Constrained DQN Detection\n({opt_pixels:,} pixels)',
                         color='blue', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['final_coastline'], cmap='hot')
    final_pixels = np.sum(result['final_coastline'] > 0.5)
    axes[1, 1].set_title(f'Final Constrained Result\n({final_pixels:,} pixels)',
                         color='red', fontweight='bold')
    axes[1, 1].axis('off')

    # HSV监督对比
    if 'hsv_analysis' in result:
        water_mask = result['hsv_analysis']['water_mask']
        pred_binary = (result['final_coastline'] > 0.5).astype(bool)

        # 水域渗透可视化
        penetration_vis = np.zeros((*result['final_coastline'].shape, 3))
        penetration_vis[:, :, 0] = result['final_coastline']  # 预测结果（红色）
        penetration_vis[:, :, 1] = water_mask.astype(float) * 0.5  # 水域（绿色）

        # 标记水域渗透（紫色）
        water_penetration = pred_binary & water_mask
        penetration_vis[:, :, 2] = water_penetration.astype(float)

        axes[1, 2].imshow(penetration_vis)
        penetration_pixels = np.sum(water_penetration)
        axes[1, 2].set_title(f'Water Penetration Check\n({penetration_pixels:,} penetrated pixels)')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Water Penetration\n(Not Available)')

    # 连通性分析
    labeled_array, num_components = label(result['final_coastline'] > 0.5)
    axes[1, 3].imshow(labeled_array, cmap='tab20')
    axes[1, 3].set_title(f'Connectivity Analysis\n({num_components} components)')
    axes[1, 3].axis('off')

    # 第三行：分析
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')

    # 统计信息
    metrics = result['quality_metrics']
    stats_text = f"""Constrained Coastline Detection Results:

Overall Score: {metrics['overall_score']:.3f}
HSV Quality: {metrics.get('hsv_quality', 0):.3f}
Status: {"✅ SUCCESS" if result['success'] else "❌ FAILED"}

Quality Analysis:
• Final pixels: {metrics['coastline_pixels']:,}
• Components: {metrics['num_components']}
• Water penetration: {metrics.get('water_penetration', 0):.1%}
• Branch control: {metrics.get('branch_control', 0):.3f}"""

    if 'f1_score' in metrics:
        stats_text += f"""

GT Matching Metrics:
• Precision: {metrics['precision']:.3f}
• Recall: {metrics['recall']:.3f}
• F1-Score: {metrics['f1_score']:.3f}
• IoU: {metrics['iou']:.3f}"""

    stats_text += f"""

Constraint Features:
✓ HSV attention supervision
✓ Horizontal primary branching
✓ Limited vertical expansion  
✓ Water penetration prevention
✓ Curiosity-driven exploration
✓ Branch over-growth control
✓ Adaptive action masking

Technical Improvements:
• HSV-guided search regions
• Action constraint system
• Water boundary detection
• Connectivity gap repair
• Multi-component connection
• Enhanced reward system
• 25-dimensional features
• Device: {device}

HSV Analysis Summary:
• Water regions detected
• Land regions identified
• Coastline guidance computed
• Transition strength analyzed
• Boundary-aware exploration"""

    axes[2, 0].text(0.02, 0.98, stats_text, transform=fig.transFigure,
                    fontsize=8, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 约束版可视化已保存: {save_path}")


# ==================== 演示函数 ====================

def create_constrained_demo_image():
    """创建约束演示海岸线图像"""
    print("🎨 创建约束演示海岸线图像...")

    # 创建一个400x400的演示图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 背景 - 蓝色水体（更明确的蓝色）
    img[:, :] = [20, 100, 200]

    # 创建主要横向海岸线
    for y in range(400):
        # 主海岸线 - 主要是横向的，减少纵向变化
        main_coastline_x = int(180 + 40 * np.sin(y * 0.02) + 20 * np.sin(y * 0.08))
        main_coastline_x = max(50, min(350, main_coastline_x))

        # 陆地部分 - 更明确的绿色
        img[y, main_coastline_x:] = [100, 180, 50]

        # 海岸线过渡带
        for offset in range(-5, 6):
            x = main_coastline_x + offset
            if 0 <= x < 400:
                # 创建明确的过渡色
                mix_ratio = (5 - abs(offset)) / 5.0
                img[y, x] = [
                    int(20 + (100 - 20) * mix_ratio),
                    int(100 + (180 - 100) * mix_ratio),
                    int(200 + (50 - 200) * mix_ratio)
                ]

    # 添加小岛 - 测试连通性
    island_center = (150, 100)
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            y, x = island_center[0] + dy, island_center[1] + dx
            if 0 <= y < 400 and 0 <= x < 400:
                distance = math.sqrt(dy * dy + dx * dx)
                if distance <= 12:
                    img[y, x] = [100, 180, 50]
                elif distance <= 15:
                    # 岛屿海岸线
                    mix_ratio = (15 - distance) / 3.0
                    img[y, x] = [
                        int(20 + (100 - 20) * mix_ratio),
                        int(100 + (180 - 100) * mix_ratio),
                        int(200 + (50 - 200) * mix_ratio)
                    ]

    # 创建对应的GT - 主要横向
    gt = np.zeros((400, 400), dtype=np.uint8)

    # 主海岸线GT - 横向为主
    for y in range(400):
        main_coastline_x = int(180 + 40 * np.sin(y * 0.02) + 20 * np.sin(y * 0.08))
        main_coastline_x = max(50, min(350, main_coastline_x))

        # 海岸线宽度较窄
        for offset in range(-2, 3):
            x = main_coastline_x + offset
            if 0 <= x < 400:
                gt[y, x] = 255

    # 小岛GT
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            y, x = island_center[0] + dy, island_center[1] + dx
            if 0 <= y < 400 and 0 <= x < 400:
                distance = math.sqrt(dy * dy + dx * dx)
                if 11 <= distance <= 13:
                    gt[y, x] = 255

    return img, gt


# ==================== 主函数 ====================

def main():
    """主函数（约束版）"""
    print("🚀 启动约束海岸线检测系统...")
    print("🎯 主要特色：HSV监督 + 分支约束 + 水域渗透防护")

    detector = ConstrainedCoastlineDetector()

    # 设置路径
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

            # 寻找对应的GT文件 - 改进匹配逻辑
            gt_path = None
            if os.path.exists(ground_truth_dir):
                gt_files = [f for f in os.listdir(ground_truth_dir) if
                            f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

                print(f"   📁 初始文件: {test_file}")
                print(f"   📁 GT目录中的文件: {gt_files}")

                # 尝试多种匹配策略
                base_name = os.path.splitext(test_file)[0]

                # 策略1: 直接名称匹配
                for gt_file in gt_files:
                    gt_base = os.path.splitext(gt_file)[0]
                    if base_name == gt_base:
                        gt_path = os.path.join(ground_truth_dir, gt_file)
                        print(f"   ✅ 找到GT文件 (直接匹配): {gt_file}")
                        break

                # 策略2: 包含匹配
                if gt_path is None:
                    for gt_file in gt_files:
                        if base_name in gt_file or gt_file.replace('.pdf', '').replace('ground_', '') in base_name:
                            gt_path = os.path.join(ground_truth_dir, gt_file)
                            print(f"   ✅ 找到GT文件 (包含匹配): {gt_file}")
                            break

                # 策略3: 年份匹配 (针对ground_2017.pdf格式)
                if gt_path is None:
                    # 从初始文件名中提取可能的年份
                    import re
                    year_match = re.search(r'20\d{2}', base_name)
                    if year_match:
                        year = year_match.group()
                        gt_candidate = f"ground_{year}.pdf"
                        if gt_candidate in gt_files:
                            gt_path = os.path.join(ground_truth_dir, gt_candidate)
                            print(f"   ✅ 找到GT文件 (年份匹配): {gt_candidate}")

                # 策略4: 如果还没找到，选择第一个GT文件
                if gt_path is None and gt_files:
                    gt_path = os.path.join(ground_truth_dir, gt_files[0])
                    print(f"   ⚠️ 使用第一个GT文件: {gt_files[0]}")

                if gt_path is None:
                    print(f"   ❌ 未找到匹配的GT文件")

            print(f"\n🧪 测试处理: {test_file}")
            if gt_path:
                print(f"   📍 使用GT文件: {os.path.basename(gt_path)}")

            result = detector.process_image(initial_path, gt_path)

            if result:
                result['sample_id'] = 'constrained_real_data'

    # 如果没有真实数据或处理失败，使用约束演示数据
    if result is None:
        print("\n🎨 使用约束演示数据测试系统...")

        # 创建约束演示图像
        demo_img, demo_gt = create_constrained_demo_image()

        # 保存临时文件
        os.makedirs("./temp", exist_ok=True)
        demo_img_path = "./temp/demo_image_constrained.png"
        demo_gt_path = "./temp/demo_gt_constrained.png"

        Image.fromarray(demo_img).save(demo_img_path)
        Image.fromarray(demo_gt).save(demo_gt_path)

        print(f"   ✅ 约束演示图像已创建: {demo_img_path}")

        # 处理演示图像
        result = detector.process_image(demo_img_path, demo_gt_path)

        if result:
            result['sample_id'] = 'constrained_demo'

    # 显示结果
    if result:
        # 保存结果
        output_dir = "./constrained_coastline_results"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'constrained_coastline_detection.png')
        create_constrained_visualization(result, save_path)

        # 显示结果
        metrics = result['quality_metrics']
        print(f"\n✅ 约束版处理完成!")
        print(f"   综合得分: {metrics['overall_score']:.3f}")
        print(f"   HSV质量得分: {metrics.get('hsv_quality', 0):.3f}")
        print(f"   海岸线像素: {metrics['coastline_pixels']:,}")
        print(f"   连通组件数: {metrics['num_components']}")
        print(f"   水域渗透率: {metrics.get('water_penetration', 0):.1%}")
        print(f"   分支控制得分: {metrics.get('branch_control', 0):.3f}")

        if 'f1_score' in metrics:
            print(f"   GT匹配F1: {metrics['f1_score']:.3f}")
            print(f"   GT匹配IoU: {metrics['iou']:.3f}")

        print(f"\n🎉 约束版特色:")
        print(f"   ✅ HSV注意力监督")
        print(f"   ✅ 横向主干分支")
        print(f"   ✅ 纵向极限扩展")
        print(f"   ✅ 水域渗透防护")
        print(f"   ✅ 好奇心驱动探索")
        print(f"   ✅ 智能动作约束")
        print(f"   📊 可视化结果: {save_path}")

        # 性能提升分析
        if metrics.get('water_penetration', 1.0) < 0.1:
            print(f"\n🚫 水域渗透控制优秀! (<10%)")
        if metrics.get('branch_control', 0) > 0.8:
            print(f"🌿 分支控制良好!")
        if metrics.get('hsv_quality', 0) > 0.6:
            print(f"🎨 HSV监督效果显著!")

    else:
        print("❌ 所有处理尝试都失败了")


if __name__ == "__main__":
    main()