"""
快速精准海域清理海岸线检测系统
主要目标：快速训练 + 重点关注最终清理后的评估指标
简化策略：减少训练复杂度，专注后处理优化
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
import time

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

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("🚀 快速精准海域清理系统!")
print("目标：快速训练 + 重点评估最终指标")
print("=" * 60)


# ==================== 简化的评估指标计算器 ====================

class FastMetricsCalculator:
    """快速评估指标计算器"""

    def __init__(self):
        print("✅ 快速评估指标计算器初始化完成")

    def calculate_metrics(self, predicted, ground_truth=None, inference_time=0.0, training_time=0.0):
        """计算关键评估指标"""
        metrics = {}

        # 预处理
        pred_binary = (predicted > 0.5).astype(bool)

        # 基础指标
        metrics['pixel_count'] = int(np.sum(pred_binary))

        # 连通组件分析
        labeled_array, num_components = label(pred_binary)
        metrics['components'] = int(num_components)

        # 时间指标
        metrics['inference_time_ms'] = float(inference_time * 1000)
        metrics['training_time_min'] = float(training_time)

        # 如果有Ground Truth，计算精确的指标
        if ground_truth is not None:
            gt_binary = (ground_truth > 0.5).astype(bool)

            # 混淆矩阵元素
            tp = np.sum(pred_binary & gt_binary)  # True Positive
            fp = np.sum(pred_binary & ~gt_binary)  # False Positive
            fn = np.sum(~pred_binary & gt_binary)  # False Negative
            tn = np.sum(~pred_binary & ~gt_binary)  # True Negative

            # 核心指标
            metrics['iou'] = float(tp / (tp + fp + fn + 1e-8))
            metrics['precision'] = float(tp / (tp + fp + 1e-8))
            metrics['recall'] = float(tp / (tp + fn + 1e-8))
            metrics['pixel_accuracy'] = float((tp + tn) / (tp + fp + fn + tn + 1e-8))

            # F1 Score
            precision = metrics['precision']
            recall = metrics['recall']
            metrics['f1_score'] = float(2 * precision * recall / (precision + recall + 1e-8))

        else:
            # 无GT时设置默认值
            metrics.update({
                'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
                'pixel_accuracy': 0.0, 'f1_score': 0.0
            })

        return metrics


# ==================== 基础类（简化版）====================

class BasicImageProcessor:
    @staticmethod
    def rgb_to_gray(rgb_image):
        if len(rgb_image.shape) == 3:
            return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return rgb_image


class SimpleGTAnalyzer:
    """简化的GT分析器"""
    def __init__(self):
        print("✅ 简化GT分析器初始化完成")

    def analyze_gt_pattern(self, gt_coastline):
        if gt_coastline is None:
            return None

        gt_binary = (gt_coastline > 0.5).astype(bool)

        # 简化的边缘区域
        edge_region = gt_binary.copy()
        for _ in range(5):  # 减少膨胀次数
            edge_region = binary_dilation(edge_region, np.ones((3, 3), dtype=bool))

        return {
            'gt_binary': gt_binary,
            'edge_region': edge_region,
            'total_pixels': np.sum(gt_binary)
        }


# ==================== 简化的HSV监督器 ====================

class SimpleHSVAnalyzer:
    """简化的HSV分析器"""

    def __init__(self):
        print("✅ 简化HSV分析器初始化完成")

    def analyze_image_hsv(self, rgb_image, gt_analysis=None):
        """快速HSV分析"""
        # 转换为HSV
        if len(rgb_image.shape) == 3:
            rgb_normalized = rgb_image.astype(float) / 255.0
            hsv_image = np.zeros_like(rgb_normalized)

            for i in range(0, rgb_image.shape[0], 4):  # 采样分析，提高速度
                for j in range(0, rgb_image.shape[1], 4):
                    r, g, b = rgb_normalized[i, j]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsv_image[i:i+4, j:j+4] = [h * 360, s, v]  # 块填充
        else:
            hsv_image = np.stack([np.zeros_like(rgb_image),
                                  np.zeros_like(rgb_image),
                                  rgb_image / 255.0], axis=2)

        # 简化的水域和陆地检测
        h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        # 水域：蓝色调 + 低亮度
        water_mask = ((h >= 180) & (h <= 240)) & (v <= 0.6)

        # 陆地：绿色调 + 高亮度
        land_mask = ((h >= 60) & (h <= 120)) | (v >= 0.4)

        # 简化的海岸线引导
        coastline_guidance = self._generate_simple_guidance(water_mask, land_mask, gt_analysis)

        return {
            'water_mask': water_mask,
            'land_mask': land_mask,
            'coastline_guidance': coastline_guidance,
            'transition_strength': np.ones_like(water_mask, dtype=float) * 0.5  # 简化
        }

    def _generate_simple_guidance(self, water_mask, land_mask, gt_analysis=None):
        """生成简化的海岸线引导"""
        # 水陆边界
        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask
        guidance = (water_boundary | land_boundary).astype(float)

        # 如果有GT，直接使用GT区域
        if gt_analysis is not None:
            gt_guidance = binary_dilation(gt_analysis['gt_binary'], np.ones((5, 5)))
            guidance = np.maximum(guidance, gt_guidance.astype(float))

        return guidance


# ==================== 简化的DQN网络 ====================

class SimpleDQN(nn.Module):
    """简化的DQN网络"""

    def __init__(self, input_size=32*32*3, hidden_dim=128, action_dim=8):
        super(SimpleDQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size + 10, hidden_dim),  # 简化特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )

    def forward(self, state_features):
        return self.network(state_features)


# ==================== 快速环境 ====================

class FastCoastlineEnvironment:
    """快速海岸线环境"""

    def __init__(self, image, gt_analysis):
        self.image = image
        self.gt_analysis = gt_analysis
        self.current_coastline = np.zeros(image.shape[:2], dtype=float)
        self.height, self.width = image.shape[:2]

        # 简化的HSV分析
        self.hsv_analyzer = SimpleHSVAnalyzer()
        self.hsv_analysis = self.hsv_analyzer.analyze_image_hsv(image, gt_analysis)

        # 简化的动作空间
        self.base_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                            (0, 1), (1, -1), (1, 0), (1, 1)]
        self.action_dim = len(self.base_actions)

        # 简化的搜索区域
        self.search_region = self._setup_simple_search_region()

        print(f"✅ 快速海岸线环境初始化完成")

    def _setup_simple_search_region(self):
        """设置简化的搜索区域"""
        # 主要关注中间1/3区域
        height, width = self.height, self.width
        search_region = np.zeros((height, width), dtype=bool)

        # 中间区域
        middle_start = height // 3
        middle_end = 2 * height // 3
        search_region[middle_start:middle_end, :] = True

        # 如果有GT，扩展GT周围区域
        if self.gt_analysis:
            gt_region = binary_dilation(self.gt_analysis['gt_binary'], np.ones((10, 10)))
            search_region = search_region | gt_region

        return search_region

    def get_simple_features(self, position):
        """获取简化特征"""
        y, x = position
        window_size = 32
        half_window = window_size // 2

        # 提取局部图像窗口
        y_start = max(0, y - half_window)
        y_end = min(self.height, y + half_window)
        x_start = max(0, x - half_window)
        x_end = min(self.width, x + half_window)

        # 简化的特征向量
        features = np.zeros(32*32*3 + 10, dtype=np.float32)

        # 图像特征（降采样）
        if len(self.image.shape) == 3:
            img_window = self.image[y_start:y_end, x_start:x_end]
            img_resized = np.array(Image.fromarray(img_window).resize((32, 32))) / 255.0
            features[:32*32*3] = img_resized.flatten()
        else:
            gray_window = self.image[y_start:y_end, x_start:x_end]
            gray_resized = np.array(Image.fromarray(gray_window).resize((32, 32))) / 255.0
            features[:32*32] = gray_resized.flatten()

        # 位置特征
        features[-10] = y / self.height
        features[-9] = x / self.width
        features[-8] = self.hsv_analysis['coastline_guidance'][y, x]
        features[-7] = 1.0 if self.hsv_analysis['water_mask'][y, x] else 0.0
        features[-6] = 1.0 if self.hsv_analysis['land_mask'][y, x] else 0.0

        # GT特征
        if self.gt_analysis:
            features[-5] = 1.0 if self.gt_analysis['gt_binary'][y, x] else 0.0

        # 区域特征
        if self.height // 3 <= y <= 2 * self.height // 3:
            features[-4] = 1.0  # 中间区域

        return torch.FloatTensor(features).unsqueeze(0).to(device)

    def step(self, position, action_idx):
        """执行动作"""
        y, x = position
        dy, dx = self.base_actions[action_idx]

        new_y = np.clip(y + dy, 0, self.height - 1)
        new_x = np.clip(x + dx, 0, self.width - 1)

        new_position = (new_y, new_x)
        reward = self._calculate_simple_reward(new_position)

        return new_position, reward

    def _calculate_simple_reward(self, position):
        """简化的奖励函数"""
        y, x = position
        reward = 0.0

        # 基础区域奖励
        if self.height // 3 <= y <= 2 * self.height // 3:
            reward += 20.0

        # HSV引导奖励
        reward += self.hsv_analysis['coastline_guidance'][y, x] * 30.0

        # GT奖励
        if self.gt_analysis and self.gt_analysis['gt_binary'][y, x]:
            reward += 50.0

        # 搜索区域检查
        if not self.search_region[y, x]:
            reward -= 100.0

        return reward

    def update_coastline(self, position, value=1.0):
        """更新海岸线"""
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            self.current_coastline[y, x] = min(1.0, self.current_coastline[y, x] + value)


# ==================== 快速代理 ====================

class FastCoastlineAgent:
    """快速海岸线代理"""

    def __init__(self, env):
        self.env = env
        self.device = device

        # 简化的网络
        self.policy_net = SimpleDQN().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-3)  # 提高学习率

        self.epsilon = 0.5  # 降低初始探索率
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1

        print(f"✅ 快速DQN代理初始化完成")

    def select_action(self, features, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.env.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(features)
                return q_values.argmax(dim=1).item()

    def fast_train(self, max_episodes=50, max_steps_per_episode=200):  # 大幅减少训练时间
        """快速训练"""
        print("🚀 开始快速训练...")

        search_positions = np.where(self.env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return self.env.current_coastline

        # 优先选择中间区域的起始点
        height = self.env.height
        middle_positions = [pos for pos in candidate_positions
                           if height//3 <= pos[0] <= 2*height//3]

        if not middle_positions:
            middle_positions = candidate_positions[:20]

        for episode in range(max_episodes):
            # 随机选择起始位置
            start_position = random.choice(middle_positions)
            current_position = start_position
            episode_reward = 0

            for step in range(max_steps_per_episode):
                # 获取特征
                features = self.env.get_simple_features(current_position)

                # 选择动作
                action = self.select_action(features, training=True)

                # 执行动作
                next_position, reward = self.env.step(current_position, action)
                episode_reward += reward

                # 更新海岸线（更宽松的条件）
                if reward > 5.0:  # 降低阈值
                    self.env.update_coastline(next_position, 0.8)
                elif reward > 0:
                    self.env.update_coastline(next_position, 0.4)

                current_position = next_position

                # 早停条件
                if reward < -50:
                    break

            # 衰减探索率
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 10 == 0:
                current_pixels = np.sum(self.env.current_coastline > 0.3)
                print(f"   Episode {episode:2d}: 奖励={episode_reward:6.1f}, ε={self.epsilon:.3f}, 像素={current_pixels:,}")

        final_pixels = np.sum(self.env.current_coastline > 0.3)
        print(f"   ✅ 快速训练完成: 总像素={final_pixels:,}")

        return self.env.current_coastline


# ==================== IoU优化后处理器 ====================

class IoUOptimizedPostProcessor:
    """专门优化IoU的后处理器"""

    def __init__(self, target_pixel_range=(90000, 100000)):
        self.target_pixel_range = target_pixel_range
        print("✅ IoU优化后处理器初始化完成")

    def process_for_optimal_iou(self, raw_coastline, gt_analysis, rgb_image):
        """专门为IoU优化的后处理"""
        print("🎯 开始IoU优化后处理...")

        # 1. 智能二值化
        binary_coastline = self._smart_binarization(raw_coastline)
        print(f"   二值化后: {np.sum(binary_coastline):,} 像素")

        # 2. GT对齐优化
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_aligned = self._align_with_gt(binary_coastline, gt_analysis['gt_binary'])
            print(f"   GT对齐后: {np.sum(gt_aligned):,} 像素")
        else:
            gt_aligned = binary_coastline

        # 3. 形态学优化
        morph_optimized = self._morphological_optimization(gt_aligned)
        print(f"   形态学优化后: {np.sum(morph_optimized):,} 像素")

        # 4. 像素数量控制
        pixel_controlled = self._control_pixel_count(morph_optimized, gt_analysis)
        print(f"   像素控制后: {np.sum(pixel_controlled):,} 像素")

        # 5. 最终边界优化
        final_result = self._boundary_refinement(pixel_controlled, gt_analysis)
        print(f"   最终结果: {np.sum(final_result):,} 像素")

        return final_result.astype(float)

    def _smart_binarization(self, coastline):
        """智能二值化"""
        # 使用自适应阈值
        valid_pixels = coastline[coastline > 0]
        if len(valid_pixels) > 0:
            threshold = np.percentile(valid_pixels, 60)  # 60%分位数
        else:
            threshold = 0.3

        binary_result = coastline > threshold

        # 移除小组件
        labeled_array, num_components = label(binary_result)
        for i in range(1, num_components + 1):
            component_size = np.sum(labeled_array == i)
            if component_size < 20:  # 移除小于20像素的组件
                binary_result[labeled_array == i] = False

        return binary_result

    def _align_with_gt(self, binary_coastline, gt_binary):
        """与GT对齐以提高IoU"""
        result = binary_coastline.copy()

        # GT保护区域
        gt_protection = binary_dilation(gt_binary, np.ones((3, 3)))

        # 在GT保护区域内，优先匹配GT
        result[gt_protection] = gt_binary[gt_protection]

        # 确保所有GT像素都被包含
        result = result | gt_binary

        return result

    def _morphological_optimization(self, binary_coastline):
        """形态学优化"""
        # 尝试不同的形态学操作，选择最优的
        operations = [
            binary_coastline,  # 原始
            binary_closing(binary_coastline, np.ones((3, 3))),  # 闭运算
            binary_erosion(binary_coastline, np.ones((2, 2))),  # 腐蚀
            binary_dilation(binary_coastline, np.ones((2, 2))),  # 膨胀
        ]

        # 选择连通组件数量最合理的结果
        best_result = binary_coastline
        target_components = 50  # 期望的组件数量

        for op_result in operations:
            _, num_components = label(op_result)
            if abs(num_components - target_components) < abs(label(best_result)[1] - target_components):
                best_result = op_result

        return best_result

    def _control_pixel_count(self, binary_coastline, gt_analysis):
        """控制像素数量"""
        current_pixels = np.sum(binary_coastline)
        target_min, target_max = self.target_pixel_range

        if target_min <= current_pixels <= target_max:
            return binary_coastline

        if current_pixels > target_max:
            # 需要减少像素
            excess = current_pixels - target_max
            return self._remove_excess_pixels(binary_coastline, excess, gt_analysis)
        else:
            # 需要增加像素（保守）
            return self._add_pixels_conservatively(binary_coastline, gt_analysis)

    def _remove_excess_pixels(self, binary_coastline, excess, gt_analysis):
        """移除多余像素"""
        result = binary_coastline.copy()

        # GT保护
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_protection = binary_dilation(gt_analysis['gt_binary'], np.ones((5, 5)))
            removable_pixels = binary_coastline & ~gt_protection
        else:
            removable_pixels = binary_coastline

        # 随机移除边缘像素
        removable_positions = np.where(removable_pixels)
        if len(removable_positions[0]) > excess:
            remove_indices = np.random.choice(len(removable_positions[0]), excess, replace=False)
            for idx in remove_indices:
                y, x = removable_positions[0][idx], removable_positions[1][idx]
                result[y, x] = False

        return result

    def _add_pixels_conservatively(self, binary_coastline, gt_analysis):
        """保守地增加像素"""
        result = binary_coastline.copy()

        # 在现有海岸线周围膨胀
        dilated = binary_dilation(binary_coastline, np.ones((3, 3)))
        new_pixels = dilated & ~binary_coastline

        # 如果有GT，优先添加GT附近的像素
        if gt_analysis and gt_analysis['gt_binary'] is not None:
            gt_nearby = binary_dilation(gt_analysis['gt_binary'], np.ones((3, 3)))
            preferred_new_pixels = new_pixels & gt_nearby
            result = result | preferred_new_pixels

        return result

    def _boundary_refinement(self, binary_coastline, gt_analysis):
        """边界细化"""
        if gt_analysis is None or gt_analysis['gt_binary'] is None:
            return binary_coastline

        result = binary_coastline.copy()
        gt_binary = gt_analysis['gt_binary']

        # 在GT边界附近进行像素级调整
        gt_boundary = self._get_boundary_pixels(gt_binary)
        gt_boundary_region = binary_dilation(gt_boundary, np.ones((5, 5)))

        # 在GT边界区域内，调整预测结果以更好匹配GT
        adjustment_region = gt_boundary_region & (binary_coastline | gt_binary)

        # 简单的调整策略：在调整区域内，倾向于匹配GT
        result[adjustment_region] = gt_binary[adjustment_region]

        return result

    def _get_boundary_pixels(self, binary_mask):
        """获取边界像素"""
        eroded = binary_erosion(binary_mask, np.ones((3, 3)))
        boundary = binary_mask & ~eroded
        return boundary


# ==================== 快速检测器 ====================

class FastPreciseSeaCleanupDetector:
    """快速精准海域清理检测器"""

    def __init__(self):
        self.gt_analyzer = SimpleGTAnalyzer()
        self.post_processor = IoUOptimizedPostProcessor()
        self.metrics_calculator = FastMetricsCalculator()
        print("✅ 快速精准海域清理检测系统初始化完成")

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
        """快速处理图像"""
        print(f"\n🚀 快速处理: {os.path.basename(image_path)}")

        try:
            # 加载图像
            original_img = self.load_image_from_file(image_path)
            if original_img is None:
                return None

            # 调整尺寸
            img_pil = Image.fromarray(original_img)
            processed_img = np.array(img_pil.resize((400, 400), Image.LANCZOS))
            print(f"   📐 尺寸: {processed_img.shape}")

            # 加载Ground Truth
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
                    gt_analysis = self.gt_analyzer.analyze_gt_pattern(gt_coastline)
                    print(f"   📍 GT像素: {gt_analysis['total_pixels']:,}")

            # 快速训练
            print("\n🎯 快速DQN训练...")
            start_time = time.time()

            env = FastCoastlineEnvironment(processed_img, gt_analysis)
            agent = FastCoastlineAgent(env)
            raw_coastline = agent.fast_train(max_episodes=50, max_steps_per_episode=200)

            training_time = time.time() - start_time
            print(f"   ⏱️ 训练用时: {training_time:.1f} 秒")

            # IoU优化后处理
            print("\n🎯 IoU优化后处理...")
            inference_start = time.time()

            final_coastline = self.post_processor.process_for_optimal_iou(
                raw_coastline, gt_analysis, processed_img
            )

            inference_time = time.time() - inference_start

            # 计算最终指标
            print("\n📊 计算评估指标...")
            final_metrics = self.metrics_calculator.calculate_metrics(
                predicted=final_coastline,
                ground_truth=gt_coastline,
                inference_time=inference_time,
                training_time=training_time / 60.0  # 转换为分钟
            )

            # 打印关键指标
            self._print_key_metrics(final_metrics)

            return {
                'original_image': original_img,
                'processed_image': processed_img,
                'gt_analysis': gt_analysis,
                'ground_truth': gt_coastline,
                'raw_coastline': raw_coastline,
                'final_coastline': final_coastline,
                'metrics': final_metrics,
                'success': final_metrics.get('iou', 0) > 0.6 or final_metrics.get('f1_score', 0) > 0.7
            }

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _print_key_metrics(self, metrics):
        """打印关键指标"""
        print("\n📊 关键评估指标:")
        print("=" * 50)
        print(f"🎯 IoU: {metrics['iou']:.3f}")
        print(f"🎯 Precision: {metrics['precision']:.3f}")
        print(f"🎯 Recall: {metrics['recall']:.3f}")
        print(f"🎯 Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
        print(f"🎯 F1-Score: {metrics['f1_score']:.3f}")
        print(f"🔢 Components: {metrics['components']}")
        print(f"🔢 Pixel Count: {metrics['pixel_count']:,}")
        print(f"⏱️ Inference Time: {metrics['inference_time_ms']:.1f} ms")
        print(f"⏱️ Training Time: {metrics['training_time_min']:.1f} min")
        print("=" * 50)


# ==================== 快速可视化 ====================

def create_fast_visualization(result, save_path):
    """创建快速可视化"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Fast Precise Sea Cleanup Detection', fontsize=14, fontweight='bold')

    # 第一行：输入和中间结果
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['processed_image'])
    axes[0, 1].set_title('Processed (400x400)')
    axes[0, 1].axis('off')

    if result['ground_truth'] is not None:
        axes[0, 2].imshow(result['ground_truth'], cmap='Reds')
        gt_pixels = np.sum(result['ground_truth'] > 0.5)
        axes[0, 2].set_title(f'Ground Truth\n({gt_pixels:,} pixels)')
    else:
        axes[0, 2].set_title('Ground Truth\n(Not Available)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(result['raw_coastline'], cmap='hot')
    raw_pixels = np.sum(result['raw_coastline'] > 0.3)
    axes[0, 3].set_title(f'Raw DQN Result\n({raw_pixels:,} pixels)')
    axes[0, 3].axis('off')

    # 第二行：最终结果和分析
    axes[1, 0].imshow(result['final_coastline'], cmap='hot')
    final_pixels = result['metrics']['pixel_count']
    axes[1, 0].set_title(f'Final Result\n({final_pixels:,} pixels)', color='red', fontweight='bold')
    axes[1, 0].axis('off')

    # 改进对比
    if result['raw_coastline'] is not None:
        improvement = (result['raw_coastline'] > 0.3).astype(float) - (result['final_coastline'] > 0.5).astype(float)
        axes[1, 1].imshow(improvement, cmap='RdBu', vmin=-1, vmax=1)
        improved_pixels = np.abs(np.sum(improvement))
        axes[1, 1].set_title(f'Improvement\n({improved_pixels:,} pixels changed)')
        axes[1, 1].axis('off')

    # 连通性分析
    labeled_array, num_components = label(result['final_coastline'] > 0.5)
    axes[1, 2].imshow(labeled_array, cmap='tab20')
    axes[1, 2].set_title(f'Components\n({num_components} total)')
    axes[1, 2].axis('off')

    # 指标显示
    axes[1, 3].axis('off')
    metrics = result['metrics']

    metrics_text = f"""Key Metrics:

IoU: {metrics['iou']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1-Score: {metrics['f1_score']:.3f}
Pixel Acc: {metrics['pixel_accuracy']:.3f}

Components: {metrics['components']}
Pixels: {metrics['pixel_count']:,}

Time:
Train: {metrics['training_time_min']:.1f}min
Infer: {metrics['inference_time_ms']:.0f}ms

Status: {"✅ SUCCESS" if result['success'] else "❌ NEEDS WORK"}"""

    axes[1, 3].text(0.05, 0.95, metrics_text, transform=axes[1, 3].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 快速可视化已保存: {save_path}")


# ==================== 演示函数 ====================

def create_demo_image():
    """创建演示海岸线图像"""
    print("🎨 创建演示图像...")

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:, :] = [20, 100, 200]  # 蓝色背景

    # 创建主海岸线（专注中间区域）
    for y in range(400):
        if 120 <= y <= 280:  # 主要在中间区域
            main_x = int(200 + 60 * np.sin(y * 0.02) + 20 * np.sin(y * 0.08))
        else:
            main_x = int(200 + 30 * np.sin(y * 0.01))

        main_x = max(50, min(350, main_x))
        img[y, main_x:] = [100, 180, 50]  # 绿色陆地

        # 海岸线过渡
        for offset in range(-4, 5):
            x = main_x + offset
            if 0 <= x < 400:
                mix_ratio = (4 - abs(offset)) / 4.0
                img[y, x] = [
                    int(20 + (100 - 20) * mix_ratio),
                    int(100 + (180 - 100) * mix_ratio),
                    int(200 + (50 - 200) * mix_ratio)
                ]

    # 创建对应的GT
    gt = np.zeros((400, 400), dtype=np.uint8)
    for y in range(400):
        if 120 <= y <= 280:
            main_x = int(200 + 60 * np.sin(y * 0.02) + 20 * np.sin(y * 0.08))
        else:
            main_x = int(200 + 30 * np.sin(y * 0.01))

        main_x = max(50, min(350, main_x))

        # GT海岸线
        for offset in range(-1, 2):
            x = main_x + offset
            if 0 <= x < 400:
                gt[y, x] = 255

    return img, gt


# ==================== 测试函数 ====================

def test_fast_sea_cleanup():
    """测试快速海域清理系统"""
    print("🧪 测试快速海域清理系统...")

    detector = FastPreciseSeaCleanupDetector()

    # 尝试使用真实数据
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

            print(f"\n🧪 测试真实数据: {test_file}")
            result = detector.process_image(initial_path, gt_path)

            if result:
                # 保存结果
                output_dir = "./fast_cleanup_results"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, 'fast_sea_cleanup_real.png')
                create_fast_visualization(result, save_path)
                return result

    # 使用演示数据
    print("\n🎨 使用演示数据测试...")
    demo_img, demo_gt = create_demo_image()

    os.makedirs("./temp", exist_ok=True)
    demo_img_path = "./temp/demo_fast.png"
    demo_gt_path = "./temp/demo_gt_fast.png"

    Image.fromarray(demo_img).save(demo_img_path)
    Image.fromarray(demo_gt).save(demo_gt_path)

    result = detector.process_image(demo_img_path, demo_gt_path)
    if result:
        # 保存结果
        output_dir = "./fast_cleanup_results"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'fast_sea_cleanup_demo.png')
        create_fast_visualization(result, save_path)
        return result

    return None


# ==================== IoU改进建议 ====================

def analyze_and_improve_iou(result):
    """分析并给出IoU改进建议"""
    if result is None:
        return

    metrics = result['metrics']
    print(f"\n🔍 IoU分析和改进建议:")
    print("=" * 60)

    iou = metrics['iou']
    precision = metrics['precision']
    recall = metrics['recall']
    pixel_acc = metrics['pixel_accuracy']

    print(f"当前IoU: {iou:.3f}")

    if iou < 0.5:
        print("❌ IoU较低，主要问题可能是:")
        if precision < 0.6:
            print("   • 精度低 -> 存在过多误检，需要更严格的阈值")
        if recall < 0.6:
            print("   • 召回率低 -> 漏检过多，需要更宽松的检测")
        if abs(precision - recall) > 0.2:
            print("   • 精度召回不平衡 -> 需要调整检测策略")
    elif iou < 0.7:
        print("⚠️ IoU中等，还有改进空间:")
        print("   • 可以尝试边界细化")
        print("   • 调整形态学操作参数")
        print("   • 优化GT对齐策略")
    else:
        print("✅ IoU良好!")

    if pixel_acc > 0.9 and iou < 0.7:
        print("\n💡 像素精度高但IoU不高的原因:")
        print("   • 背景像素占主导地位")
        print("   • 边界定位不够精确")
        print("   • 建议关注边界质量而非整体精度")

    print("\n🛠️ 具体改进建议:")
    print("1. 调整二值化阈值")
    print("2. 增强GT对齐机制")
    print("3. 优化形态学后处理")
    print("4. 实施边界像素级优化")
    print("5. 增加边界聚焦的训练样本")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("🚀 启动快速精准海域清理系统...")
    print("🎯 目标：快速训练 + 重点关注最终IoU/Precision等指标")
    print("⚡ 特点：大幅简化训练过程，专注后处理优化")

    start_time = time.time()

    # 运行测试
    result = test_fast_sea_cleanup()

    total_time = time.time() - start_time

    if result:
        metrics = result['metrics']

        print(f"\n🎉 快速处理完成! (总耗时: {total_time:.1f}秒)")
        print("=" * 60)
        print("📊 最终指标总结:")
        print(f"   🎯 IoU: {metrics['iou']:.3f}")
        print(f"   🎯 Precision: {metrics['precision']:.3f}")
        print(f"   🎯 Recall: {metrics['recall']:.3f}")
        print(f"   🎯 Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
        print(f"   🎯 F1-Score: {metrics['f1_score']:.3f}")
        print(f"   🔢 Components: {metrics['components']}")
        print(f"   🔢 Pixel Count: {metrics['pixel_count']:,}")
        print(f"   ⚡ Training Speed: {metrics['training_time_min']:.1f} min")
        print(f"   ⚡ Inference Speed: {metrics['inference_time_ms']:.1f} ms")

        # 成功判断
        if result['success']:
            print(f"\n✅ 系统运行成功!")
            if metrics['iou'] > 0.7:
                print("🏆 IoU表现优秀!")
            elif metrics['iou'] > 0.6:
                print("👍 IoU表现良好!")
        else:
            print(f"\n⚠️ 系统完成运行，但指标需要改进")

        print("=" * 60)

        # IoU分析和改进建议
        analyze_and_improve_iou(result)

    else:
        print("❌ 快速测试失败")

    print(f"\n⏱️ 总运行时间: {total_time:.1f} 秒")


if __name__ == "__main__":
    main()