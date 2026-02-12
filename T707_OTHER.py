"""
英国其他城市海岸线检测测试脚本 - 无GT版本
基于已训练的DQN模型测试英国其他城市的海岸线分割
路径：E:/Other/ (Blackpool, Liverpool, Ortsmouth, Southport)
特点：无Ground Truth，重点评估分割质量和视觉效果
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

print("🇬🇧 英国其他城市海岸线检测系统!")
print("目标城市：Blackpool, Liverpool, Ortsmouth, Southport")
print("特点：无GT依赖 + 预训练模型 + 质量评估")
print("=" * 90)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ==================== 复用核心类（精简版） ====================

class BasicImageProcessor:
    @staticmethod
    def rgb_to_gray(rgb_image):
        if len(rgb_image.shape) == 3:
            return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])
        return rgb_image


class HSVAttentionSupervisor:
    """HSV注意力监督器 - 无GT版本"""

    def __init__(self):
        print("✅ HSV注意力监督器初始化完成（无GT模式）")
        self.water_hsv_range = self._define_water_hsv_range()
        self.land_hsv_range = self._define_land_hsv_range()

    def _define_water_hsv_range(self):
        return {
            'hue_range': (180, 240),
            'saturation_min': 0.2,
            'value_min': 0.1
        }

    def _define_land_hsv_range(self):
        return {
            'hue_range': (60, 120),
            'saturation_min': 0.1,
            'value_min': 0.2
        }

    def analyze_image_hsv(self, rgb_image, gt_analysis=None):
        """分析图像的HSV特征（无GT版本）"""
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

        coastline_guidance = self._generate_coastline_guidance(water_mask, land_mask)

        return {
            'hsv_image': hsv_image,
            'water_mask': water_mask,
            'land_mask': land_mask,
            'coastline_guidance': coastline_guidance,
            'transition_strength': self._calculate_transition_strength(hsv_image, water_mask, land_mask)
        }

    def _detect_water_regions(self, hsv_image):
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

    def _generate_coastline_guidance(self, water_mask, land_mask):
        water_boundary = binary_dilation(water_mask, np.ones((3, 3))) & ~water_mask
        land_boundary = binary_dilation(land_mask, np.ones((3, 3))) & ~land_mask
        coastline_candidates = water_boundary | land_boundary

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
        guidance_strength = coastline_guidance * guidance_strength

        if guidance_strength.max() > 0:
            guidance_strength = guidance_strength / guidance_strength.max()

        return guidance_strength

    def _calculate_transition_strength(self, hsv_image, water_mask, land_mask):
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

        return transition_strength


class ConstrainedActionSpace:
    def __init__(self):
        self.base_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                             (0, 1), (1, -1), (1, 0), (1, 1)]
        print("✅ 约束动作空间初始化完成")

    def get_allowed_actions(self, current_position, coastline_state, hsv_analysis):
        allowed_actions = []
        context = self._analyze_position_context(current_position, coastline_state, hsv_analysis)

        for i, action in enumerate(self.base_actions):
            if self._is_action_allowed(action, context, current_position, hsv_analysis):
                allowed_actions.append(i)

        return allowed_actions if allowed_actions else [0, 1, 3, 4]

    def _analyze_position_context(self, position, coastline_state, hsv_analysis):
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
        dy, dx = action

        if context['near_water'] and abs(dy) > 0:
            if abs(dy) > 1 or (abs(dy) == 1 and abs(dx) == 0):
                return False

        if abs(dy) + abs(dx) > 2:
            return False

        return True


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

        hsv_bonus = 0.0
        if hsv_analysis:
            coastline_guidance = hsv_analysis['coastline_guidance']
            if coastline_guidance[y, x] > 0.3:
                hsv_bonus = coastline_guidance[y, x] * 15.0

        self.visit_history[pos_key] = visit_count + 1
        self.step_count += 1

        return visit_bonus + hsv_bonus


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


class ConstrainedCoastlineEnvironment:
    """约束的海岸线环境 - 无GT版本"""

    def __init__(self, image, gt_analysis=None):
        self.image = image
        self.gt_analysis = gt_analysis  # 对于其他城市，这将是None
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

        print(f"✅ 约束海岸线环境初始化完成（无GT模式）")

    def _detect_edges(self):
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

    def _identify_effective_coastline_region(self):
        height, width = self.height, self.width

        # 无GT情况下，使用标准的中间1/3区域策略
        effective_y_min = height//20
        effective_y_max =20*height//20

        effective_region = np.zeros((height, width), dtype=bool)
        effective_region[effective_y_min:effective_y_max, :] = True

        return effective_region

    def get_state_tensor(self, position):
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

        curiosity_bonus = self.curiosity_explorer.get_curiosity_bonus(
            position, self.hsv_analysis, self.current_coastline
        )
        features[15] = min(1.0, curiosity_bonus / 50.0)

        features[19] = y / self.height
        features[20] = x / self.width

        middle_start = self.height//100000
        middle_end = 3 * self.height // 3

        if middle_start <= y <= middle_end:
            features[24] = 1.0
        elif y < middle_start:
            features[24] = -1.0
        else:
            features[24] = -0.5

        return torch.FloatTensor(features).unsqueeze(0).to(device)

    def step(self, position, action_idx):
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
        """计算奖励函数 - 无GT版本"""
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
        reward += hsv_reward * 100.0

        # 海陆分离奖励
        sea_land_separation_reward = self._calculate_sea_land_separation_reward(new_pos)
        reward += sea_land_separation_reward

        return reward

    def _calculate_hsv_reward(self, position):
        y, x = position
        guidance_score = self.hsv_analysis['coastline_guidance'][y, x]
        transition_score = self.hsv_analysis['transition_strength'][y, x]
        return guidance_score + transition_score

    def _calculate_sea_land_separation_reward(self, position):
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
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            self.current_coastline[y, x] = min(1.0, self.current_coastline[y, x] + value)


class ConstrainedCoastlineAgent:
    """约束的海岸线代理"""

    def __init__(self, env, lr=1e-4, gamma=0.98, epsilon_start=0.1, epsilon_end=0.05, epsilon_decay=0.995):
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

        print(f"✅ 约束DQN代理初始化完成（推理模式）")

    def select_action(self, rgb_state, hsv_state, position, enhanced_features, training=False):
        """选择动作 - 推理模式"""
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

    def load_model(self, load_path):
        """加载预训练模型"""
        if os.path.exists(load_path):
            try:
                checkpoint = torch.load(load_path, map_location=device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.epsilon = self.epsilon_end  # 使用低探索率进行推理
                print(f"✅ 预训练模型已加载: {load_path}")
                return True
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                return False
        return False

    def apply_pretrained_inference(self, max_inference_steps=1200):
        """使用预训练模型进行推理 - 优化版"""
        print("🔮 使用预训练模型进行海岸线推理（优化版）...")

        # 获取搜索区域
        search_positions = np.where(self.env.search_region)
        candidate_positions = list(zip(search_positions[0], search_positions[1]))

        if not candidate_positions:
            print("   ⚠️ 未找到搜索区域")
            return self.env.current_coastline

        height = self.env.height
        middle_start = height // 3
        middle_end = 2 * height // 3

        # 重点关注中间区域
        middle_positions = [pos for pos in candidate_positions if middle_start <= pos[0] <= middle_end]

        if not middle_positions:
            middle_positions = candidate_positions[:200]

        # 智能位置选择策略 - 更严格的筛选
        high_value_positions = []
        for pos in middle_positions:
            y, x = pos
            guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
            transition_score = self.env.hsv_analysis['transition_strength'][y, x]
            edge_score = self.env.edge_map[y, x]

            # 综合评分，提高阈值
            combined_score = guidance_score * 0.4 + transition_score * 0.4 + edge_score * 0.2

            if combined_score > 0.4:  # 提高阈值
                high_value_positions.append((combined_score, pos))

        # 按价值排序
        high_value_positions.sort(reverse=True)
        prioritized_positions = [pos for _, pos in high_value_positions[:max_inference_steps//2]]

        # 如果高价值位置不够，降低阈值补充
        if len(prioritized_positions) < max_inference_steps // 4:
            print("   🔄 高价值位置不足，扩展搜索...")
            additional_positions = []
            for pos in middle_positions:
                if pos not in prioritized_positions:
                    y, x = pos
                    guidance_score = self.env.hsv_analysis['coastline_guidance'][y, x]
                    transition_score = self.env.hsv_analysis['transition_strength'][y, x]
                    combined_score = guidance_score + transition_score

                    if combined_score > 0.25:  # 降低的阈值
                        additional_positions.append((combined_score, pos))

            additional_positions.sort(reverse=True)
            prioritized_positions.extend([pos for _, pos in additional_positions[:max_inference_steps//2]])

        # 补充随机位置
        remaining_positions = [pos for pos in middle_positions if pos not in prioritized_positions]
        random.shuffle(remaining_positions)

        final_positions = prioritized_positions + remaining_positions[:max_inference_steps]

        print(f"   🎯 推理位置数: {len(final_positions)}")
        print(f"   📊 高价值位置: {len(prioritized_positions)}")

        improvements = 0
        total_reward = 0.0

        # 多轮推理策略
        for round_num in range(3):  # 分3轮进行
            round_improvements = 0
            round_positions = final_positions[round_num * len(final_positions):(round_num + 1) * len(final_positions)]

            print(f"   🔄 第 {round_num + 1} 轮推理，位置数: {len(round_positions)}")

            for step, position in enumerate(round_positions):
                # 获取状态
                rgb_state, hsv_state = self.env.get_state_tensor(position)
                enhanced_features = self.env.get_enhanced_features(position)

                # 使用预训练模型进行推理
                action = self.select_action(rgb_state, hsv_state, position, enhanced_features, training=False)

                # 执行动作
                next_position, reward = self.env.step(position, action)
                total_reward += reward

                # 动态阈值策略
                if round_num == 0:
                    reward_threshold = 12.0  # 第一轮要求较高
                elif round_num == 1:
                    reward_threshold = 8.0   # 第二轮适中
                else:
                    reward_threshold = 5.0   # 第三轮较低

                if reward > reward_threshold:
                    y, x = next_position
                    is_middle_region = middle_start <= y <= middle_end

                    # 根据轮次调整更新值
                    if round_num == 0:
                        update_value = 0.9 if is_middle_region else 0.7
                    elif round_num == 1:
                        update_value = 0.8 if is_middle_region else 0.6
                    else:
                        update_value = 0.7 if is_middle_region else 0.5

                    self.env.update_coastline(next_position, update_value)
                    improvements += 1
                    round_improvements += 1

            print(f"   ✅ 第 {round_num + 1} 轮完成，改进: {round_improvements}")

        final_pixels = np.sum(self.env.current_coastline > 0.3)
        avg_reward = total_reward / len(final_positions) if final_positions else 0

        print(f"   ✅ 推理完成: {final_pixels:,} 像素, 总改进: {improvements}")
        print(f"   📊 平均奖励: {avg_reward:.2f}")

        # 如果像素数量太少，进行补充
        if final_pixels < 80000:
            print("   🔧 像素数量偏少，执行补充策略...")
            self._supplement_coastline()
            final_pixels = np.sum(self.env.current_coastline > 0.3)
            print(f"   📈 补充后像素: {final_pixels:,}")

        return self.env.current_coastline

    def _supplement_coastline(self):
        """补充海岸线像素"""
        # 找到现有海岸线的邻近区域
        current_binary = (self.env.current_coastline > 0.3).astype(bool)

        # 膨胀现有海岸线
        from scipy.ndimage import binary_dilation
        expanded = binary_dilation(current_binary, np.ones((5, 5)))
        new_candidates = expanded & ~current_binary

        # 在搜索区域内的候选
        valid_candidates = new_candidates & self.env.search_region

        candidate_positions = np.where(valid_candidates)

        for i in range(len(candidate_positions[0])):
            y, x = candidate_positions[0][i], candidate_positions[1][i]

            # 检查HSV指导
            guidance = self.env.hsv_analysis['coastline_guidance'][y, x]
            transition = self.env.hsv_analysis['transition_strength'][y, x]

            if guidance > 0.2 or transition > 0.3:
                self.env.update_coastline((y, x), 0.6)


# ==================== 无GT质量评估器 ====================

class NoGTQualityAssessor:
    """无Ground Truth质量评估器"""

    def __init__(self):
        print("✅ 无GT质量评估器初始化完成")

    def assess_coastline_quality(self, coastline, hsv_analysis, original_image):
        """评估海岸线质量（无GT版本）"""
        print("📊 评估海岸线质量（无GT模式）...")

        metrics = {}
        pred_binary = (coastline > 0.5).astype(bool)
        coastline_pixels = np.sum(pred_binary)

        # 基础统计
        metrics['coastline_pixels'] = int(coastline_pixels)

        # 1. 连通性分析
        labeled_array, num_components = label(pred_binary)
        metrics['num_components'] = int(num_components)

        # 计算主要组件大小
        if num_components > 0:
            component_sizes = []
            for i in range(1, num_components + 1):
                size = np.sum(labeled_array == i)
                component_sizes.append(size)

            main_component_ratio = max(component_sizes) / coastline_pixels if coastline_pixels > 0 else 0
            metrics['main_component_ratio'] = float(main_component_ratio)
            metrics['fragmentation_score'] = float(1.0 - main_component_ratio)
        else:
            metrics['main_component_ratio'] = 0.0
            metrics['fragmentation_score'] = 1.0

        # 2. 海域清理效果评估
        water_mask = hsv_analysis['water_mask']
        water_intrusion = np.sum(pred_binary.astype(bool) & water_mask.astype(bool)) / (coastline_pixels + 1e-8)
        metrics['water_intrusion_ratio'] = float(water_intrusion)
        metrics['sea_cleanup_score'] = float(max(0.0, 1.0 - water_intrusion * 2))

        # 3. 区域分布分析
        height = pred_binary.shape[0]
        third_height = height // 3

        upper_pixels = np.sum(pred_binary[:third_height, :])
        middle_pixels = np.sum(pred_binary[third_height:2*third_height, :])
        lower_pixels = np.sum(pred_binary[2*third_height:, :])

        if coastline_pixels > 0:
            upper_ratio = upper_pixels / coastline_pixels
            middle_ratio = middle_pixels / coastline_pixels
            lower_ratio = lower_pixels / coastline_pixels

            # 理想情况：海岸线主要集中在中间
            distribution_score = middle_ratio * 1.0 + (1 - abs(upper_ratio - lower_ratio)) * 0.5
        else:
            upper_ratio = middle_ratio = lower_ratio = 0.0
            distribution_score = 0.0

        metrics['upper_ratio'] = float(upper_ratio)
        metrics['middle_ratio'] = float(middle_ratio)
        metrics['lower_ratio'] = float(lower_ratio)
        metrics['distribution_score'] = float(distribution_score)

        # 4. 密度合理性评估 - 调整目标范围
        target_min, target_max = 5000, 50000  # 降低目标范围，适应英国城市
        if target_min <= coastline_pixels <= target_max:
            density_score = 1.0
        elif coastline_pixels < target_min:
            # 对于像素过少的情况，给予适度惩罚而非完全失败
            density_score = max(0.3, coastline_pixels / target_min)
        else:
            density_score = max(0.0, 1.0 - (coastline_pixels - target_max) / target_max)
        metrics['density_score'] = float(density_score)

        # 5. 形状连续性评估
        continuity_score = self._assess_coastline_continuity(pred_binary)
        metrics['continuity_score'] = float(continuity_score)

        # 6. HSV指导符合度
        guidance_conformity = self._assess_hsv_guidance_conformity(pred_binary, hsv_analysis)
        metrics['guidance_conformity'] = float(guidance_conformity)

        # 7. 边缘质量评估
        edge_quality = self._assess_edge_quality(pred_binary, original_image)
        metrics['edge_quality'] = float(edge_quality)

        # 8. 综合质量评分（无GT版本）
        overall_score = self._calculate_overall_score_no_gt(metrics)
        metrics['overall_score'] = float(overall_score)

        # 9. 质量等级评定
        quality_level = self._determine_quality_level(overall_score)
        metrics['quality_level'] = quality_level

        return metrics

    def _assess_coastline_continuity(self, coastline_binary):
        """评估海岸线连续性"""
        if not np.any(coastline_binary):
            return 0.0

        # 计算海岸线的骨架
        try:
            if HAS_SKIMAGE:
                skeleton = skeletonize(coastline_binary)
                skeleton_pixels = np.sum(skeleton)
                total_pixels = np.sum(coastline_binary)
                continuity = skeleton_pixels / (total_pixels + 1e-8)
            else:
                # 简化的连续性评估
                continuity = self._simple_continuity_assessment(coastline_binary)
        except:
            continuity = self._simple_continuity_assessment(coastline_binary)

        return min(1.0, continuity * 2)

    def _simple_continuity_assessment(self, coastline_binary):
        """简化的连续性评估"""
        # 计算每行的连续段数
        total_segments = 0
        total_rows_with_coastline = 0

        height, width = coastline_binary.shape

        for y in range(height):
            row = coastline_binary[y, :]
            if np.any(row):
                total_rows_with_coastline += 1
                # 计算连续段
                segments = 0
                in_segment = False
                for x in range(width):
                    if row[x] and not in_segment:
                        segments += 1
                        in_segment = True
                    elif not row[x] and in_segment:
                        in_segment = False
                total_segments += segments

        if total_rows_with_coastline == 0:
            return 0.0

        avg_segments_per_row = total_segments / total_rows_with_coastline
        # 理想情况是每行平均1个连续段
        continuity = 1.0 / (avg_segments_per_row + 1e-8)
        return min(1.0, continuity)

    def _assess_hsv_guidance_conformity(self, coastline_binary, hsv_analysis):
        """评估与HSV指导的符合度"""
        guidance = hsv_analysis['coastline_guidance']
        transition = hsv_analysis['transition_strength']

        # 计算海岸线位置的平均指导值
        coastline_positions = np.where(coastline_binary)
        if len(coastline_positions[0]) == 0:
            return 0.0

        guidance_values = guidance[coastline_positions]
        transition_values = transition[coastline_positions]

        avg_guidance = np.mean(guidance_values)
        avg_transition = np.mean(transition_values)

        conformity = (avg_guidance + avg_transition) / 2.0
        return conformity

    def _assess_edge_quality(self, coastline_binary, original_image):
        """评估边缘质量"""
        if not np.any(coastline_binary):
            return 0.0

        # 计算图像边缘
        if len(original_image.shape) == 3:
            gray = np.dot(original_image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = original_image.copy()

        # Sobel边缘检测
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        grad_x = ndimage.convolve(gray, sobel_x)
        grad_y = ndimage.convolve(gray, sobel_y)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 归一化
        edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-8)

        # 计算海岸线位置的边缘强度
        coastline_positions = np.where(coastline_binary)
        edge_values = edge_magnitude[coastline_positions]

        avg_edge_strength = np.mean(edge_values)
        return avg_edge_strength

    def _calculate_overall_score_no_gt(self, metrics):
        """计算综合得分（无GT版本） - 优化版"""
        score = 0.0

        # 权重分配 - 调整权重以适应实际情况
        weights = {
            'density_score': 0.15,           # 降低密度权重
            'sea_cleanup_score': 0.25,      # 提高海域清理权重
            'distribution_score': 0.20,     # 提高区域分布权重
            'continuity_score': 0.15,       # 连续性
            'guidance_conformity': 0.15,    # HSV指导符合度
            'edge_quality': 0.10,           # 边缘质量
        }

        score += metrics['density_score'] * weights['density_score']
        score += metrics['sea_cleanup_score'] * weights['sea_cleanup_score']
        score += metrics['distribution_score'] * weights['distribution_score']
        score += metrics['continuity_score'] * weights['continuity_score']
        score += metrics['guidance_conformity'] * weights['guidance_conformity']
        score += metrics['edge_quality'] * weights['edge_quality']

        # 碎片化惩罚 - 减轻惩罚
        fragmentation_penalty = min(0.2, metrics['fragmentation_score'] * 0.3)
        score -= fragmentation_penalty

        # 连通组件惩罚 - 更宽容的阈值
        component_count = metrics['num_components']
        pixel_count = metrics['coastline_pixels']

        # 根据像素数量调整组件数量的合理范围
        if pixel_count < 1000:
            reasonable_components = 50
        elif pixel_count < 5000:
            reasonable_components = 100
        elif pixel_count < 20000:
            reasonable_components = 200
        else:
            reasonable_components = 300

        if component_count > reasonable_components:
            component_penalty = min(0.15, (component_count - reasonable_components) / reasonable_components * 0.3)
            score -= component_penalty

        # 特殊加分：如果主要组件比例很高，给予加分
        if metrics['main_component_ratio'] > 0.7:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _determine_quality_level(self, score):
        """确定质量等级"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.65:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        elif score >= 0.35:
            return "Poor"
        else:
            return "Very Poor"


# ==================== 英国城市海岸线检测器 ====================

class UKCitiesCoastlineDetector:
    """英国其他城市海岸线检测器"""

    def __init__(self):
        self.quality_assessor = NoGTQualityAssessor()
        print("✅ 英国城市海岸线检测器初始化完成")
        print("   🎯 目标：Blackpool, Liverpool, Ortsmouth, Southport")

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

    def process_uk_city(self, image_path, city_name, pretrained_model_path):
        """
        处理英国城市海岸线检测

        Args:
            image_path: 城市图像路径
            city_name: 城市名称
            pretrained_model_path: 预训练模型路径
        """
        print(f"\n🏴󠁧󠁢󠁥󠁮󠁧󠁿 处理英国城市: {city_name}")
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

            # 2. 创建环境（无GT）
            print("\n📍 步骤1: 创建检测环境（无GT模式）")
            env = ConstrainedCoastlineEnvironment(processed_img, gt_analysis=None)

            # 3. 创建代理并加载预训练模型
            print("\n📍 步骤2: 加载预训练模型")
            agent = ConstrainedCoastlineAgent(env)

            if not agent.load_model(pretrained_model_path):
                print(f"❌ 无法加载预训练模型: {pretrained_model_path}")
                return None

            # 4. 执行推理
            print("\n📍 步骤3: 执行海岸线推理")
            coastline_result = agent.apply_pretrained_inference(max_inference_steps=800)

            # 5. 质量评估
            print("\n📍 步骤4: 质量评估")
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
                'success': quality_metrics['overall_score'] > 0.4,
                'model_path': pretrained_model_path
            }

            # 显示结果摘要
            self._display_result_summary(city_name, quality_metrics)

            return result

        except Exception as e:
            print(f"❌ 处理 {city_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _display_result_summary(self, city_name, metrics):
        """显示结果摘要"""
        print(f"\n📊 {city_name} 检测结果摘要:")
        print(f"   🎯 综合得分: {metrics['overall_score']:.3f}")
        print(f"   📏 海岸线像素: {metrics['coastline_pixels']:,}")
        print(f"   🏆 质量等级: {metrics['quality_level']}")
        print(f"   🌊 海域清理: {metrics['sea_cleanup_score']:.3f}")
        print(f"   📍 区域分布: {metrics['distribution_score']:.3f}")
        print(f"   🔗 连续性: {metrics['continuity_score']:.3f}")
        print(f"   🧩 连通组件: {metrics['num_components']}")

        if metrics['overall_score'] > 0.65:
            print(f"   ✅ {city_name} 检测成功!")
        elif metrics['overall_score'] > 0.4:
            print(f"   ⚠️ {city_name} 检测一般")
        else:
            print(f"   ❌ {city_name} 检测需要改进")


# ==================== 可视化函数 ====================

def create_uk_cities_visualization(result, save_path):
    """创建英国城市海岸线检测可视化"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    city_name = result['city_name']
    fig.suptitle(f'UK City Coastline Detection - {city_name}',
                 fontsize=16, fontweight='bold')

    # 第一行：原图和分析
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title(f'{city_name} - Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['processed_image'])
    axes[0, 1].set_title('Processed Image (400x400)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['hsv_analysis']['water_mask'], cmap='Blues')
    axes[0, 2].set_title('Water Regions (HSV)')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(result['hsv_analysis']['land_mask'], cmap='Greens')
    axes[0, 3].set_title('Land Regions (HSV)')
    axes[0, 3].axis('off')

    # 第二行：指导图和结果
    axes[1, 0].imshow(result['hsv_analysis']['coastline_guidance'], cmap='plasma')
    axes[1, 0].set_title('Coastline Guidance')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['hsv_analysis']['transition_strength'], cmap='viridis')
    axes[1, 1].set_title('Transition Strength')
    axes[1, 1].axis('off')

    coastline_binary = (result['coastline_result'] > 0.5).astype(float)
    axes[1, 2].imshow(coastline_binary, cmap='Reds')
    pixels = np.sum(coastline_binary)
    axes[1, 2].set_title(f'Detected Coastline\n({pixels:,} pixels)')
    axes[1, 2].axis('off')

    # 叠加显示
    overlay = result['processed_image'].copy()
    coastline_coords = np.where(coastline_binary)
    if len(coastline_coords[0]) > 0:
        overlay[coastline_coords[0], coastline_coords[1]] = [255, 0, 0]  # 红色海岸线
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title('Coastline Overlay')
    axes[1, 3].axis('off')

    # 第三行：连通性和统计分析
    labeled_coastline, num_components = label(coastline_binary)
    axes[2, 0].imshow(labeled_coastline, cmap='tab20')
    axes[2, 0].set_title(f'Connected Components\n({num_components} components)')
    axes[2, 0].axis('off')

    # 区域分布分析
    height = coastline_binary.shape[0]
    third = height // 3

    region_analysis = np.zeros_like(coastline_binary)
    region_analysis[:third, :] = coastline_binary[:third, :] * 0.3  # 上部 - 暗
    region_analysis[third:2*third, :] = coastline_binary[third:2*third, :] * 1.0  # 中部 - 亮
    region_analysis[2*third:, :] = coastline_binary[2*third:, :] * 0.6  # 下部 - 中等

    axes[2, 1].imshow(region_analysis, cmap='hot')
    axes[2, 1].set_title('Regional Distribution\n(Bright=Middle)')
    axes[2, 1].axis('off')

    # 海域入侵分析
    water_mask = result['hsv_analysis']['water_mask']
    water_intrusion = coastline_binary.astype(bool) & water_mask.astype(bool)
    axes[2, 2].imshow(water_intrusion.astype(float), cmap='Blues')
    intrusion_pixels = np.sum(water_intrusion)
    axes[2, 2].set_title(f'Water Intrusion\n({intrusion_pixels:,} pixels)')
    axes[2, 2].axis('off')

    # 清除第四个子图用于统计信息
    axes[2, 3].axis('off')

    # 统计信息文本
    metrics = result['quality_metrics']
    stats_text = f"""🏴󠁧󠁢󠁥󠁮󠁧󠁿 {city_name} Detection Results

Overall Quality: {metrics['overall_score']:.3f}
Quality Level: {metrics['quality_level']}
Status: {"✅ SUCCESS" if result['success'] else "❌ NEEDS IMPROVEMENT"}

Coastline Statistics:
• Total pixels: {metrics['coastline_pixels']:,}
• Connected components: {metrics['num_components']}
• Main component ratio: {metrics['main_component_ratio']:.1%}
• Fragmentation score: {metrics['fragmentation_score']:.3f}

Quality Metrics:
• Sea cleanup score: {metrics['sea_cleanup_score']:.3f}
• Water intrusion: {metrics['water_intrusion_ratio']:.1%}
• Distribution score: {metrics['distribution_score']:.3f}
• Continuity score: {metrics['continuity_score']:.3f}
• Guidance conformity: {metrics['guidance_conformity']:.3f}
• Edge quality: {metrics['edge_quality']:.3f}
• Density score: {metrics['density_score']:.3f}

Regional Distribution:
• Upper region: {metrics['upper_ratio']:.1%}
• Middle region: {metrics['middle_ratio']:.1%}
• Lower region: {metrics['lower_ratio']:.1%}

Technical Info:
• No Ground Truth available
• Uses pretrained DQN model
• HSV-guided detection
• Device: {device}

Assessment: {city_name} coastline detection 
{"completed successfully" if result['success'] else "needs refinement"}
with {"excellent" if metrics['overall_score'] > 0.8 else "good" if metrics['overall_score'] > 0.65 else "acceptable" if metrics['overall_score'] > 0.5 else "poor"} quality metrics."""

    # 添加统计文本到图形
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
                verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ {city_name} 可视化已保存: {save_path}")


# ==================== 批量处理函数 ====================

def process_all_uk_cities():
    """批量处理所有英国城市"""
    print("🇬🇧 开始批量处理英国其他城市海岸线...")
    print("=" * 80)

    # 路径设置
    cities_dir = "E:/Other"
    output_dir = "./uk_cities_results"
    os.makedirs(output_dir, exist_ok=True)

    # 预训练模型路径
    model_paths = [
        "./saved_models/coastline_general_model.pth",
        "./saved_models/coastline_dqn_model.pth"
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

    # 创建检测器
    detector = UKCitiesCoastlineDetector()

    # 处理结果
    results = []
    successful_count = 0
    failed_count = 0

    # 逐个处理城市
    for i, city_file in enumerate(city_files):
        print(f"\n{'='*60}")
        print(f"🔄 处理城市 {i+1}/{len(city_files)}: {city_file}")
        print(f"{'='*60}")

        # 提取城市名称
        city_name = os.path.splitext(city_file)[0]
        city_path = os.path.join(cities_dir, city_file)

        try:
            # 处理单个城市
            result = detector.process_uk_city(
                image_path=city_path,
                city_name=city_name,
                pretrained_model_path=pretrained_model_path
            )

            if result and result['success']:
                successful_count += 1

                # 保存可视化结果
                vis_filename = f"{city_name}_coastline_detection.png"
                vis_path = os.path.join(output_dir, vis_filename)
                create_uk_cities_visualization(result, vis_path)

                # 保存数值结果
                save_city_metrics(result, output_dir)

                # 记录结果摘要
                results.append({
                    'city_name': city_name,
                    'file': city_file,
                    'success': True,
                    'overall_score': result['quality_metrics']['overall_score'],
                    'quality_level': result['quality_metrics']['quality_level'],
                    'coastline_pixels': result['quality_metrics']['coastline_pixels'],
                    'num_components': result['quality_metrics']['num_components'],
                    'sea_cleanup_score': result['quality_metrics']['sea_cleanup_score'],
                    'distribution_score': result['quality_metrics']['distribution_score']
                })

                print(f"✅ {city_name} 处理成功!")

            else:
                failed_count += 1
                results.append({
                    'city_name': city_name,
                    'file': city_file,
                    'success': False
                })
                print(f"❌ {city_name} 处理失败")

        except Exception as e:
            failed_count += 1
            results.append({
                'city_name': city_name,
                'file': city_file,
                'success': False,
                'error': str(e)
            })
            print(f"❌ 处理 {city_name} 时出错: {e}")

    # 生成批量处理报告
    generate_uk_cities_report(results, output_dir, successful_count, failed_count)

    print(f"\n{'='*80}")
    print(f"🎉 英国城市批量处理完成!")
    print(f"   ✅ 成功: {successful_count} 个城市")
    print(f"   ❌ 失败: {failed_count} 个城市")
    print(f"   📁 结果保存在: {output_dir}")
    print(f"{'='*80}")

    return results


def save_city_metrics(result, output_dir):
    """保存城市指标数据"""
    import json

    city_name = result['city_name']
    metrics_data = {
        'city_name': city_name,
        'processing_info': {
            'success': result['success'],
            'model_path': result['model_path'],
            'image_shape': result['processed_image'].shape,
            'processing_time': get_current_time()
        },
        'quality_metrics': result['quality_metrics'],
        'analysis_details': {
            'hsv_water_coverage': float(np.sum(result['hsv_analysis']['water_mask']) / (400 * 400)),
            'hsv_land_coverage': float(np.sum(result['hsv_analysis']['land_mask']) / (400 * 400)),
            'guidance_strength': float(np.mean(result['hsv_analysis']['coastline_guidance'])),
            'transition_strength': float(np.mean(result['hsv_analysis']['transition_strength']))
        }
    }

    # 保存JSON文件
    json_filename = f"{city_name}_metrics.json"
    json_path = os.path.join(output_dir, json_filename)

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 {city_name} 指标已保存: {json_filename}")
    except Exception as e:
        print(f"   ⚠️ 保存 {city_name} 指标失败: {e}")


def generate_uk_cities_report(results, output_dir, successful_count, failed_count):
    """生成英国城市批量处理报告"""
    import json
    from datetime import datetime

    # 创建汇总报告
    report = {
        'uk_cities_processing_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_cities': successful_count + failed_count,
            'successful_cities': successful_count,
            'failed_cities': failed_count,
            'success_rate': successful_count / (successful_count + failed_count) if (successful_count + failed_count) > 0 else 0,
            'target_cities': ['Blackpool', 'Liverpool', 'Ortsmouth', 'Southport']
        },
        'detailed_results': results
    }

    # 计算统计信息
    successful_results = [r for r in results if r.get('success', False)]

    if successful_results:
        overall_scores = [r['overall_score'] for r in successful_results]
        coastline_pixels = [r['coastline_pixels'] for r in successful_results]
        components = [r['num_components'] for r in successful_results]

        report['statistics'] = {
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
            'connectivity': {
                'mean_components': float(np.mean(components)),
                'max_components': int(np.max(components)),
                'min_components': int(np.min(components))
            }
        }

        # 质量等级分布
        quality_levels = [r['quality_level'] for r in successful_results]
        level_counts = {}
        for level in quality_levels:
            level_counts[level] = level_counts.get(level, 0) + 1

        report['quality_distribution'] = level_counts

    # 保存报告
    report_path = os.path.join(output_dir, 'uk_cities_processing_report.json')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   📋 批量处理报告已保存: uk_cities_processing_report.json")
    except Exception as e:
        print(f"   ⚠️ 保存报告失败: {e}")

    # 生成CSV报告
    generate_uk_cities_csv_report(successful_results, output_dir)

    # 生成可读性报告
    generate_readable_summary(successful_results, output_dir)


def generate_uk_cities_csv_report(results, output_dir):
    """生成CSV格式报告"""
    import csv

    if not results:
        return

    csv_path = os.path.join(output_dir, 'uk_cities_summary.csv')

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'city_name', 'overall_score', 'quality_level', 'coastline_pixels',
                'num_components', 'sea_cleanup_score', 'distribution_score',
                'continuity_score', 'guidance_conformity', 'edge_quality'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # 从原始结果中提取所需字段
                row = {
                    'city_name': result['city_name'],
                    'overall_score': result['overall_score'],
                    'quality_level': result['quality_level'],
                    'coastline_pixels': result['coastline_pixels'],
                    'num_components': result['num_components'],
                    'sea_cleanup_score': result['sea_cleanup_score'],
                    'distribution_score': result['distribution_score'],
                    'continuity_score': result.get('continuity_score', 'N/A'),
                    'guidance_conformity': result.get('guidance_conformity', 'N/A'),
                    'edge_quality': result.get('edge_quality', 'N/A')
                }
                writer.writerow(row)

        print(f"   📊 CSV报告已保存: uk_cities_summary.csv")
    except Exception as e:
        print(f"   ⚠️ 保存CSV报告失败: {e}")


def generate_readable_summary(results, output_dir):
    """生成可读性总结报告"""
    if not results:
        return

    summary_path = os.path.join(output_dir, 'UK_Cities_Summary_Report.txt')

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("🇬🇧 英国其他城市海岸线检测总结报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"处理时间: {get_current_time()}\n")
            f.write(f"处理城市数量: {len(results)}\n")
            f.write(f"目标城市: Blackpool, Liverpool, Ortsmouth, Southport\n\n")

            # 总体统计
            scores = [r['overall_score'] for r in results]
            pixels = [r['coastline_pixels'] for r in results]

            f.write("📊 总体统计:\n")
            f.write(f"   平均质量得分: {np.mean(scores):.3f}\n")
            f.write(f"   得分范围: {np.min(scores):.3f} - {np.max(scores):.3f}\n")
            f.write(f"   平均海岸线像素: {np.mean(pixels):,.0f}\n")
            f.write(f"   像素范围: {np.min(pixels):,} - {np.max(pixels):,}\n\n")

            # 质量等级分布
            quality_levels = [r['quality_level'] for r in results]
            level_counts = {}
            for level in quality_levels:
                level_counts[level] = level_counts.get(level, 0) + 1

            f.write("🏆 质量等级分布:\n")
            for level, count in sorted(level_counts.items()):
                f.write(f"   {level}: {count} 个城市\n")
            f.write("\n")

            # 逐城市详细结果
            f.write("🏙️ 逐城市详细结果:\n")
            f.write("-" * 60 + "\n")

            # 按得分排序
            sorted_results = sorted(results, key=lambda x: x['overall_score'], reverse=True)

            for i, result in enumerate(sorted_results, 1):
                f.write(f"\n{i}. {result['city_name']}\n")
                f.write(f"   质量得分: {result['overall_score']:.3f} ({result['quality_level']})\n")
                f.write(f"   海岸线像素: {result['coastline_pixels']:,}\n")
                f.write(f"   连通组件: {result['num_components']}\n")
                f.write(f"   海域清理: {result['sea_cleanup_score']:.3f}\n")
                f.write(f"   区域分布: {result['distribution_score']:.3f}\n")

                # 状态评估
                score = result['overall_score']
                if score >= 0.8:
                    status = "✅ 优秀"
                elif score >= 0.65:
                    status = "✅ 良好"
                elif score >= 0.5:
                    status = "⚠️ 一般"
                else:
                    status = "❌ 需改进"

                f.write(f"   状态: {status}\n")

            # 技术说明
            f.write(f"\n" + "=" * 60 + "\n")
            f.write("🔧 技术说明:\n")
            f.write("• 使用预训练DQN模型进行海岸线检测\n")
            f.write("• 基于HSV颜色空间进行水陆分割指导\n")
            f.write("• 无Ground Truth参考，依赖质量评估指标\n")
            f.write("• 重点关注海域清理和连通性控制\n")
            f.write("• 目标像素范围：80,000 - 110,000\n")
            f.write(f"• 运行设备: {device}\n")

        print(f"   📖 可读性报告已保存: UK_Cities_Summary_Report.txt")
    except Exception as e:
        print(f"   ⚠️ 保存可读性报告失败: {e}")


def get_current_time():
    """获取当前时间字符串"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== 快速测试函数 ====================

def quick_test_single_city():
    """快速测试单个城市"""
    print("🧪 快速测试单个英国城市...")

    # 路径设置
    cities_dir = "E:/Other"
    output_dir = "./quick_test_uk"
    os.makedirs(output_dir, exist_ok=True)

    # 查找预训练模型
    model_paths = [
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
    test_file = city_files[0]
    city_name = os.path.splitext(test_file)[0]
    city_path = os.path.join(cities_dir, test_file)

    print(f"📁 测试城市: {city_name}")
    print(f"📁 文件路径: {city_path}")
    print(f"🤖 模型路径: {pretrained_model_path}")

    # 创建检测器并处理
    detector = UKCitiesCoastlineDetector()
    result = detector.process_uk_city(city_path, city_name, pretrained_model_path)

    if result:
        # 保存结果
        vis_path = os.path.join(output_dir, f"{city_name}_test_result.png")
        create_uk_cities_visualization(result, vis_path)

        save_city_metrics(result, output_dir)

        print(f"\n🎉 {city_name} 测试完成!")
        print(f"   📊 质量得分: {result['quality_metrics']['overall_score']:.3f}")
        print(f"   🏆 质量等级: {result['quality_metrics']['quality_level']}")
        print(f"   📁 结果保存在: {output_dir}")

        return result
    else:
        print(f"❌ {city_name} 测试失败")
        return None


# ==================== 主函数 ====================

def main():
    """主函数 - 英国城市海岸线检测"""
    print("🚀 启动英国其他城市海岸线检测系统...")
    print("\n请选择测试模式:")
    print("1. 快速测试单个城市")
    print("2. 批量处理所有城市")
    print("3. 查看已有结果")

    choice = input("请输入选择 (1-3): ").strip()

    if choice == "1":
        print("\n🧪 快速测试模式")
        result = quick_test_single_city()
        if result:
            print("\n✅ 快速测试完成!")

    elif choice == "2":
        print("\n🏭 批量处理模式")
        results = process_all_uk_cities()
        if results:
            successful = [r for r in results if r.get('success', False)]
            print(f"\n📈 批量处理汇总:")
            print(f"   成功处理: {len(successful)} 个城市")
            if successful:
                avg_score = np.mean([r['overall_score'] for r in successful])
                print(f"   平均得分: {avg_score:.3f}")

                print(f"   城市列表:")
                for r in successful:
                    print(f"      {r['city_name']}: {r['overall_score']:.3f} ({r['quality_level']})")

    elif choice == "3":
        print("\n📊 查看已有结果")
        result_dirs = ["./uk_cities_results", "./quick_test_uk"]

        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                files = os.listdir(result_dir)
                png_files = [f for f in files if f.endswith('.png')]
                json_files = [f for f in files if f.endswith('.json')]

                if png_files or json_files:
                    print(f"\n📁 {result_dir}:")
                    print(f"   可视化文件: {len(png_files)} 个")
                    print(f"   数据文件: {len(json_files)} 个")

                    for png_file in png_files[:3]:  # 显示前3个
                        print(f"      📸 {png_file}")

                    if len(png_files) > 3:
                        print(f"      ... 还有 {len(png_files) - 3} 个文件")
    else:
        print("❌ 无效选择")


# ==================== 直接执行测试函数 ====================

def test_uk_cities_directly():
    """直接执行英国城市测试（无交互）"""
    print("🇬🇧 直接执行英国城市海岸线检测测试...")

    # 首先尝试快速测试
    print("\n📍 步骤1: 快速测试单个城市")
    quick_result = quick_test_single_city()

    if quick_result:
        print("\n📍 步骤2: 批量处理所有城市")
        batch_results = process_all_uk_cities()

        if batch_results:
            successful = [r for r in batch_results if r.get('success', False)]
            print(f"\n🎉 英国城市检测完成!")
            print(f"   成功处理: {len(successful)} 个城市")

            if successful:
                avg_score = np.mean([r['overall_score'] for r in successful])
                best_city = max(successful, key=lambda x: x['overall_score'])

                print(f"   平均质量得分: {avg_score:.3f}")
                print(f"   最佳城市: {best_city['city_name']} (得分: {best_city['overall_score']:.3f})")

                return {
                    'quick_test': quick_result,
                    'batch_results': batch_results,
                    'summary': {
                        'total_successful': len(successful),
                        'average_score': avg_score,
                        'best_city': best_city
                    }
                }

    return None


if __name__ == "__main__":
    # 可以选择交互式或直接执行

    # 方式1: 交互式菜单
    # main()

    # 方式2: 直接执行测试
    test_uk_cities_directly()