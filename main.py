import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from PIL import Image
import fitz  # PyMuPDF
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleImageProcessor:
    """超简化图像处理器"""

    @staticmethod
    def resize_image(image, target_size):
        """调整图像大小"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                pil_img = Image.fromarray(image.astype(np.uint8))
            else:
                pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = image

        resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)

    @staticmethod
    def rgb_to_gray(image):
        """RGB转灰度"""
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            return gray.astype(image.dtype)
        return image

    @staticmethod
    def simple_threshold(image, low, high):
        """简单阈值处理"""
        result = np.zeros_like(image)
        mask = (image >= low) & (image <= high)
        result[mask] = 255
        return result

    @staticmethod
    def simple_blur(image, kernel_size=5):
        """简单模糊（均值滤波）"""
        from scipy import ndimage
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return ndimage.convolve(image.astype(np.float32), kernel).astype(np.uint8)

    @staticmethod
    def simple_edge_detection(image):
        """简单边缘检测（梯度）"""
        grad_x = np.abs(np.gradient(image.astype(np.float32))[1])
        grad_y = np.abs(np.gradient(image.astype(np.float32))[0])
        edges = grad_x + grad_y
        edges = (edges / edges.max() * 255).astype(np.uint8)
        return edges


class CoastalDataProcessor:
    """简化数据处理器"""

    def __init__(self, initial_dir="E:/initial", ground_dir="E:/ground"):
        self.initial_dir = initial_dir
        self.ground_dir = ground_dir
        self.target_size = (256, 256)  # 更小的尺寸，更快处理

        os.makedirs(initial_dir, exist_ok=True)
        os.makedirs(ground_dir, exist_ok=True)

        logger.info(f"简化数据处理器初始化完成")

    def pdf_to_image(self, pdf_path):
        """PDF转图像"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)

            zoom = 150 / 72  # 降低DPI，更快处理
            mat = fitz.Matrix(zoom, zoom)

            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            from io import BytesIO
            img = Image.open(BytesIO(img_data))
            img_array = np.array(img)

            doc.close()
            return img_array

        except Exception as e:
            logger.error(f"PDF转换失败 {pdf_path}: {e}")
            return None

    def preprocess_image(self, image, is_ground_truth=False):
        """预处理图像"""
        if image is None:
            return None

        # 调整大小
        image = SimpleImageProcessor.resize_image(image, self.target_size)

        if is_ground_truth:
            # 简化的标签处理
            image = self.process_ground_truth_simple(image)
        else:
            # 归一化
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0

        return image

    def process_ground_truth_simple(self, gt_image):
        """简化的标签处理"""
        if len(gt_image.shape) == 2:
            return gt_image

        height, width = gt_image.shape[:2]
        label_mask = np.zeros((height, width), dtype=np.uint8)

        # 简化的颜色检测
        if gt_image.shape[2] >= 3:
            # 检测蓝色（海洋）
            blue_mask = (gt_image[:, :, 2] > gt_image[:, :, 0]) & (gt_image[:, :, 2] > gt_image[:, :, 1])
            # 检测绿色（植被）
            green_mask = (gt_image[:, :, 1] > gt_image[:, :, 0]) & (gt_image[:, :, 1] > gt_image[:, :, 2])
            # 检测白色（边界）
            white_mask = (gt_image[:, :, 0] > 200) & (gt_image[:, :, 1] > 200) & (gt_image[:, :, 2] > 200)

            label_mask[blue_mask] = 1  # 海洋
            label_mask[green_mask] = 2  # 植被
            label_mask[white_mask] = 3  # 边界

        return label_mask

    def load_dataset(self):
        """加载数据集"""
        initial_files = [f for f in os.listdir(self.initial_dir) if f.endswith('.pdf')]
        ground_files = [f for f in os.listdir(self.ground_dir) if f.endswith('.pdf')]

        logger.info(f"找到 {len(initial_files)} 个原始PDF, {len(ground_files)} 个标签PDF")

        dataset = []
        for initial_file in initial_files:
            year = self.extract_year(initial_file)
            ground_file = self.find_ground_file(year, ground_files)
            if ground_file:
                dataset.append((initial_file, ground_file, year))

        logger.info(f"匹配 {len(dataset)} 对图像")
        return dataset

    def extract_year(self, filename):
        import re
        years = re.findall(r'20\d{2}', filename)
        return years[0] if years else None

    def find_ground_file(self, year, ground_files):
        for ground_file in ground_files:
            if year and year in ground_file:
                return ground_file
        return None


class SimpleTMC:
    """超简化TMC处理器"""

    def __init__(self):
        self.param_ranges = {
            'threshold_low': (0.2, 0.4),
            'threshold_high': (0.6, 0.8),
            'blur_size': (3, 9),
            'edge_threshold': (50, 150)
        }

    def run_tmc(self, image, params):
        """运行简化TMC"""
        try:
            # 转灰度
            if len(image.shape) == 3:
                gray = SimpleImageProcessor.rgb_to_gray(image)
            else:
                gray = image.copy()

            # 确保0-255范围
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)

            # 1. 模糊
            blur_size = max(3, int(params.get('blur_size', 5)))
            if blur_size % 2 == 0:
                blur_size += 1
            blurred = SimpleImageProcessor.simple_blur(gray, blur_size)

            # 2. 阈值
            low = int(params.get('threshold_low', 0.3) * 255)
            high = int(params.get('threshold_high', 0.7) * 255)
            binary = SimpleImageProcessor.simple_threshold(blurred, low, high)

            # 3. 边缘检测
            edges = SimpleImageProcessor.simple_edge_detection(binary)

            # 4. 简单的轮廓处理
            threshold = params.get('edge_threshold', 100)
            result = (edges > threshold).astype(np.float32)

            return result

        except Exception as e:
            logger.error(f"简化TMC处理失败: {e}")
            return np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image).astype(np.float32)


class SimpleCNN(nn.Module):
    """简化CNN网络"""

    def __init__(self, input_channels=3, num_params=4):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.param_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, num_params),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        params = self.param_predictor(features)
        return params


class SimpleDQN(nn.Module):
    """简化DQN网络"""

    def __init__(self, state_dim=15, action_dim=12):  # 减少状态和动作维度
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.network(state)


class SimpleDQNAgent:
    """简化DQN智能体"""

    def __init__(self, state_dim=15, action_dim=12):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1

        self.q_network = SimpleDQN(state_dim, action_dim)
        self.target_network = SimpleDQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.memory = deque(maxlen=1000)
        self.batch_size = 8

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([torch.FloatTensor(e[0]) for e in batch])
        actions = torch.tensor([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32)
        next_states = torch.stack([torch.FloatTensor(e[3]) for e in batch])
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


class SimpleFeatureExtractor:
    """简化特征提取器"""

    def extract_image_features(self, image):
        """提取5个简单图像特征"""
        if len(image.shape) == 3:
            gray = SimpleImageProcessor.rgb_to_gray(image)
        else:
            gray = image.copy()

        if gray.max() > 1.0:
            gray = gray.astype(np.float32) / 255.0

        features = [
            np.mean(gray),  # 均值亮度
            np.std(gray),  # 标准差
            np.percentile(gray, 25),  # 25%分位数
            np.percentile(gray, 75),  # 75%分位数
            np.var(gray)  # 方差
        ]

        return np.array(features, dtype=np.float32)

    def calculate_quality_metrics(self, segmentation, ground_truth=None):
        """计算5个简单质量指标"""
        if segmentation is None:
            return np.zeros(5, dtype=np.float32)

        metrics = [
            self.calculate_simple_accuracy(segmentation, ground_truth) if ground_truth is not None else 0.7,
            np.mean(segmentation),  # 分割区域比例
            np.std(segmentation),  # 分割一致性
            np.sum(segmentation > 0.5) / segmentation.size,  # 前景比例
            1.0 - np.var(segmentation)  # 稳定性
        ]

        return np.array(metrics, dtype=np.float32)

    def calculate_simple_accuracy(self, pred, gt):
        """简单准确率计算"""
        if gt is None:
            return 0.7

        pred_binary = (pred > 0.5).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)

        correct = np.sum(pred_binary == gt_binary)
        total = pred_binary.size

        return correct / total


class SimpleCoastalTrainer:
    """简化训练器"""

    def __init__(self, data_dir_initial="E:/initial", data_dir_ground="E:/ground"):
        self.data_processor = CoastalDataProcessor(data_dir_initial, data_dir_ground)
        self.tmc_processor = SimpleTMC()
        self.feature_extractor = SimpleFeatureExtractor()

        # 简化网络
        self.cnn_model = SimpleCNN(input_channels=3, num_params=4)
        self.dqn_agent = SimpleDQNAgent(state_dim=15, action_dim=12)  # 4参数+5图像特征+5质量=14, 向上取15

        self.cnn_optimizer = optim.Adam(self.cnn_model.parameters(), lr=1e-3)
        self.cnn_criterion = nn.MSELoss()

        logger.info("简化训练器初始化完成")

    def denormalize_params(self, normalized_params):
        """反归一化参数"""
        param_names = list(self.tmc_processor.param_ranges.keys())
        params = {}

        for i, (param_name, (min_val, max_val)) in enumerate(self.tmc_processor.param_ranges.items()):
            if i < len(normalized_params):
                normalized_val = normalized_params[i]
                actual_val = min_val + normalized_val * (max_val - min_val)
                params[param_name] = actual_val

        return params

    def get_state(self, image, current_params, prev_segmentation=None):
        """提取状态向量"""
        # 参数状态 (4维)
        param_state = np.array(list(current_params.values())[:4], dtype=np.float32)
        param_ranges = list(self.tmc_processor.param_ranges.values())[:4]

        # 归一化参数
        for i, (min_val, max_val) in enumerate(param_ranges):
            if i < len(param_state):
                param_state[i] = (param_state[i] - min_val) / (max_val - min_val)

        # 图像特征 (5维)
        image_features = self.feature_extractor.extract_image_features(image)

        # 质量指标 (5维)
        quality_metrics = self.feature_extractor.calculate_quality_metrics(prev_segmentation)

        # 拼接状态 (4+5+5=14维，填充到15维)
        state = np.concatenate([param_state, image_features, quality_metrics])
        if len(state) < 15:
            state = np.pad(state, (0, 15 - len(state)), 'constant')

        return state.astype(np.float32)

    def apply_action(self, current_params, action):
        """应用动作"""
        # 简化的动作映射：4个参数 × 3种操作 = 12个动作
        param_names = list(self.tmc_processor.param_ranges.keys())
        param_idx = action // 3
        adjustment = (action % 3) - 1  # -1, 0, +1

        new_params = current_params.copy()

        if param_idx < len(param_names):
            param_name = param_names[param_idx]
            min_val, max_val = self.tmc_processor.param_ranges[param_name]

            if param_name in ['threshold_low', 'threshold_high']:
                step = 0.05
            elif param_name == 'blur_size':
                step = 2
            else:
                step = 20

            new_value = current_params[param_name] + adjustment * step
            new_params[param_name] = np.clip(new_value, min_val, max_val)

        return new_params

    def calculate_reward(self, segmentation, ground_truth):
        """计算简单奖励"""
        accuracy = self.feature_extractor.calculate_simple_accuracy(segmentation, ground_truth)
        consistency = 1.0 - np.std(segmentation)
        coverage = np.mean(segmentation)

        reward = accuracy + 0.5 * consistency + 0.3 * coverage
        return reward

    def train_sample(self, initial_file, ground_file, year):
        """训练单个样本"""
        try:
            # 加载图像
            initial_path = os.path.join(self.data_processor.initial_dir, initial_file)
            ground_path = os.path.join(self.data_processor.ground_dir, ground_file)

            initial_img = self.data_processor.pdf_to_image(initial_path)
            ground_img = self.data_processor.pdf_to_image(ground_path)

            if initial_img is None or ground_img is None:
                return False

            # 预处理
            initial_processed = self.data_processor.preprocess_image(initial_img, False)
            ground_processed = self.data_processor.preprocess_image(ground_img, True)

            if initial_processed is None or ground_processed is None:
                return False

            # 转换为tensor
            if len(initial_processed.shape) == 2:
                initial_processed = np.stack([initial_processed] * 3, axis=2)

            input_tensor = torch.FloatTensor(initial_processed).permute(2, 0, 1).unsqueeze(0)

            # CNN训练
            self.cnn_model.train()
            predicted_params = self.cnn_model(input_tensor)
            params_dict = self.denormalize_params(predicted_params.squeeze().detach().numpy())

            # 运行TMC
            tmc_result = self.tmc_processor.run_tmc(initial_processed, params_dict)

            # CNN损失
            target_quality = self.calculate_reward(tmc_result, ground_processed)
            target_tensor = torch.FloatTensor([target_quality])
            predicted_quality = torch.mean(predicted_params)

            cnn_loss = self.cnn_criterion(predicted_quality.unsqueeze(0), target_tensor)

            self.cnn_optimizer.zero_grad()
            cnn_loss.backward()
            self.cnn_optimizer.step()

            # DQN训练
            current_params = params_dict.copy()
            total_reward = 0

            for step in range(5):  # 5步DQN优化
                state = self.get_state(initial_processed, current_params)
                action = self.dqn_agent.act(state)
                new_params = self.apply_action(current_params, action)

                segmentation_result = self.tmc_processor.run_tmc(initial_processed, new_params)
                reward = self.calculate_reward(segmentation_result, ground_processed)
                next_state = self.get_state(initial_processed, new_params, segmentation_result)

                done = (step == 4)
                self.dqn_agent.remember(state, action, reward, next_state, done)

                current_params = new_params
                total_reward += reward

            # DQN更新
            dqn_loss = self.dqn_agent.replay()

            logger.info(
                f"样本 {year}: CNN Loss={cnn_loss.item():.4f}, DQN奖励={total_reward:.3f}, 质量={target_quality:.3f}")
            return True

        except Exception as e:
            logger.error(f"训练样本失败 {initial_file}: {e}")
            return False

    def train(self, epochs=10):
        """简化训练"""
        dataset = self.data_processor.load_dataset()
        if len(dataset) == 0:
            logger.error("没有找到数据集")
            return

        logger.info(f"开始简化训练，数据集大小: {len(dataset)}")

        for epoch in range(epochs):
            successful = 0

            for initial_file, ground_file, year in dataset:
                if self.train_sample(initial_file, ground_file, year):
                    successful += 1

            # 更新DQN目标网络
            if epoch % 5 == 0:
                self.dqn_agent.update_target_network()

            logger.info(f"Epoch {epoch + 1}/{epochs}: 成功训练 {successful}/{len(dataset)} 个样本")

        logger.info("训练完成！")


def main():
    print("简化版 DQN-Enhanced TMC海岸线监测系统")
    print("=" * 50)

    # 检查文件夹
    initial_dir = "E:/initial"
    ground_dir = "E:/ground"

    os.makedirs(initial_dir, exist_ok=True)
    os.makedirs(ground_dir, exist_ok=True)

    # 检查数据
    initial_files = [f for f in os.listdir(initial_dir) if f.endswith('.pdf')]
    ground_files = [f for f in os.listdir(ground_dir) if f.endswith('.pdf')]

    print(f"找到 {len(initial_files)} 个原始PDF文件")
    print(f"找到 {len(ground_files)} 个标签PDF文件")

    if len(initial_files) == 0 or len(ground_files) == 0:
        print("\n请将PDF文件放入对应文件夹:")
        print(f"原始图像: {initial_dir}")
        print(f"标签图像: {ground_dir}")
        return

    # 初始化简化训练器
    trainer = SimpleCoastalTrainer(initial_dir, ground_dir)

    print("\n开始简化训练（10轮）...")
    trainer.train(epochs=10)

    print("训练完成！")


if __name__ == "__main__":
    try:
        import fitz
        import torch
        from scipy import ndimage

        print("✓ 所有依赖已安装")
        print("\n" + "=" * 50)
        main()
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install PyMuPDF torch scipy pillow numpy scikit-learn")


