"""
Coastline Prediction Tool - 基于白色海岸线检测和Transects
- 检测白色海岸线，提取下半部分坐标
- 离散化坐标作为Transects
- 用2015-2020年数据训练，预测2021-2025年
- 可视化GT与预测海岸线对比（含重合区域显示）

核心思路：
    1. 从PDF/图像中提取白线作为海岸线GT
    2. 将白线离散化为若干个transect点 (x, y)
    3. 使用前几年(2015-2020)的transect点训练时序模型
    4. 预测后续年份(2021-2025)的海岸线位置
    5. 对比GT与预测，重合部分用特殊颜色标记

使用方法：
    python coastline_prediction_transects_v2.py --input_path <输入目录> --output_path <输出目录>
"""

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 尝试导入PDF处理库
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    try:
        from pdf2image import convert_from_path
        PDF_SUPPORT = 'pdf2image'
    except ImportError:
        PDF_SUPPORT = False

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 图像读取函数 ====================

def read_image_file(image_path):
    """读取图像文件，支持常规图片格式和PDF"""
    ext = os.path.splitext(image_path)[1].lower()

    if ext == '.pdf':
        if PDF_SUPPORT == True:
            doc = fitz.open(image_path)
            page = doc[0]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img_rgb = img_rgb[:, :, :3]
            doc.close()
            return img_rgb
        elif PDF_SUPPORT == 'pdf2image':
            images = convert_from_path(image_path, dpi=150)
            img_rgb = np.array(images[0])
            if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 4:
                img_rgb = img_rgb[:, :, :3]
            return img_rgb
        else:
            raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")
    else:
        img_rgb = plt.imread(image_path)
        if img_rgb.dtype == np.float32 or img_rgb.dtype == np.float64:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]
        return img_rgb


# ==================== 白色海岸线检测函数 ====================

def detect_white_coastline(image_path, num_transects=50):
    """
    检测白色海岸线，提取下半部分坐标，并离散化为Transects

    核心逻辑：
        1. 读取PDF/图像
        2. 检测白色/亮色像素（这就是海岸线GT）
        3. 形态学处理，连接断开的线段
        4. 将连续的海岸线离散化为num_transects个点

    参数:
        image_path: 图像路径
        num_transects: 离散化的transect数量

    返回:
        img_rgb: 原始图像
        coastline_coords: 海岸线坐标点 [(x, y), ...]
        transects: 离散化的transect坐标 [(x, y), ...]
        coastline_img: 带海岸线标注的图像
        coastline_mask: 海岸线掩膜
    """
    img_rgb = read_image_file(image_path)
    height, width = img_rgb.shape[:2]

    # 检测白色/亮色像素（海岸线）
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # 白色检测阈值
    white_threshold = 200
    white_mask = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)

    # 也检测浅色（沙滩色）
    brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
    bright_mask = brightness > 180

    # 合并
    coastline_mask = white_mask | bright_mask

    # 只保留下半部分（从25%高度开始）
    cutoff_line = int(height * 0.25)
    coastline_mask[:cutoff_line, :] = False

    # 形态学操作：连接断开的线段
    kernel = np.ones((3, 3), bool)
    coastline_mask = ndimage.binary_closing(coastline_mask, structure=kernel)
    coastline_mask = ndimage.binary_opening(coastline_mask, structure=kernel)

    # 获取所有海岸线坐标点
    coords = np.where(coastline_mask)
    if len(coords[0]) == 0:
        # 如果没检测到白色，尝试降低阈值
        white_threshold = 170
        white_mask = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
        bright_mask = brightness > 160
        coastline_mask = white_mask | bright_mask
        coastline_mask[:cutoff_line, :] = False
        coastline_mask = ndimage.binary_closing(coastline_mask, structure=kernel)
        coords = np.where(coastline_mask)

    coastline_coords = list(zip(coords[1], coords[0]))  # (x, y) 格式

    # 离散化为Transects：沿x轴均匀采样
    transects = extract_transects(coastline_mask, num_transects, height, width)

    # 创建可视化图像
    coastline_img = img_rgb.copy()
    coastline_img[coastline_mask] = [0, 255, 0]

    return img_rgb, coastline_coords, transects, coastline_img, coastline_mask


def extract_transects(coastline_mask, num_transects, height, width):
    """
    从海岸线掩膜中提取离散化的Transects
    沿x轴均匀采样，每个x位置取海岸线的y坐标

    返回: [(x, y), ...] 离散化的坐标点
    """
    transects = []

    # 沿x轴均匀采样
    x_positions = np.linspace(0, width - 1, num_transects, dtype=int)

    for x in x_positions:
        # 获取该x位置上的所有海岸线y坐标
        y_coords = np.where(coastline_mask[:, x])[0]

        if len(y_coords) > 0:
            # 取平均值作为该x位置的海岸线y坐标
            y = int(np.mean(y_coords))
            transects.append((x, y))

    return transects


# ==================== 时间序列预测模型 ====================

class TransectPredictor:
    """
    基于Transects的海岸线预测模型

    核心思路：
        - 每个x位置的y坐标形成一个时间序列
        - 对每个x位置单独训练一个Ridge回归模型
        - 使用过去seq_length年的y值预测下一年的y值
    """
    def __init__(self, seq_length=3):
        self.seq_length = seq_length
        self.models = {}  # 每个x位置一个模型
        self.scalers = {}

    def fit(self, yearly_transects):
        """
        训练模型
        yearly_transects: {year: [(x, y), ...], ...}
        """
        years = sorted(yearly_transects.keys())

        # 获取所有x位置
        all_x = set()
        for year in years:
            for x, y in yearly_transects[year]:
                all_x.add(x)

        # 对每个x位置建立时间序列模型
        for x_pos in all_x:
            # 收集该位置的时间序列数据
            y_series = []
            for year in years:
                y_val = None
                for x, y in yearly_transects[year]:
                    if x == x_pos:
                        y_val = y
                        break
                if y_val is not None:
                    y_series.append(y_val)

            if len(y_series) >= self.seq_length + 1:
                # 训练该位置的模型
                y_array = np.array(y_series, dtype=float)

                # 创建特征: 使用过去seq_length个y值预测下一个y值
                X, y_target = [], []
                for i in range(self.seq_length, len(y_array)):
                    X.append(y_array[i-self.seq_length:i])
                    y_target.append(y_array[i])

                X = np.array(X)
                y_target = np.array(y_target)

                # 标准化
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 训练Ridge回归
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, y_target)

                self.models[x_pos] = model
                self.scalers[x_pos] = scaler

        self.yearly_transects = yearly_transects
        self.years = years

    def predict(self, num_years=5):
        """
        预测未来num_years年的海岸线
        返回: {year_offset: [(x, y), ...], ...}
        """
        predictions = {}

        for offset in range(1, num_years + 1):
            pred_transects = []

            for x_pos in sorted(self.models.keys()):
                # 获取该位置的历史数据
                y_series = []
                for year in self.years:
                    for x, y in self.yearly_transects[year]:
                        if x == x_pos:
                            y_series.append(y)
                            break

                # 加上之前的预测值
                for prev_offset in range(1, offset):
                    if prev_offset in predictions:
                        for x, y in predictions[prev_offset]:
                            if x == x_pos:
                                y_series.append(y)
                                break

                if len(y_series) >= self.seq_length:
                    # 使用最近的seq_length个值预测
                    recent = np.array(y_series[-self.seq_length:]).reshape(1, -1)
                    recent_scaled = self.scalers[x_pos].transform(recent)
                    pred_y = self.models[x_pos].predict(recent_scaled)[0]
                    pred_transects.append((x_pos, int(pred_y)))

            predictions[offset] = pred_transects

        return predictions


# ==================== 可视化函数 ====================

def visualize_segmentation_results(results_dict, output_dir, title_prefix="训练"):
    """可视化分割/检测结果"""
    years = sorted(results_dict.keys())
    n_years = len(years)

    fig, axes = plt.subplots(n_years, 3, figsize=(15, 4*n_years))
    if n_years == 1:
        axes = axes.reshape(1, -1)

    for i, year in enumerate(years):
        data = results_dict[year]

        axes[i, 0].imshow(data['image'])
        axes[i, 0].set_title(f'{year} - Original', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(data['mask'], cmap='Blues')
        axes[i, 1].set_title(f'{year} - Coastline Detection', fontsize=12)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(data['coastline_img'])
        axes[i, 2].set_title(f'{year} - Coastline (Transects: {len(data["transects"])})', fontsize=12)
        axes[i, 2].axis('off')

    plt.suptitle(f'{title_prefix} Data - Coastline Detection Results', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(output_dir, f'segmentation_results_{title_prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_transects_comparison(train_data, gt_data, predictions, pred_years, output_dir):
    """可视化Transects预测对比 - 时间序列图"""
    train_years = sorted(train_data.keys())

    train_avg_y = []
    for year in train_years:
        transects = train_data[year]['transects']
        if transects:
            avg_y = np.mean([t[1] for t in transects])
            train_avg_y.append(avg_y)
        else:
            train_avg_y.append(np.nan)

    gt_years_list = sorted(gt_data.keys())
    gt_avg_y = []
    for year in gt_years_list:
        transects = gt_data[year]['transects']
        if transects:
            avg_y = np.mean([t[1] for t in transects])
            gt_avg_y.append(avg_y)
        else:
            gt_avg_y.append(np.nan)

    pred_avg_y = []
    for offset in range(1, len(pred_years) + 1):
        if offset in predictions:
            transects = predictions[offset]
            if transects:
                avg_y = np.mean([t[1] for t in transects])
                pred_avg_y.append(avg_y)
            else:
                pred_avg_y.append(np.nan)
        else:
            pred_avg_y.append(np.nan)

    plt.figure(figsize=(14, 7))

    plt.plot(train_years, train_avg_y, 'bo-', linewidth=2.5, markersize=12,
             label='Training Data (2015-2020)')
    plt.plot(gt_years_list, gt_avg_y, 'gs-', linewidth=2.5, markersize=12,
             label='Ground Truth (2021-2025)')
    plt.plot(pred_years, pred_avg_y, 'r^--', linewidth=2.5, markersize=12,
             label='Model Prediction')

    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Coastline Average Y (pixels)', fontsize=14)
    plt.title('Coastline Position Prediction: Model vs Ground Truth', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)

    all_years = train_years + pred_years
    plt.xticks(range(min(all_years), max(all_years)+1), fontsize=12)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'prediction_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_gt_vs_prediction_with_overlap(gt_data, predictions, pred_years, output_dir,
                                            overlap_tolerance=10):
    """
    【核心新功能】GT与预测海岸线对比，包含重合区域显示

    颜色方案：
        - 青色 (Cyan): GT独有区域
        - 洋红色 (Magenta): 预测独有区域
        - 黄色 (Yellow): GT与预测重合区域

    参数:
        overlap_tolerance: 判断重合的容差（像素），在此范围内认为是重合
    """
    years = sorted(gt_data.keys())

    for i, year in enumerate(years):
        data = gt_data[year]
        offset = i + 1
        pred_transects = predictions.get(offset, [])
        gt_transects = data['transects']

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # ========== 图1: 仅GT (青色/Cyan) ==========
        gt_img = data['image'].copy()
        for x, y in gt_transects:
            if 0 <= y < gt_img.shape[0] and 0 <= x < gt_img.shape[1]:
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gt_img.shape[0] and 0 <= nx < gt_img.shape[1]:
                            gt_img[ny, nx] = [0, 255, 255]  # 青色
        axes[0].imshow(gt_img)
        axes[0].set_title(f'GT Coastline (Cyan)\n{len(gt_transects)} points', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # ========== 图2: 仅预测 (洋红色/Magenta) ==========
        pred_img = data['image'].copy()
        for x, y in pred_transects:
            if 0 <= y < pred_img.shape[0] and 0 <= x < pred_img.shape[1]:
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < pred_img.shape[0] and 0 <= nx < pred_img.shape[1]:
                            pred_img[ny, nx] = [255, 0, 255]  # 洋红色
        axes[1].imshow(pred_img)
        axes[1].set_title(f'Predicted Coastline (Magenta)\n{len(pred_transects)} points', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # ========== 图3: GT + 预测简单叠加 ==========
        simple_compare_img = data['image'].copy()
        for x, y in gt_transects:
            if 0 <= y < simple_compare_img.shape[0] and 0 <= x < simple_compare_img.shape[1]:
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < simple_compare_img.shape[0] and 0 <= nx < simple_compare_img.shape[1]:
                            simple_compare_img[ny, nx] = [0, 255, 255]  # 青色
        for x, y in pred_transects:
            if 0 <= y < simple_compare_img.shape[0] and 0 <= x < simple_compare_img.shape[1]:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < simple_compare_img.shape[0] and 0 <= nx < simple_compare_img.shape[1]:
                            simple_compare_img[ny, nx] = [255, 0, 255]  # 洋红色

        axes[2].imshow(simple_compare_img)
        if gt_transects and pred_transects:
            gt_avg_y = np.mean([t[1] for t in gt_transects])
            pred_avg_y = np.mean([t[1] for t in pred_transects])
            error = abs(pred_avg_y - gt_avg_y)
            axes[2].set_title(f'Simple Overlay\nError: {error:.1f} px', fontsize=14, fontweight='bold')
        else:
            axes[2].set_title('Simple Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # ========== 图4: 【重点】带重合区域的对比图 ==========
        overlap_img = data['image'].copy()
        height, width = overlap_img.shape[:2]

        # 创建标记掩膜
        gt_mask = np.zeros((height, width), dtype=bool)
        pred_mask = np.zeros((height, width), dtype=bool)

        # 标记GT区域（稍大一些以便检测重合）
        for x, y in gt_transects:
            for dy in range(-overlap_tolerance, overlap_tolerance + 1):
                for dx in range(-overlap_tolerance, overlap_tolerance + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        gt_mask[ny, nx] = True

        # 标记预测区域
        for x, y in pred_transects:
            for dy in range(-overlap_tolerance, overlap_tolerance + 1):
                for dx in range(-overlap_tolerance, overlap_tolerance + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        pred_mask[ny, nx] = True

        # 计算三种区域
        overlap_mask = gt_mask & pred_mask        # 重合区域
        gt_only_mask = gt_mask & ~pred_mask       # GT独有
        pred_only_mask = pred_mask & ~gt_mask     # 预测独有

        # 着色：先画独有区域，再画重合区域（重合会覆盖）
        overlap_img[gt_only_mask] = [0, 255, 255]       # 青色 - GT独有
        overlap_img[pred_only_mask] = [255, 0, 255]     # 洋红色 - 预测独有
        overlap_img[overlap_mask] = [255, 255, 0]       # 黄色 - 重合区域

        axes[3].imshow(overlap_img)

        # 计算重合率
        total_gt = np.sum(gt_mask)
        total_pred = np.sum(pred_mask)
        total_overlap = np.sum(overlap_mask)
        if total_gt > 0:
            overlap_rate_gt = total_overlap / total_gt * 100
        else:
            overlap_rate_gt = 0
        if total_pred > 0:
            overlap_rate_pred = total_overlap / total_pred * 100
        else:
            overlap_rate_pred = 0

        axes[3].set_title(f'Overlap Analysis (tol={overlap_tolerance}px)\n'
                         f'Cyan=GT only | Magenta=Pred only | Yellow=Overlap\n'
                         f'Overlap: {overlap_rate_gt:.1f}% of GT, {overlap_rate_pred:.1f}% of Pred',
                         fontsize=12, fontweight='bold')
        axes[3].axis('off')

        # 添加统一图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='cyan', label='GT Only'),
            Patch(facecolor='magenta', label='Prediction Only'),
            Patch(facecolor='yellow', label='Overlap'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
                   bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(f'{year} - Coastline Prediction vs Ground Truth', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(output_dir, f'comparison_with_overlap_{year}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def visualize_final_combined_comparison(gt_data, predictions, pred_years, output_dir,
                                        overlap_tolerance=10):
    """
    【最终汇总图】所有预测年份的GT与预测对比，带重合区域
    在一张大图中展示所有年份的对比结果
    """
    years = sorted(gt_data.keys())
    n_years = len(years)

    fig, axes = plt.subplots(n_years, 2, figsize=(16, 5*n_years))
    if n_years == 1:
        axes = axes.reshape(1, -1)

    for i, year in enumerate(years):
        data = gt_data[year]
        offset = i + 1
        pred_transects = predictions.get(offset, [])
        gt_transects = data['transects']

        height, width = data['image'].shape[:2]

        # ========== 左图: 原图 + GT海岸线 ==========
        left_img = data['image'].copy()
        for x, y in gt_transects:
            if 0 <= y < left_img.shape[0] and 0 <= x < left_img.shape[1]:
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            left_img[ny, nx] = [0, 255, 255]  # 青色
        axes[i, 0].imshow(left_img)
        axes[i, 0].set_title(f'{year} - Ground Truth (Cyan)\n{len(gt_transects)} transect points',
                            fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # ========== 右图: 带重合区域的对比图 ==========
        right_img = data['image'].copy()

        # 创建掩膜
        gt_mask = np.zeros((height, width), dtype=bool)
        pred_mask = np.zeros((height, width), dtype=bool)

        for x, y in gt_transects:
            for dy in range(-overlap_tolerance, overlap_tolerance + 1):
                for dx in range(-overlap_tolerance, overlap_tolerance + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        gt_mask[ny, nx] = True

        for x, y in pred_transects:
            for dy in range(-overlap_tolerance, overlap_tolerance + 1):
                for dx in range(-overlap_tolerance, overlap_tolerance + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        pred_mask[ny, nx] = True

        # 计算三种区域
        overlap_mask = gt_mask & pred_mask
        gt_only_mask = gt_mask & ~pred_mask
        pred_only_mask = pred_mask & ~gt_mask

        # 着色
        right_img[gt_only_mask] = [0, 255, 255]       # 青色
        right_img[pred_only_mask] = [255, 0, 255]     # 洋红色
        right_img[overlap_mask] = [255, 255, 0]       # 黄色

        axes[i, 1].imshow(right_img)

        # 计算统计
        if gt_transects and pred_transects:
            gt_avg_y = np.mean([t[1] for t in gt_transects])
            pred_avg_y = np.mean([t[1] for t in pred_transects])
            error = abs(pred_avg_y - gt_avg_y)

            total_overlap = np.sum(overlap_mask)
            total_gt = np.sum(gt_mask)
            overlap_rate = total_overlap / total_gt * 100 if total_gt > 0 else 0

            axes[i, 1].set_title(f'{year} - GT(Cyan) vs Pred(Magenta) | Overlap(Yellow)\n'
                                f'Error: {error:.1f}px | Overlap: {overlap_rate:.1f}%',
                                fontsize=12, fontweight='bold')
        else:
            axes[i, 1].set_title(f'{year} - GT vs Prediction', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='cyan', label='GT Only (Ground Truth)'),
        Patch(facecolor='magenta', label='Prediction Only'),
        Patch(facecolor='yellow', label='Overlap Region'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('Coastline Prediction Summary: GT vs Prediction with Overlap Analysis\n'
                 'Training: 2015-2020 → Prediction: 2021-2025',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    save_path = os.path.join(output_dir, 'final_combined_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_figure(train_data, gt_data, predictions, pred_years, output_dir):
    """创建综合汇总图"""
    fig = plt.figure(figsize=(16, 12))

    train_years = sorted(train_data.keys())
    gt_years_list = sorted(gt_data.keys())

    train_avg_y = [np.mean([t[1] for t in train_data[y]['transects']]) if train_data[y]['transects'] else np.nan for y in train_years]
    gt_avg_y = [np.mean([t[1] for t in gt_data[y]['transects']]) if gt_data[y]['transects'] else np.nan for y in gt_years_list]
    pred_avg_y = [np.mean([t[1] for t in predictions[i+1]]) if predictions.get(i+1) else np.nan for i in range(len(pred_years))]

    # 子图1: 时间序列对比
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(train_years, train_avg_y, 'bo-', linewidth=2, markersize=10, label='Training')
    ax1.plot(gt_years_list, gt_avg_y, 'gs-', linewidth=2, markersize=10, label='Ground Truth')
    ax1.plot(pred_years, pred_avg_y, 'r^--', linewidth=2, markersize=10, label='Prediction')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Coastline Avg Y (pixels)', fontsize=12)
    ax1.set_title('Coastline Position Time Series', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # 子图2: 预测误差
    ax2 = fig.add_subplot(2, 2, 2)
    errors = [abs(pred_avg_y[i] - gt_avg_y[i]) if not (np.isnan(pred_avg_y[i]) or np.isnan(gt_avg_y[i])) else 0
              for i in range(len(pred_years))]

    colors = ['#ff6b6b' if e > 20 else '#4ecdc4' for e in errors]
    bars = ax2.bar(pred_years, errors, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Prediction Error (pixels)', fontsize=12)
    ax2.set_title('Prediction Error by Year', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, err in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{err:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 子图3-4: 示例图像
    ax3 = fig.add_subplot(2, 2, 3)
    first_year = pred_years[0]
    if first_year in gt_data:
        ax3.imshow(gt_data[first_year]['coastline_img'])
        ax3.set_title(f'{first_year} - GT Coastline', fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 2, 4)
    last_year = pred_years[-1]
    if last_year in gt_data:
        ax4.imshow(gt_data[last_year]['coastline_img'])
        ax4.set_title(f'{last_year} - GT Coastline', fontsize=12, fontweight='bold')
    ax4.axis('off')

    plt.suptitle('Coastline Prediction Summary Report\nTraining: 2015-2020 → Prediction: 2021-2025',
                fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(output_dir, 'prediction_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==================== 主处理函数 ====================

def process_time_series(input_dir, output_dir, train_years=(2015, 2020), predict_years=(2021, 2025),
                        num_transects=50, seq_length=3, overlap_tolerance=10):
    """
    主处理函数

    核心流程：
        1. 从PDF/图像中提取白色海岸线作为GT
        2. 将海岸线离散化为transect点
        3. 用训练年份的transect点训练预测模型
        4. 预测后续年份的海岸线位置
        5. 生成各种可视化对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp', '.pdf']

    train_data = {}
    gt_data = {}
    yearly_transects = {}

    print("="*60)
    print("Coastline Prediction System (Transects Method)")
    print("="*60)
    print(f"Input path: {input_dir}")
    print(f"Output path: {output_dir}")
    print(f"Training years: {train_years[0]}-{train_years[1]}")
    print(f"Prediction years: {predict_years[0]}-{predict_years[1]}")
    print(f"Number of transects: {num_transects}")
    print(f"Overlap tolerance: {overlap_tolerance} pixels")
    if PDF_SUPPORT:
        print("PDF support: Enabled")
    else:
        print("PDF support: Disabled (pip install PyMuPDF)")
    print("="*60)

    print("\n[Core Algorithm]")
    print("  1. Extract white lines from PDF/images as GT coastline")
    print("  2. Discretize coastline into transect points (x, y)")
    print("  3. Train time-series model using training years' transects")
    print("  4. Predict future coastline positions")
    print("  5. Compare GT vs Prediction with overlap analysis")
    print("="*60)

    if not os.path.exists(input_dir):
        print(f"\nError: Input directory does not exist - {input_dir}")
        return None, None

    print("\n[Step 1] Detecting coastlines and extracting transects...")
    files_found = sorted(os.listdir(input_dir))

    for filename in files_found:
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            try:
                year = int(''.join(filter(str.isdigit, filename.split('.')[0])))
            except:
                continue

            if year < 2000 or year > 2030:
                continue

            image_path = os.path.join(input_dir, filename)
            print(f"  Processing: {filename} (Year {year})")

            try:
                img, coords, transects, coastline_img, mask = detect_white_coastline(
                    image_path, num_transects=num_transects
                )

                result_data = {
                    'image': img,
                    'mask': mask,
                    'coastline_img': coastline_img,
                    'coords': coords,
                    'transects': transects
                }

                if train_years[0] <= year <= train_years[1]:
                    train_data[year] = result_data
                    yearly_transects[year] = transects
                    print(f"    → Training data: {len(transects)} transects")
                elif predict_years[0] <= year <= predict_years[1]:
                    gt_data[year] = result_data
                    print(f"    → Ground Truth: {len(transects)} transects")

            except Exception as e:
                print(f"    → Processing failed: {e}")
                import traceback
                traceback.print_exc()

    if len(train_data) < seq_length + 1:
        print(f"\nError: Insufficient training data")
        return None, None

    print(f"\n[Step 2] Visualizing training data...")
    visualize_segmentation_results(train_data, output_dir, "Training")

    if gt_data:
        visualize_segmentation_results(gt_data, output_dir, "GT")

    print(f"\n[Step 3] Training transect prediction model...")
    predictor = TransectPredictor(seq_length=seq_length)
    predictor.fit(yearly_transects)
    print(f"  Training complete: {len(predictor.models)} transect models")

    print(f"\n[Step 4] Predicting coastlines for {predict_years[0]}-{predict_years[1]}...")
    num_pred_years = predict_years[1] - predict_years[0] + 1
    predictions = predictor.predict(num_years=num_pred_years)

    pred_years_list = list(range(predict_years[0], predict_years[1] + 1))

    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)

    for i, year in enumerate(pred_years_list):
        offset = i + 1
        pred_transects = predictions.get(offset, [])
        gt_transects = gt_data.get(year, {}).get('transects', [])

        if pred_transects:
            pred_avg_y = np.mean([t[1] for t in pred_transects])
        else:
            pred_avg_y = np.nan

        if gt_transects:
            gt_avg_y = np.mean([t[1] for t in gt_transects])
            error = abs(pred_avg_y - gt_avg_y) if not np.isnan(pred_avg_y) else np.nan
            print(f"{year}: Pred_Y={pred_avg_y:.1f}, GT_Y={gt_avg_y:.1f}, Error={error:.1f}px")
        else:
            print(f"{year}: Pred_Y={pred_avg_y:.1f}, GT=N/A")

    print("="*60)

    print(f"\n[Step 5] Generating visualizations...")

    if gt_data:
        visualize_transects_comparison(train_data, gt_data, predictions, pred_years_list, output_dir)

        # 核心新功能：带重合区域的对比图
        visualize_gt_vs_prediction_with_overlap(gt_data, predictions, pred_years_list, output_dir,
                                                overlap_tolerance=overlap_tolerance)

        # 最终汇总对比图
        visualize_final_combined_comparison(gt_data, predictions, pred_years_list, output_dir,
                                           overlap_tolerance=overlap_tolerance)

        create_summary_figure(train_data, gt_data, predictions, pred_years_list, output_dir)

    # 保存结果
    results_path = os.path.join(output_dir, 'prediction_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Coastline Prediction Results (Transects Method)\n")
        f.write("="*50 + "\n\n")
        f.write("Core Algorithm:\n")
        f.write("  1. Extract white lines from PDF/images as GT coastline\n")
        f.write("  2. Discretize coastline into transect points\n")
        f.write("  3. Train time-series model using training years\n")
        f.write("  4. Predict future coastline positions\n\n")
        f.write(f"Training years: {train_years[0]}-{train_years[1]}\n")
        f.write(f"Prediction years: {predict_years[0]}-{predict_years[1]}\n")
        f.write(f"Number of transects: {num_transects}\n")
        f.write(f"Overlap tolerance: {overlap_tolerance} pixels\n\n")

        f.write("Training data:\n")
        for year in sorted(train_data.keys()):
            transects = train_data[year]['transects']
            avg_y = np.mean([t[1] for t in transects]) if transects else np.nan
            f.write(f"  {year}: {len(transects)} transects, Avg Y={avg_y:.1f}\n")

        f.write("\nPrediction vs Ground Truth:\n")
        for i, year in enumerate(pred_years_list):
            offset = i + 1
            pred_transects = predictions.get(offset, [])
            gt_transects = gt_data.get(year, {}).get('transects', [])

            pred_avg_y = np.mean([t[1] for t in pred_transects]) if pred_transects else np.nan
            gt_avg_y = np.mean([t[1] for t in gt_transects]) if gt_transects else np.nan

            if not np.isnan(gt_avg_y):
                error = abs(pred_avg_y - gt_avg_y)
                f.write(f"  {year}: Pred={pred_avg_y:.1f}, GT={gt_avg_y:.1f}, Error={error:.1f}px\n")
            else:
                f.write(f"  {year}: Pred={pred_avg_y:.1f}, GT=N/A\n")

    print(f"\nResults saved to: {results_path}")
    print(f"All output files saved in: {output_dir}")
    print("\nProcessing complete!")

    return predictions, gt_data


def parse_args():
    parser = argparse.ArgumentParser(description='Coastline Prediction Tool (Transects Method)')
    parser.add_argument("--input_path", type=str, default=r"E:\ground")
    parser.add_argument("--output_path", type=str, default=r"E:\prediction_results")
    parser.add_argument("--num_transects", type=int, default=50, help="Number of transects")
    parser.add_argument("--seq_length", type=int, default=3, help="Sequence length for prediction")
    parser.add_argument("--overlap_tolerance", type=int, default=10,
                        help="Tolerance in pixels for overlap detection")
    return parser.parse_args()


def main(opt):
    process_time_series(
        opt.input_path,
        opt.output_path,
        train_years=(2015, 2020),
        predict_years=(2021, 2025),
        num_transects=opt.num_transects,
        seq_length=opt.seq_length,
        overlap_tolerance=opt.overlap_tolerance
    )


if __name__ == "__main__":
    opt = parse_args()
    main(opt)