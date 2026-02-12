print("🔧 改进的海岸线检测系统 - 修复字符和提升检测效果")
print("=" * 70)

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import fitz
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib
from collections import deque
import random

# 修复中文字体问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedCoastalSystem:
    """改进的海岸线检测系统"""

    def __init__(self):
        # 简单的网络（避免过度复杂）
        self.param_ranges = {
            'white_threshold_low': (0.6, 0.85),  # 白色检测下限
            'white_threshold_high': (0.85, 0.98),  # 白色检测上限
            'tolerance': (0.02, 0.15),  # 容忍度
            'morphology_size': (1, 4),  # 形态学大小
            'connectivity_min': (5, 30),  # 最小连通区域
            'edge_enhance': (0.5, 2.5),  # 边缘增强
            'blur_factor': (0.0, 1.5)  # 模糊因子
        }

        print("✅ 改进系统初始化完成")

    def enhanced_white_detection(self, image, params=None):
        """改进的白色检测算法"""

        if params is None:
            params = {
                'white_threshold_low': 0.75,
                'white_threshold_high': 0.92,
                'tolerance': 0.08,
                'morphology_size': 2,
                'connectivity_min': 10,
                'edge_enhance': 1.2,
                'blur_factor': 0.3
            }

        print(f"🔧 使用检测参数:")
        for key, value in params.items():
            print(f"   {key}: {value:.3f}")

        # 确保图像是RGB格式
        if len(image.shape) == 3:
            rgb_image = image.copy()
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            gray = image.copy()
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            rgb_image = np.stack([gray, gray, gray], axis=2)

        r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

        # 多策略白色检测
        strategies = {}

        # 策略1: 严格白色检测
        white_high = int(params['white_threshold_high'] * 255)
        white_low = int(params['white_threshold_low'] * 255)

        strategies['strict'] = (r >= white_high) & (g >= white_high) & (b >= white_high)

        # 策略2: 范围白色检测
        tolerance = int(params['tolerance'] * 255)
        strategies['range'] = (
                (r >= white_low) & (g >= white_low) & (b >= white_low) &
                (np.abs(r.astype(int) - g.astype(int)) <= tolerance) &
                (np.abs(g.astype(int) - b.astype(int)) <= tolerance) &
                (np.abs(r.astype(int) - b.astype(int)) <= tolerance)
        )

        # 策略3: 亮度检测
        brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
        strategies['brightness'] = brightness >= (white_low + white_high) / 2

        # 策略4: 相对亮度检测（比周围亮）
        brightness_blur = ndimage.gaussian_filter(brightness, sigma=3)
        strategies['relative'] = (brightness - brightness_blur) > 20

        # 组合所有策略
        white_mask = np.zeros_like(r, dtype=bool)
        for name, mask in strategies.items():
            white_mask |= mask
            pixels = np.sum(mask)
            print(f"   策略 {name}: 检测到 {pixels} 个白色像素")

        total_white = np.sum(white_mask)
        print(f"🎯 组合检测: {total_white} 个白色像素 ({total_white / white_mask.size * 100:.2f}%)")

        # 形态学处理
        morph_size = max(1, int(params['morphology_size']))
        if morph_size > 1:
            from scipy.ndimage import binary_closing, binary_opening, binary_dilation

            # 结构元素
            kernel = np.ones((morph_size, morph_size), bool)

            # 先膨胀连接断裂，再腐蚀去噪声
            white_mask = binary_dilation(white_mask, structure=kernel, iterations=1)
            white_mask = binary_closing(white_mask, structure=kernel, iterations=1)
            white_mask = binary_opening(white_mask, structure=kernel, iterations=1)

        # 连通区域过滤
        min_size = int(params['connectivity_min'])
        if min_size > 0:
            from scipy.ndimage import label
            labeled, num_features = label(white_mask)

            filtered_mask = np.zeros_like(white_mask)
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) >= min_size:
                    filtered_mask |= component

            removed = total_white - np.sum(filtered_mask)
            white_mask = filtered_mask
            print(f"🔍 连通性过滤: 移除 {removed} 个噪声像素")

        # 边缘增强
        edge_factor = params.get('edge_enhance', 1.0)
        if edge_factor != 1.0:
            # 计算边缘
            grad_x = np.abs(ndimage.sobel(white_mask.astype(float), axis=1))
            grad_y = np.abs(ndimage.sobel(white_mask.astype(float), axis=0))
            edges = grad_x + grad_y

            # 增强边缘区域
            enhanced = white_mask.astype(float) + (edge_factor - 1.0) * edges
            white_mask = enhanced > 0.5

        # 轻微模糊平滑
        blur_sigma = params.get('blur_factor', 0)
        if blur_sigma > 0:
            white_mask_float = ndimage.gaussian_filter(white_mask.astype(float), sigma=blur_sigma)
            white_mask = white_mask_float > 0.3  # 降低阈值保留更多细节

        final_pixels = np.sum(white_mask)
        print(f"✅ 最终检测: {final_pixels} 个海岸线像素")

        return white_mask.astype(float), strategies, {
            'rgb_image': rgb_image,
            'total_white_pixels': final_pixels,
            'detection_ratio': final_pixels / white_mask.size
        }

    def smart_parameter_adjustment(self, image, initial_params, max_iterations=5):
        """智能参数调整"""

        best_params = initial_params.copy()
        best_score = 0
        best_result = None

        print(f"🎯 开始智能参数调整 (最多{max_iterations}次迭代)...")

        for iteration in range(max_iterations):
            print(f"\n  迭代 {iteration + 1}:")

            # 测试当前参数
            result, strategies, info = self.enhanced_white_detection(image, best_params)

            # 计算得分（基于检测到的像素数量和分布）
            pixel_count = info['total_white_pixels']
            ratio = info['detection_ratio']

            # 理想的海岸线像素比例应该在0.5%-5%之间
            if 0.005 <= ratio <= 0.05:
                ratio_score = 1.0
            elif 0.001 <= ratio <= 0.1:
                ratio_score = 0.5
            else:
                ratio_score = 0.1

            # 连通性得分（更少但更大的连通区域更好）
            from scipy.ndimage import label
            labeled, num_components = label(result > 0.5)
            if num_components > 0:
                avg_component_size = pixel_count / num_components
                connectivity_score = min(1.0, avg_component_size / 50.0)
            else:
                connectivity_score = 0.0

            # 综合得分
            current_score = ratio_score * 0.6 + connectivity_score * 0.4

            print(f"    像素数: {pixel_count}, 比例: {ratio:.3%}")
            print(f"    连通区域: {num_components}, 平均大小: {avg_component_size:.1f}")
            print(f"    得分: {current_score:.3f}")

            if current_score > best_score:
                best_score = current_score
                best_params = best_params.copy()
                best_result = result
                print(f"    ✅ 更新最佳参数 (得分: {best_score:.3f})")
            else:
                print(f"    ⚠️ 未改进 (最佳得分: {best_score:.3f})")

            # 如果得分已经很好，提前停止
            if best_score > 0.8:
                print(f"    🎉 达到满意效果，提前停止")
                break

            # 调整参数进行下一次尝试
            if iteration < max_iterations - 1:
                # 根据当前结果调整参数
                if pixel_count < 100:  # 检测太少，降低阈值
                    best_params['white_threshold_low'] = max(0.6, best_params['white_threshold_low'] - 0.05)
                    best_params['tolerance'] = min(0.15, best_params['tolerance'] + 0.02)
                elif pixel_count > 5000:  # 检测太多，提高阈值
                    best_params['white_threshold_high'] = min(0.98, best_params['white_threshold_high'] + 0.02)
                    best_params['tolerance'] = max(0.02, best_params['tolerance'] - 0.01)

                if num_components > 20:  # 太多碎片，增加连通性要求
                    best_params['connectivity_min'] = min(30, best_params['connectivity_min'] + 5)
                    best_params['morphology_size'] = min(4, best_params['morphology_size'] + 0.5)

        print(f"\n🏆 参数调整完成，最佳得分: {best_score:.3f}")
        return best_params, best_result, best_score

    def process_image(self, image_path):
        """处理单张图像"""
        print(f"\n🖼️ 处理图像: {os.path.basename(image_path)}")

        try:
            # 加载图像
            doc = fitz.open(image_path)
            page = doc.load_page(0)
            zoom = 300 / 72  # 提高分辨率
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")

            from io import BytesIO
            img = Image.open(BytesIO(img_data))
            original_img = np.array(img)
            doc.close()

            # 预处理
            processed_img = self.preprocess_image(original_img, (512, 512))

            # 初始参数
            initial_params = {
                'white_threshold_low': 0.72,
                'white_threshold_high': 0.90,
                'tolerance': 0.1,
                'morphology_size': 2,
                'connectivity_min': 15,
                'edge_enhance': 1.5,
                'blur_factor': 0.5
            }

            # 智能参数调整
            best_params, best_result, score = self.smart_parameter_adjustment(
                processed_img, initial_params, max_iterations=3
            )

            return {
                'original_image': original_img,
                'processed_image': processed_img,
                'initial_params': initial_params,
                'best_params': best_params,
                'coastline_result': best_result,
                'quality_score': score,
                'success': score > 0.3
            }

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_image(self, image, target_size):
        """图像预处理"""
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = image

        resized = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)


def create_improved_visualization(result, year, save_path):
    """创建改进的可视化（修复中文显示）"""

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # 使用英文标题避免字符问题
    fig.suptitle(f'Improved Coastline Detection - {year}', fontsize=20, fontweight='bold')

    # 第一行：原始数据和结果
    axes[0, 0].imshow(result['original_image'])
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['processed_image'])
    axes[0, 1].set_title('Processed Image', fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['coastline_result'], cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Coastline Detection', fontsize=14, color='red', fontweight='bold')
    axes[0, 2].axis('off')

    # 叠加结果
    overlay = result['processed_image'].copy()
    if len(overlay.shape) == 3:
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], result['coastline_result'] * 255)
    else:
        overlay = np.stack([overlay] * 3, axis=2)
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], result['coastline_result'] * 255)

    axes[0, 3].imshow(overlay)
    axes[0, 3].set_title('Overlay Result', fontsize=14)
    axes[0, 3].axis('off')

    # 第二行：参数对比
    axes[1, 0].axis('off')
    initial_text = "Initial Parameters:\n\n"
    for key, value in result['initial_params'].items():
        initial_text += f"{key}:\n  {value:.3f}\n"
    axes[1, 0].text(0.05, 0.95, initial_text, transform=axes[1, 0].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    axes[1, 0].set_title('Initial Parameters', fontsize=14)

    axes[1, 1].axis('off')
    best_text = "Optimized Parameters:\n\n"
    for key, value in result['best_params'].items():
        best_text += f"{key}:\n  {value:.3f}\n"
    axes[1, 1].text(0.05, 0.95, best_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    axes[1, 1].set_title('Optimized Parameters', fontsize=14)

    # 参数变化对比
    param_names = list(result['initial_params'].keys())
    initial_values = [result['initial_params'][p] for p in param_names]
    best_values = [result['best_params'][p] for p in param_names]

    x_pos = np.arange(len(param_names))
    width = 0.35

    axes[1, 2].bar(x_pos - width / 2, initial_values, width, label='Initial', alpha=0.7, color='blue')
    axes[1, 2].bar(x_pos + width / 2, best_values, width, label='Optimized', alpha=0.7, color='green')
    axes[1, 2].set_xlabel('Parameters')
    axes[1, 2].set_ylabel('Values')
    axes[1, 2].set_title('Parameter Comparison')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([p.replace('_', '\n') for p in param_names], fontsize=8, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # 质量得分
    axes[1, 3].axis('off')
    score_text = f"""Quality Assessment:

Quality Score: {result['quality_score']:.3f}

Detection Status: {"SUCCESS" if result['success'] else "NEEDS IMPROVEMENT"}

Coastline Pixels: {np.sum(result['coastline_result'] > 0.5):,}

Coverage Ratio: {np.mean(result['coastline_result'] > 0.5) * 100:.2f}%

Assessment: """

    if result['quality_score'] > 0.7:
        assessment = "EXCELLENT"
        color = "green"
    elif result['quality_score'] > 0.4:
        assessment = "GOOD"
        color = "orange"
    else:
        assessment = "POOR"
        color = "red"

    score_text += assessment

    axes[1, 3].text(0.1, 0.9, score_text, transform=axes[1, 3].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
    axes[1, 3].set_title('Quality Assessment', fontsize=14)

    # 第三行：统计分析
    # 像素分布
    axes[2, 0].hist(result['coastline_result'].flatten(), bins=50, alpha=0.7, color='red')
    axes[2, 0].set_title('Pixel Distribution')
    axes[2, 0].set_xlabel('Pixel Values')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].grid(True, alpha=0.3)

    # 二值化结果
    binary_result = result['coastline_result'] > 0.5
    coastline_pixels = np.sum(binary_result)
    background_pixels = np.sum(~binary_result)

    axes[2, 1].pie([coastline_pixels, background_pixels],
                   labels=['Coastline', 'Background'],
                   autopct='%1.1f%%',
                   colors=['red', 'lightblue'])
    axes[2, 1].set_title('Pixel Ratio')

    # 连通区域分析
    from scipy.ndimage import label
    labeled, num_components = label(binary_result)

    if num_components > 0:
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled == i)
            component_sizes.append(size)

        axes[2, 2].hist(component_sizes, bins=min(20, num_components), alpha=0.7, color='green')
        axes[2, 2].set_title(f'Component Sizes (n={num_components})')
        axes[2, 2].set_xlabel('Component Size')
        axes[2, 2].set_ylabel('Count')
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'No Components\nDetected', ha='center', va='center')
        axes[2, 2].set_title('Component Analysis')

    # 改进总结
    axes[2, 3].axis('off')
    summary_text = f"""Improvement Summary:

System Features:
• Multi-strategy detection
• Smart parameter tuning  
• Connectivity filtering
• Edge enhancement
• Morphological processing

Detection Results:
• Components: {num_components}
• Total pixels: {coastline_pixels:,}
• Avg component: {coastline_pixels / max(1, num_components):.1f}

Quality Score: {result['quality_score']:.3f}/1.0
Status: {assessment}"""

    axes[2, 3].text(0.05, 0.95, summary_text, transform=axes[2, 3].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender"))
    axes[2, 3].set_title('System Summary', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 改进版可视化已保存: {save_path}")


def main():
    """主函数"""
    print("🚀 启动改进版海岸线检测系统...")

    # 初始化系统
    system = ImprovedCoastalSystem()

    # 检查数据
    initial_dir = "E:/initial"
    initial_files = [f for f in os.listdir(initial_dir) if f.endswith('.pdf')]

    print(f"📁 找到 {len(initial_files)} 个测试文件")

    # 创建输出目录
    output_dir = "./improved_coastline_results"
    os.makedirs(output_dir, exist_ok=True)

    # 处理前几个样本
    for i, pdf_file in enumerate(initial_files[:3]):
        print(f"\n{'=' * 60}")
        print(f"处理样本 {i + 1}/{min(3, len(initial_files))}: {pdf_file}")

        # 提取年份
        import re
        years = re.findall(r'20\d{2}', pdf_file)
        year = years[0] if years else f"sample_{i + 1}"

        # 处理图像
        pdf_path = os.path.join(initial_dir, pdf_file)
        result = system.process_image(pdf_path)

        if result is not None:
            # 创建可视化
            save_path = os.path.join(output_dir, f'improved_coastline_{year}.png')
            create_improved_visualization(result, year, save_path)

            print(f"✅ 样本 {year} 处理完成，质量得分: {result['quality_score']:.3f}")
        else:
            print(f"❌ 样本 {year} 处理失败")

    print(f"\n🎉 改进版检测完成！")
    print(f"📂 结果保存在: {output_dir}")
    print(f"💡 改进版特点:")
    print(f"   ✅ 修复中文字符显示问题")
    print(f"   ✅ 多策略白色检测算法")
    print(f"   ✅ 智能参数自动调整")
    print(f"   ✅ 更完整的海岸线检测")


if __name__ == "__main__":
    main()