import os
import numpy as np
from tifffile import imread, imwrite
from skimage.measure import label
from skimage.transform import resize
import colorsys

# === 路径设置 ===
input_folder = '/Users/frank/Desktop/morf3/colorize/raw'
mask_folder = '/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0417_czi/segmentation/seg_good'

output_folder = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0417_czi/good_output/"
combined_output_file = output_folder + 'combined_output.tif'
combined_color_mask_file = output_folder + 'combined_color_mask_raw.tif'
combined_color_noise_file = output_folder + 'combined_color_noise.tif'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(combined_output_file), exist_ok=True)

# === 目标尺寸 ===
target_shape = (51, 512, 512, 3)

# === 全局颜色表 ===
def generate_large_color_map(max_labels=10000, seed=42):
    np.random.seed(seed)
    hues = np.linspace(0, 1, max_labels, endpoint=False)
    np.random.shuffle(hues)
    hsv_colors = [(h, 0.9, 1.0) for h in hues]
    rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
    rgb_colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors]
    return np.array([[0, 0, 0]] + rgb_colors, dtype=np.uint8)

global_color_map = generate_large_color_map(max_labels=10000)

# === 主处理函数 ===
def adjust_shape(data, mask, label_offset=0, min_voxel_size=10):
    if data.ndim == 3:
        data = data[:, np.newaxis, :, :]
    elif data.ndim == 4 and data.shape[1] > 1:
        data = data[:, :1, :, :]
    data = data[:, 0, :, :]  # (Z, Y, X)

    z, z_target = data.shape[0], target_shape[0]
    z_resize = min(z, z_target)
    y_target, x_target = target_shape[1], target_shape[2]

    resized_crop = resize(data[:z_resize], (z_resize, y_target, x_target),
                          anti_aliasing=True, preserve_range=True).astype(np.float32)
    resized_mask_crop = resize(mask[:z_resize], (z_resize, y_target, x_target),
                               anti_aliasing=False, preserve_range=True).astype(np.float32)

    resized = np.zeros((z_target, y_target, x_target), dtype=np.float32)
    resized_mask = np.zeros((z_target, y_target, x_target), dtype=np.float32)
    resized[:z_resize] = resized_crop
    resized_mask[:z_resize] = resized_mask_crop

    # === 连通区域标记 + 体素过滤
    binary = (resized * resized_mask > 10).astype(np.uint8)
    raw_labeled = label(binary, connectivity=2)

    labeled = np.zeros_like(raw_labeled, dtype=np.int32)
    region_sizes = np.bincount(raw_labeled.flatten())
    current_label = label_offset + 1

    for region_id, size in enumerate(region_sizes):
        if region_id == 0:
            continue
        if size >= min_voxel_size:
            labeled[raw_labeled == region_id] = current_label
            current_label += 1

    new_max_label = current_label - 1

    # === 灰度归一化
    brightness_factor = 10.0
    norm_intensity = (resized / np.clip(resized.max(), 1e-5, None)) * brightness_factor
    norm_intensity = np.clip(norm_intensity, 0, 1)[..., np.newaxis]

    # === 上色区域
    if labeled.max() >= len(global_color_map):
        raise ValueError("Too many labels. Increase max_labels in color map.")
    color_mask = global_color_map[labeled]
    colored_region = (color_mask * norm_intensity).astype(np.uint8)

    # === 背景区域使用原图灰度（灰色）
    background_gray = np.clip(resized / np.clip(resized.max(), 1e-5, None), 0, 1) * 800
    background_rgb = np.stack([background_gray] * 3, axis=-1).astype(np.uint8)

    # === 合并前景（彩色）与背景（灰色）
    mask_binary = (resized_mask > 5).astype(np.uint8)[..., np.newaxis]
    colored_volume = colored_region * mask_binary + background_rgb * (1 - mask_binary)

    return colored_volume, labeled, resized, color_mask  # 👈 添加 color_mask 输出

# === 批量处理并合并 ===
combined_label = None
combined_gray = None
combined_color_mask = None  # 👈 新增
global_label_offset = 0
number = 0

for file in sorted(os.listdir(input_folder)):
    if file.endswith('.tif'):
        file_path = os.path.join(input_folder, file)
        mask_path = os.path.join(mask_folder, file)

        if os.path.exists(mask_path):
            print(f"Processing: {file}")
            data = imread(file_path)
            mask = imread(mask_path)

            colored, labeled, gray, color_mask = adjust_shape(
                data, mask, label_offset=global_label_offset, min_voxel_size=10)
            global_label_offset = labeled.max()

            if combined_label is None:
                combined_label = labeled
                combined_gray = gray
                combined_img = colored.astype(np.float32)
                combined_color_mask = color_mask.astype(np.float32)
            else:
                combined_label = np.maximum(combined_label, labeled)
                combined_gray = np.maximum(combined_gray, gray)
                combined_img += colored.astype(np.float32)
                combined_color_mask += color_mask.astype(np.float32)  # 👈 合并 color mask

            number += 1
            output_file = os.path.join(output_folder, file)
            imwrite(output_file, colored, imagej=True)
            print(f"  Saved to: {output_file}")
        else:
            print(f"Mask not found for {file}, skipping.")


def add_color_noise_and_save(image, noise_strength, base_save_path):
    """
    给图像分别添加高斯噪声和均匀噪声，并各自保存成文件。

    参数:
        image (np.ndarray): 原始彩色图像（uint8 或 float32）
        noise_strength (float): 噪声强度（最大像素扰动范围）
        base_save_path (str): 不带扩展名的基础保存路径

    返回:
        (noisy_normal, noisy_uniform): 加噪后的两个图像（均为uint8）
    """
    image = image.astype(np.float32)

    # === 高斯噪声 ===
    noise_normal = np.random.normal(loc=0.0, scale=noise_strength, size=image.shape).astype(np.float32)
    noisy_normal = image + noise_normal
    noisy_normal = np.clip(noisy_normal, 0, 255).astype(np.uint8)
    normal_path = f"{base_save_path}_{int(noise_strength)}_normal.tif"
    imwrite(normal_path, noisy_normal, imagej=True)
    print(f"📸 Saved Gaussian noise image to: {normal_path}")

    # === 均匀噪声 ===
    noise_uniform = np.random.uniform(low=-noise_strength, high=noise_strength, size=image.shape).astype(np.float32)
    noisy_uniform = image + noise_uniform
    noisy_uniform = np.clip(noisy_uniform, 0, 255).astype(np.uint8)
    uniform_path = f"{base_save_path}_{int(noise_strength)}_uniform.tif"
    imwrite(uniform_path, noisy_uniform, imagej=True)
    print(f"📸 Saved Uniform noise image to: {uniform_path}")

    return noisy_normal, noisy_uniform



# === 保存合并图像 ===
if number > 0:
    brightness_factor = 3.0

    # 彩色图像（灰度背景 + 彩色 mask）
    combined_img = combined_img / number
    combined_img = np.clip(combined_img * brightness_factor, 0, 255).astype(np.uint8)
    imwrite(combined_output_file, combined_img, imagej=True)
    print(f"\n✅ Combined file saved to: {combined_output_file}")
    # 添加颜色浮动（每个像素的RGB通道加入小扰动）
    add_color_noise_and_save(combined_img, 10, combined_output_file)
    add_color_noise_and_save(combined_img, 20, combined_output_file)
    add_color_noise_and_save(combined_img, 40, combined_output_file)


    # 合并后的纯 color_mask（无亮度变化、无灰度背景）
    brightness_factor = 2.0
    combined_color_mask = combined_color_mask / number
    combined_color_mask = np.clip(combined_color_mask* brightness_factor, 0, 255).astype(np.uint8)
    imwrite(combined_color_mask_file, combined_color_mask, imagej=True)
    print(f"🎨 Combined color mask saved to: {combined_color_mask_file}")


else:
    print("❌ No images were processed.")
