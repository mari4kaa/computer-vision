import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

def image_read(file_name: str):
    image = Image.open(file_name)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    pix = image.load()
    return {"image_file": image, "image_draw": draw, "image_width": width, "image_height": height, "image_pix": pix}

def change_brightness(image_info, factor=100):
    image, draw, width, height, pix = (image_info["image_file"], image_info["image_draw"],
                                       image_info["image_width"], image_info["image_height"], image_info["image_pix"])

    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j][:3]
            r = min(255, max(0, r + factor))
            g = min(255, max(0, g + factor))
            b = min(255, max(0, b + factor))
            draw.point((i, j), (r, g, b))

def apply_sepia(image_info, depth=30):
    image, draw, width, height, pix = (image_info["image_file"], image_info["image_draw"],
                                       image_info["image_width"], image_info["image_height"], image_info["image_pix"])
    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j][:3]
            gray = (r + g + b) // 3
            r_sepia = min(255, gray + depth * 2)
            g_sepia = min(255, gray + depth)
            b_sepia = min(255, gray)
            draw.point((i, j), (r_sepia, g_sepia, b_sepia))

def apply_grayscale(image_info, intensity=1.0):
    image, draw, width, height, pix = (image_info["image_file"], image_info["image_draw"],
                                       image_info["image_width"], image_info["image_height"], image_info["image_pix"])
    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j][:3]
            gray = (r + g + b) // 3
            r = int((1 - intensity) * r + intensity * gray)
            g = int((1 - intensity) * g + intensity * gray)
            b = int((1 - intensity) * b + intensity * gray)
            draw.point((i, j), (r, g, b))

def apply_negative(image_info, intensity=1.0):
    image, draw, width, height, pix = (image_info["image_file"], image_info["image_draw"],
                                       image_info["image_width"], image_info["image_height"], image_info["image_pix"])

    for i in range(width):
        for j in range(height):
            r, g, b = pix[i, j][:3]
            intensity_scaled = intensity ** 2  # Squaring the intensity for non-linear blending
            r_neg = 255 - r
            g_neg = 255 - g
            b_neg = 255 - b
            r = int((1 - intensity_scaled) * r + intensity_scaled * r_neg)
            g = int((1 - intensity_scaled) * g + intensity_scaled * g_neg)
            b = int((1 - intensity_scaled) * b + intensity_scaled * b_neg)
            draw.point((i, j), (r, g, b))

def apply_gradient(image_info, effect_func, direction):
    image = image_info["image_file"]
    width, height = image_info["image_width"], image_info["image_height"]

    if direction == "none":
        effect_func(image_info)
        return

    mask = np.zeros((width, height), dtype=np.float32)

    if direction == "diagonal":
        for i in range(width):
            for j in range(height):
                mask[i, j] = (i + j) / (width + height)
    elif direction == "to_center":
        center_x, center_y = width // 2, height // 2
        for i in range(width):
            for j in range(height):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                mask[i, j] = distance / max_distance
    elif direction == "from_center":
        center_x, center_y = width // 2, height // 2
        for i in range(width):
            for j in range(height):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                mask[i, j] = 1 - (distance / max_distance)

    mask = np.clip(mask, 0, 1)

    effect_image = image.copy()
    effect_info = {"image_file": effect_image, "image_draw": ImageDraw.Draw(effect_image),
                   "image_width": width, "image_height": height, "image_pix": effect_image.load()}

    effect_func(effect_info)

    for i in range(width):
        for j in range(height):
            original_pixel = image_info["image_pix"][i, j][:3]
            effect_pixel = effect_info["image_pix"][i, j][:3]
            blended_pixel = tuple(
                int(mask[i, j] * e + (1 - mask[i, j]) * o) for e, o in zip(effect_pixel, original_pixel))
            image_info["image_draw"].point((i, j), blended_pixel)

if __name__ == "__main__":
    file_name_start = 'input_picture.jpg'
    file_name_stop = 'output_picture.jpg'

    print('Choose an effect:')
    print('0 - grayscale')
    print('1 - sepia')
    print('2 - negative')
    print('3 - change brightness')
    mode = int(input('effect: '))

    print('Choose a gradient direction:')
    print('0 - no gradient')
    print('1 - from center')
    print('2 - to center')
    print('3 - diagonal')
    gradient_direction = int(input('gradient direction: '))

    direction_mapping = {0: "none", 1: "from_center", 2: "to_center", 3: "diagonal"}
    direction = direction_mapping.get(gradient_direction, "diagonal")

    image_info = image_read(file_name_start)

    if mode == 1:  # Sepia
        depth = int(input('Enter sepia depth (recommended range 10-60): '))
        apply_gradient(image_info, lambda img: apply_sepia(img, depth), direction=direction)
    elif mode == 0 or mode == 2: # Grayscale or Negative
        intensity = float(input('Enter effect intensity (0.0 to 1.0): '))
        if mode == 0:
            apply_gradient(image_info, lambda img: apply_grayscale(img, intensity), direction=direction)
        else:
            apply_gradient(image_info, lambda img: apply_negative(img, intensity), direction=direction)
    elif mode == 3: # Brightness
        brightness_factor = int(input('Enter brightness factor (-100 to 100): '))
        apply_gradient(image_info, lambda img: change_brightness(img, brightness_factor), direction=direction)

    plt.imshow(image_info["image_file"])
    plt.show()
    image_info["image_file"].save(file_name_stop)
    print(f'Image saved as {file_name_stop}')
