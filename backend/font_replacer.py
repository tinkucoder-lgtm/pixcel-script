from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

BOLD_FONTS = {"bebas-neue", "abril-fatface", "oswald", "lobster", "righteous"}

def get_font_path(font_name):
    path = os.path.join(FONTS_DIR, f"{font_name}.ttf")
    if not os.path.exists(path):
        raise ValueError(f"Font not found: {font_name}")
    return path

def get_text_color(image, bbox):
    img_w, img_h = image.size
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    samples = []
    strip = 6
    for sx in range(max(0,x), min(img_w,x+w), max(1,w//8)):
        if y - strip >= 0:
            samples.append(image.getpixel((sx, y-strip))[:3])
        if y + h + strip < img_h:
            samples.append(image.getpixel((sx, y+h+strip))[:3])
    if not samples:
        return (0, 0, 0)
    r = sum(p[0] for p in samples) // len(samples)
    g = sum(p[1] for p in samples) // len(samples)
    b = sum(p[2] for p in samples) // len(samples)
    brightness = (r + g + b) / 3
    if brightness > 200:
        return (0, 0, 0)
    elif brightness > 140:
        return (30, 30, 30)
    else:
        return (255, 255, 255)

def find_font_size(draw, text, font_path, bbox, is_bold=False):
    w, h = bbox["width"], bbox["height"]
    n = len(text)
    if n <= 2:
        base = int(h * 0.65)
    elif n <= 6:
        base = int(h * 0.85)
    elif n <= 20:
        base = int(h * 1.25)
    else:
        base = int(h * 1.20)
    if is_bold:
        base = int(base * 0.75)
    width_limit = 0.80 if is_bold else (0.95 if n <= 6 else 0.92 if n <= 20 else 0.88)
    for size in range(base, 6, -1):
        try:
            font = ImageFont.truetype(font_path, size)
            bb = draw.textbbox((0,0), text, font=font)
            tw = bb[2] - bb[0]
            if tw <= w * width_limit:
                if tw > w:
                    size = int(size * 0.85)
                    font = ImageFont.truetype(font_path, max(6, size))
                return font
        except:
            continue
    return ImageFont.truetype(font_path, max(6, base))

def replace_fonts(image_path, regions, font_name, output_path):
    font_path = get_font_path(font_name)
    is_bold = font_name in BOLD_FONTS
    original_pil = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    img_h, img_w = img_cv.shape[:2]

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for region in regions:
        bbox = region["bounding_box"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        x1, y1 = max(0, x-2), max(0, y-2)
        x2, y2 = min(img_w, x+w+2), min(img_h, y+h+2)
        mask[y1:y2, x1:x2] = 255

    inpainted = cv2.inpaint(img_cv, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    image = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(image)
    for region in regions:
        text = region["text"]
        bbox = region["bounding_box"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        color = get_text_color(original_pil, bbox)
        font = find_font_size(draw, text, font_path, bbox, is_bold)
        bb = draw.textbbox((0,0), text, font=font)
        text_h = bb[3] - bb[1]
        y_offset = max(0, (h - text_h) // 2)
        draw.text((x, y + y_offset), text, font=font, fill=color)

    image.save(output_path, quality=95)
    return output_path
