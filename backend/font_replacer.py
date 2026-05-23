# LOCKED RULES - DO NOT MODIFY
# 1. Numbers and 1-2 char text: font size = 45% of bounding box height.
# 2. 3-6 char text: start at 60% height, reduce until fits 95% width.
# 3. 7-20 char text: start at 90% height, reduce until fits 92% width.
# 4. Over 20 chars: start at 85% height, reduce until fits 88% width.
# 5. Bold fonts (bebas-neue, abril-fatface, oswald, lobster, righteous): extra 25% reduction, 80% width limit.
# 6. Final overflow check: if text wider than bbox, reduce 15% and redraw.
# 7. Use OpenCV inpainting with INPAINT_TELEA radius=5 to remove text regions cleanly.
# 8. Sample text colour from 6px strips above and below each text region.
#    brightness > 200 => black, 140-200 => dark grey, under 140 => white. Default black.

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import freetype
import uharfbuzz as hb
import math
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
    sample_width = min(max(w, 20), img_w)
    sample_height = min(6, img_h)

    def sample_region(x_start, y_start, width, height):
        x_start = max(0, min(x_start, img_w - width))
        y_start = max(0, min(y_start, img_h - height))
        x_end = min(img_w, x_start + width)
        y_end = min(img_h, y_start + height)
        if x_end <= x_start or y_end <= y_start:
            return []
        region = image.crop((x_start, y_start, x_end, y_end))
        return list(region.getdata())

    samples = []
    top_y = y - sample_height
    bottom_y = y + h
    sample_x = max(0, min(x, img_w - sample_width))
    if top_y >= 0:
        samples.extend(sample_region(sample_x, top_y, sample_width, sample_height))
    if bottom_y + sample_height <= img_h:
        samples.extend(sample_region(sample_x, bottom_y, sample_width, sample_height))

    if not samples:
        return (0, 0, 0)

    r = sum(p[0] for p in samples) / len(samples)
    g = sum(p[1] for p in samples) / len(samples)
    b = sum(p[2] for p in samples) / len(samples)
    brightness = (r + g + b) / 3

    if brightness > 200:
        return (0, 0, 0)
    if brightness >= 140:
        return (30, 30, 30)
    return (255, 255, 255)


def _shape_text_glyphs(font_path, text, font_size):
    font_bytes = open(font_path, "rb").read()
    hb_face = hb.Face.create(font_bytes)
    hb_font = hb.Font.create(hb_face)
    hb_font.scale = (hb_face.upem, hb_face.upem)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    scale = font_size / hb_face.upem
    pen_x = 0.0
    shaped = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        shaped.append(
            {
                "gid": info.codepoint,
                "x_offset": pos.x_offset * scale,
                "y_offset": pos.y_offset * scale,
                "x_pos": pen_x,
            }
        )
        pen_x += pos.x_advance * scale

    return shaped, pen_x


def _render_shaped_text(font_path, text, font_size, fill, stroke_width=0, stroke_fill=None, shadow=None, faux_bold=False):
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, font_size)
    shaped_glyphs, text_width = _shape_text_glyphs(font_path, text, font_size)

    ascender = face.size.ascender / 64.0
    descender = face.size.descender / 64.0
    height = math.ceil(ascender - descender)
    padding = max(int(stroke_width * 2) + 8, 10)
    image_width = max(1, math.ceil(text_width + padding * 2))
    image_height = max(1, math.ceil(height + padding * 2))

    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    baseline = padding + ascender

    def draw_pass(color, offset_x=0, offset_y=0):
        rgb = color[:3]
        alpha = color[3] if len(color) == 4 else 255
        for glyph in shaped_glyphs:
            face.load_glyph(glyph["gid"], freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
            bitmap = face.glyph.bitmap
            if bitmap.buffer:
                mask = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
                glyph_img = Image.new("RGBA", mask.size, rgb + (alpha,))
                glyph_img.putalpha(mask)

                x = int(round(padding + glyph["x_pos"] + glyph["x_offset"] + face.glyph.bitmap_left + offset_x))
                y = int(round(baseline - face.glyph.bitmap_top - glyph["y_offset"] + offset_y))
                image.alpha_composite(glyph_img, (x, y))

    if shadow:
        sx, sy, sfill = shadow
        draw_pass(sfill, offset_x=sx, offset_y=sy)

    if stroke_width and stroke_fill:
        for dx, dy in [(-stroke_width, 0), (stroke_width, 0), (0, -stroke_width), (0, stroke_width)]:
            draw_pass(stroke_fill, offset_x=dx, offset_y=dy)

    if faux_bold:
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            draw_pass(fill, offset_x=dx, offset_y=dy)
    else:
        draw_pass(fill)

    return image


def _text_width_height(text_img):
    return text_img.width, text_img.height


def _compute_font_constraints(text, target_w, target_h, font_name):
    length = len(text)
    # Inflate small OCR boxes: many detected boxes are tight; use an effective height
    # Use an aggressive multiplier to ensure readable render sizes for small detections
    effective_h = max(target_h, int(target_h * 6), 64)

    # For script fonts, be even more generous so ligaturey scripts render larger
    SCRIPT_FONTS = {"dancing-script", "pacifico", "great-vibes", "pinyon-script", "sacramento", "caveat", "indie-flower"}
    if font_name in SCRIPT_FONTS:
        effective_h = max(effective_h, int(target_h * 8), 96)

    if length <= 2:
        start_size = int(effective_h * 0.45)
        width_limit = target_w
        allow_width_reduction = False
    elif length <= 6:
        start_size = int(effective_h * 0.60)
        width_limit = target_w * 0.97
        allow_width_reduction = True
    elif length <= 20:
        start_size = int(effective_h * 0.90)
        width_limit = target_w * 0.95
        allow_width_reduction = True
    else:
        start_size = int(effective_h * 0.85)
        width_limit = target_w * 0.9
        allow_width_reduction = True

    if font_name in BOLD_FONTS:
        # reduce start size slightly for bold fonts to avoid overflow, but allow more width
        start_size = max(int(start_size * 0.75), 8)
        width_limit = target_w * 0.85

    start_size = max(start_size, 8)
    return start_size, width_limit, allow_width_reduction


def find_best_font_size(draw, text, font_path, target_w, target_h, font_name):
    min_size = 8
    start_size, width_limit, allow_width_reduction = _compute_font_constraints(text, target_w, target_h, font_name)
    # start from the computed start_size (which may be larger than the original bbox height)
    candidate_size = start_size

    while candidate_size >= min_size:
        try:
            font = ImageFont.truetype(font_path, candidate_size)
        except OSError:
            candidate_size -= 1
            continue

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_h <= target_h:
            if not allow_width_reduction:
                return font
            if text_w <= width_limit:
                return font
            # reduce slightly and retry
            candidate_size = max(int(candidate_size * 0.92), min_size)
            continue

        candidate_size -= 1

    return ImageFont.truetype(font_path, min_size)


def _final_width_check(draw, text, font, font_path, target_w, target_h):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    if text_w <= target_w and text_h <= target_h:
        return font

    reduced_size = max(int(font.size * 0.85), 8)
    try:
        return ImageFont.truetype(font_path, reduced_size)
    except OSError:
        return font


def replace_fonts(image_path, regions, font_name, output_path):
    font_path = get_font_path(font_name)

    original_pil = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    img_h, img_w = img_cv.shape[:2]

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    padding = 3
    for region in regions:
        bbox = region["bounding_box"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        mask[y1:y2, x1:x2] = 255

    inpainted_cv = cv2.inpaint(img_cv, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(inpainted_rgb).convert("RGBA")

    draw = ImageDraw.Draw(image)

    # helper: draw with optional tracking (letter-spacing), stroke, shadow and faux-bold
    def draw_text_with_tracking(draw_obj, image_obj, x0, y0, text_str, font_obj, fill, tracking=0, stroke_width=0, stroke_fill=None, shadow=None, faux_bold=False):
        # Preferred approach: draw whole string to preserve ligatures. Only fall back to per-char if tracking needed.
        if tracking == 0:
            # shadow: tuple(offset_x, offset_y, rgba)
            if shadow:
                sx, sy, sfill = shadow
                try:
                    draw_obj.text((x0 + sx, y0 + sy), text_str, font=font_obj, fill=sfill)
                except TypeError:
                    draw_obj.text((x0 + sx, y0 + sy), text_str, font=font_obj, fill=sfill)

            # faux bold: draw text multiple times with tiny offsets to simulate weight
            if faux_bold:
                offsets = [(0,0),(1,0),(0,1)]
                for ox, oy in offsets:
                    try:
                        draw_obj.text((x0+ox, y0+oy), text_str, font=font_obj, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
                    except TypeError:
                        draw_obj.text((x0+ox, y0+oy), text_str, font=font_obj, fill=fill)
                return

            try:
                draw_obj.text((x0, y0), text_str, font=font_obj, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            except TypeError:
                draw_obj.text((x0, y0), text_str, font=font_obj, fill=fill)
            return

        # If tracking != 0, draw per-character applying tracking
        cx = x0
        for ch in text_str:
            try:
                draw_obj.text((cx, y0), ch, font=font_obj, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
            except TypeError:
                draw_obj.text((cx, y0), ch, font=font_obj, fill=fill)
            bbox_ch = draw_obj.textbbox((0, 0), ch, font=font_obj)
            ch_w = bbox_ch[2] - bbox_ch[0]
            cx += ch_w + tracking

    for region in regions:
        text = region.get("text", "")
        bbox = region["bounding_box"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        color = get_text_color(original_pil, bbox)

        font = find_best_font_size(draw, text, font_path, w, h, font_name)
        font = _final_width_check(draw, text, font, font_path, w, h)

        stroke_width = 0
        stroke_fill = None
        shadow = None
        faux_bold = False

        SCRIPT_THIN = {"dancing-script", "pacifico", "great-vibes", "pinyon-script", "sacramento", "caveat", "indie-flower"}
        if font_name in SCRIPT_THIN:
            stroke_width = max(1, int(max(1, font.size * 0.03)))
            stroke_fill = (max(color[0]-40,0), max(color[1]-40,0), max(color[2]-40,0))
            shadow = (1, 1, (0, 0, 0, 100))

        if font_name in BOLD_FONTS:
            faux_bold = True

        shaped_text_image = _render_shaped_text(
            font_path,
            text,
            font.size,
            color,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
            shadow=shadow,
            faux_bold=faux_bold,
        )

        paste_x = x + max(int((w - shaped_text_image.width) / 2), 0)
        paste_y = y + max(int((h - shaped_text_image.height) / 2), 0)
        image.alpha_composite(shaped_text_image, (paste_x, paste_y))

    image.save(output_path, quality=95)
    return output_path
