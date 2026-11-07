import os
import io
import re
import json
import math
import base64
import logging
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter, ImageChops
import colorsys
import requests
from dotenv import load_dotenv
import time


# --- App and Config ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, 'RUNS')
os.makedirs(RUNS_DIR, exist_ok=True)

# Asset directories
CIRCLE_TYPES_DIR = os.path.join(BASE_DIR, 'Circle-Types')
CONNECTOR_TYPES_DIR = os.path.join(BASE_DIR, 'Connector-Types')
os.makedirs(CIRCLE_TYPES_DIR, exist_ok=True)
os.makedirs(CONNECTOR_TYPES_DIR, exist_ok=True)

# --- Scene and Styling Config (edit these) ---
GRID_SIZE = 50  # meters (scene is GRID_SIZE x GRID_SIZE)
IMG_BASE_SIZE = 1200
LINE_COLOR = (220, 220, 220)
AXIS_LABEL_COLOR = (80, 80, 80)
CIRCLE_BORDER_COLOR = (0, 120, 255)
TEXT_COLOR = (20, 20, 20)
BG_COLOR = (255, 255, 255)
DEFAULT_CIRCLE_RADIUS_CELLS = 3.0
DEFAULT_FILL_COLOR = "#ffffff"
DEFAULT_SATURATION = 30.0
DEFAULT_CONNECTOR_LENGTH_M = 2.0  # along tangent
DEFAULT_CONNECTOR_WIDTH_M = 0.5   # perpendicular to tangent

# Variation/autopilot configuration (editable)
# - VARIATIONS_PER_STEP: how many candidate placements per step
# - AUTO_STEPS_DEFAULT: how many steps Auto Run performs by default
VARIATIONS_PER_STEP = int(os.environ.get('VARIATIONS_PER_STEP', '3'))
AUTO_STEPS_DEFAULT = int(os.environ.get('AUTO_STEPS_DEFAULT', '4'))

# First circle automatic placement (editable)
FIRST_CIRCLE_X = float(os.environ.get('FIRST_CIRCLE_X', '25.0'))
FIRST_CIRCLE_Y = float(os.environ.get('FIRST_CIRCLE_Y', '25.0'))
FIRST_CIRCLE_RADIUS = float(os.environ.get('FIRST_CIRCLE_RADIUS', '2.75'))
FIRST_CIRCLE_FILL = os.environ.get('FIRST_CIRCLE_FILL', '#f2f2f2')
FIRST_CIRCLE_TYPE = os.environ.get('FIRST_CIRCLE_TYPE', '')

# OpenRouter
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_SITE_URL = os.getenv('OPENROUTER_SITE_URL', 'http://localhost')
OPENROUTER_SITE_NAME = os.getenv('OPENROUTER_SITE_NAME', 'Circles')
OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Models config
MODELS_FILE_PRIMARY = 'json.models'
MODELS_FILE_FALLBACK = 'models.json'


# --- Utilities ---
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_font(preferred_size: int) -> ImageFont.FreeTypeFont:
    try:
        font_path = (
            'C:/Windows/Fonts/arialbd.ttf' if os.path.exists('C:/Windows/Fonts/arialbd.ttf')
            else 'C:/Windows/Fonts/arial.ttf' if os.path.exists('C:/Windows/Fonts/arial.ttf')
            else None
        )
        if font_path:
            return ImageFont.truetype(font_path, preferred_size)
    except Exception:
        pass
    return ImageFont.load_default()


# --- Asset/type utilities ---
_CIRCLE_TYPE_CACHE: Dict[str, Image.Image] = {}
_CONNECTOR_TYPE_CACHE: Dict[str, Image.Image] = {}


def _list_types_from_dir(folder: str) -> List[str]:
    try:
        return [os.path.splitext(f)[0] for f in os.listdir(folder) if f.lower().endswith('.png')]
    except Exception:
        return []


def _available_circle_types() -> List[str]:
    return _list_types_from_dir(CIRCLE_TYPES_DIR)


def _available_connector_types() -> List[str]:
    return _list_types_from_dir(CONNECTOR_TYPES_DIR)


def _load_type_image(folder: str, name: str) -> Image.Image | None:
    try:
        path = os.path.join(folder, f"{name}.png")
        if not os.path.exists(path):
            return None
        im = Image.open(path).convert('RGBA')
        return im
    except Exception:
        return None


def _get_circle_type_image(name: str) -> Image.Image | None:
    if not name:
        return None
    img = _CIRCLE_TYPE_CACHE.get(name)
    if img is None:
        img = _load_type_image(CIRCLE_TYPES_DIR, name)
        if img is not None:
            _CIRCLE_TYPE_CACHE[name] = img
    return img


def _get_connector_type_image(name: str) -> Image.Image | None:
    if not name:
        return None
    img = _CONNECTOR_TYPE_CACHE.get(name)
    if img is None:
        img = _load_type_image(CONNECTOR_TYPES_DIR, name)
        if img is not None:
            _CONNECTOR_TYPE_CACHE[name] = img
    return img

def _session_dir_for_index(index: int) -> str:
    return os.path.join(RUNS_DIR, f'H{index}')


def _get_next_session_index() -> int:
    existing = [d for d in os.listdir(RUNS_DIR) if d.startswith('H') and os.path.isdir(os.path.join(RUNS_DIR, d))]
    max_i = 0
    for name in existing:
        try:
            i = int(name[1:])
            max_i = max(max_i, i)
        except Exception:
            continue
    return max_i + 1


def _state_path(session_dir: str) -> str:
    return os.path.join(session_dir, 'state.json')


def _history_path(session_dir: str) -> str:
    return os.path.join(session_dir, 'History.txt')


def _read_state(session_dir: str) -> Dict[str, Any]:
    p = _state_path(session_dir)
    if not os.path.exists(p):
        # Initialize with the first circle automatically
        first_type = FIRST_CIRCLE_TYPE or (_available_circle_types()[0] if _available_circle_types() else '')
        return {
            'grid_size': GRID_SIZE,
            'elements': [{
                'type': 'circle',
                'id': 'C1',
                'x': FIRST_CIRCLE_X,
                'y': FIRST_CIRCLE_Y,
                'radius': FIRST_CIRCLE_RADIUS,
                'fill_color': FIRST_CIRCLE_FILL,
                'circle_type': first_type,
                'rotation_deg': 0.0
            }],
            'step_index': 0,
            'counters': {'C': 2, 'T': 1},
            'options': {
                'saturation': DEFAULT_SATURATION,
                'render_mode': 'image',
                'connector': {
                    'enabled': True,
                    'length_m': DEFAULT_CONNECTOR_LENGTH_M,
                    'width_m': DEFAULT_CONNECTOR_WIDTH_M,
                    'fill_color': '#ffffff',
                    'type': '',
                    # back-compat keys
                    'width': DEFAULT_CONNECTOR_LENGTH_M,
                    'height': DEFAULT_CONNECTOR_WIDTH_M,
                }
            }
        }
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Ensure new fields exist for backward compatibility
        if 'options' not in data:
            data['options'] = {
                'saturation': DEFAULT_SATURATION,
                'connector': {
                    'enabled': True,
                    'length_m': DEFAULT_CONNECTOR_LENGTH_M,
                    'width_m': DEFAULT_CONNECTOR_WIDTH_M,
                    'fill_color': '#ffffff',
                    'width': DEFAULT_CONNECTOR_LENGTH_M,
                    'height': DEFAULT_CONNECTOR_WIDTH_M,
                }
            }
        else:
            if 'saturation' not in data['options']:
                data['options']['saturation'] = DEFAULT_SATURATION
            if 'render_mode' not in data['options']:
                data['options']['render_mode'] = 'image'
            if 'connector' not in data['options']:
                data['options']['connector'] = {
                    'enabled': True,
                    'length_m': DEFAULT_CONNECTOR_LENGTH_M,
                    'width_m': DEFAULT_CONNECTOR_WIDTH_M,
                    'fill_color': '#ffffff',
                    'type': '',
                    'width': DEFAULT_CONNECTOR_LENGTH_M,
                    'height': DEFAULT_CONNECTOR_WIDTH_M,
                }
            else:
                data['options']['connector'].setdefault('enabled', True)
                conn = data['options']['connector']
                conn.setdefault('length_m', conn.get('width', DEFAULT_CONNECTOR_LENGTH_M))
                conn.setdefault('width_m', conn.get('height', DEFAULT_CONNECTOR_WIDTH_M))
                conn.setdefault('width', conn.get('length_m', DEFAULT_CONNECTOR_LENGTH_M))
                conn.setdefault('height', conn.get('width_m', DEFAULT_CONNECTOR_WIDTH_M))
                data['options']['connector'].setdefault('fill_color', '#ffffff')
                data['options']['connector'].setdefault('type', '')
        # Auto-initialize first circle if no elements present
        try:
            if not data.get('elements'):
                ctype = FIRST_CIRCLE_TYPE or (_available_circle_types()[0] if _available_circle_types() else '')
                data['elements'] = [{
                    'type': 'circle',
                    'id': 'C1',
                    'x': FIRST_CIRCLE_X,
                    'y': FIRST_CIRCLE_Y,
                    'radius': FIRST_CIRCLE_RADIUS,
                    'fill_color': FIRST_CIRCLE_FILL,
                    'circle_type': ctype,
                    'rotation_deg': 0.0
                }]
                counters = data.get('counters') or {'C': 1, 'T': 1}
                counters['C'] = max(int(counters.get('C', 1)), 2)
                data['counters'] = counters
        except Exception:
            pass
        return data


def _write_state(session_dir: str, state: Dict[str, Any]) -> None:
    with open(_state_path(session_dir), 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def _append_history(session_dir: str, actions: List[Dict[str, Any]], turn_note: str = '') -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"--- {ts} UTC ---\n"
    line += json.dumps({'actions': actions, 'turn_note': turn_note}, ensure_ascii=False)
    line += "\n\n"
    with open(_history_path(session_dir), 'a', encoding='utf-8') as f:
        f.write(line)


def _read_history_text(session_dir: str, max_chars: int = 40000) -> str:
    hp = _history_path(session_dir)
    if not os.path.exists(hp):
        return ''
    with open(hp, 'r', encoding='utf-8') as f:
        data = f.read()
        if len(data) > max_chars:
            return data[-max_chars:]
        return data


def _cell_size(img_size: int) -> float:
    return img_size / GRID_SIZE


def _grid_to_px(x: float, y: float, img_size: int) -> Tuple[int, int]:
    cs = _cell_size(img_size)
    return (int(x * cs), int((GRID_SIZE - y) * cs))


def _draw_grid(image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    cs = _cell_size(w)
    for i in range(GRID_SIZE + 1):
        x = int(i * cs)
        y = int(i * cs)
        draw.line([(x, 0), (x, h)], fill=LINE_COLOR, width=max(1, int(cs / 60)))
        draw.line([(0, y), (w, y)], fill=LINE_COLOR, width=max(1, int(cs / 60)))
    font = _load_font(max(10, int(w / 50)))
    for i in range(GRID_SIZE + 1):
        x = int(i * cs)
        draw.text((x + 2, 2), str(i), fill=AXIS_LABEL_COLOR, font=font)
        y_px = int(h - (i * cs) - 2)
        if y_px >= 0:
            draw.text((2, y_px), str(i), fill=AXIS_LABEL_COLOR, font=font)
    title_font = _load_font(max(12, int(w / 40)))
    draw.text((w // 2, 10), f'{GRID_SIZE}m x {GRID_SIZE}m Grid (centers anchored)', fill=(0, 0, 0), font=title_font, anchor='ma')
    return image


def _parse_color(color_str: str) -> Tuple[int, int, int]:
    try:
        return ImageColor.getrgb(color_str)
    except Exception:
        try:
            return ImageColor.getrgb(DEFAULT_FILL_COLOR)
        except Exception:
            return (255, 255, 255)


def _apply_saturation(rgb: Tuple[int, int, int], saturation_percent: float) -> Tuple[int, int, int]:
    try:
        r, g, b = rgb
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        hf, sf, vf = colorsys.rgb_to_hsv(rf, gf, bf)
        factor = max(0.0, min(1.0, float(saturation_percent) / 100.0))
        s2 = max(0.0, min(1.0, sf * factor))
        r2, g2, b2 = colorsys.hsv_to_rgb(hf, s2, vf)
        return (int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255)))
    except Exception:
        return rgb


# Legacy pattern utilities removed (floral, fractal_chain, square_ring) in favor of unified variation-based placement


def _draw_elements(base: Image.Image, elements: List[Dict[str, Any]], with_grid: bool, options: Dict[str, Any] | None = None) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cs = _cell_size(w)
    if with_grid:
        img = _draw_grid(img)
        draw = ImageDraw.Draw(img)

    options = options or {}
    saturation_percent = float(options.get('saturation', DEFAULT_SATURATION))
    render_mode = str(options.get('render_mode', 'image') or 'image').lower()
    connector_opt = options.get('connector', {}) if isinstance(options.get('connector', {}), dict) else {}

    # Helper to draw shadows
    def _shadow_ellipse(bbox, blur_px=None, offset_px=None, alpha=90):
        try:
            if blur_px is None:
                blur_px = max(3, int(cs * 0.6))
            if offset_px is None:
                offset_px = max(2, int(cs * 0.25))
            shadow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(shadow)
            x0, y0, x1, y1 = bbox
            sdraw.ellipse((x0 + offset_px, y0 + offset_px, x1 + offset_px, y1 + offset_px), fill=(0, 0, 0, alpha))
            blur = shadow.filter(ImageFilter.GaussianBlur(radius=blur_px))
            img.alpha_composite(blur)
        except Exception:
            pass

    def _shadow_polygon(points, blur_px=None, offset_px=None, alpha=90):
        try:
            if blur_px is None:
                blur_px = max(3, int(cs * 0.6))
            if offset_px is None:
                offset_px = max(2, int(cs * 0.25))
            shadow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(shadow)
            pts = [(px + offset_px, py + offset_px) for (px, py) in points]
            sdraw.polygon(pts, fill=(0, 0, 0, alpha))
            blur = shadow.filter(ImageFilter.GaussianBlur(radius=blur_px))
            img.alpha_composite(blur)
        except Exception:
            pass

    circles = [e for e in elements if e.get('type') == 'circle']
    for j, el in enumerate(circles):
        x, y = float(el['x']), float(el['y'])
        cx, cy = _grid_to_px(x, y, w)
        r_cells = float(el.get('radius', DEFAULT_CIRCLE_RADIUS_CELLS))
        r = max(5, int(cs * r_cells))
        base_rgb = _parse_color(str(el.get('fill_color') or DEFAULT_FILL_COLOR))
        fill_rgb = _apply_saturation(base_rgb, saturation_percent)
        _shadow_ellipse((cx - r, cy - r, cx + r, cy + r), blur_px=max(1, int(cs * 0.1)), offset_px=max(1, int(cs * 0.05)))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill_rgb, outline=CIRCLE_BORDER_COLOR, width=max(2, int(cs / 20)))
        cid = str(el.get('id') or '')
        if cid:
            font = _load_font(max(10, int(cs * 0.5)))
            draw.text((cx + 1, cy + 1), cid, fill=(0, 0, 0, 80), font=font, anchor='mm')
            draw.text((cx, cy), cid, fill=(0, 0, 0), font=font, anchor='mm')

        # Overlay circle type image (if provided and in image render mode)
        if render_mode == 'image':
            try:
                ctype = str(el.get('circle_type') or '').strip()
                if ctype:
                    tex = _get_circle_type_image(ctype)
                    if tex is not None:
                        box_size = max(1, 2 * r)
                        tex_resized = tex.resize((box_size, box_size), resample=Image.BICUBIC)
                        rot_cw = float(el.get('rotation_deg', 0.0) or 0.0)
                        rot_ccw = -rot_cw
                        tex_rot = tex_resized.rotate(rot_ccw, resample=Image.BICUBIC, expand=False)
                        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                        # circular mask
                        mask_sq = Image.new('L', (box_size, box_size), 0)
                        mdraw = ImageDraw.Draw(mask_sq)
                        mdraw.ellipse((0, 0, box_size, box_size), fill=255)
                        overlay.paste(tex_rot, (cx - r, cy - r), mask_sq)
                        img.alpha_composite(overlay)
            except Exception:
                pass

        # Draw connectors for tangency pairs (i, j) immediately after circle j
        try:
            if connector_opt.get('enabled', True):
                tol = 0.01
                length = float(connector_opt.get('length_m', connector_opt.get('width', DEFAULT_CONNECTOR_LENGTH_M)))
                width_n = float(connector_opt.get('width_m', connector_opt.get('height', DEFAULT_CONNECTOR_WIDTH_M)))
                cfill = _apply_saturation(_parse_color(str(connector_opt.get('fill_color', '#ffffff'))), saturation_percent)
                stroke_w = max(2, int(cs / 20))
                xj, yj, rj = float(el.get('x')), float(el.get('y')), float(el.get('radius', DEFAULT_CIRCLE_RADIUS_CELLS))
                for i in range(j):
                    other = circles[i]
                    xi, yi, ri = float(other.get('x')), float(other.get('y')), float(other.get('radius', DEFAULT_CIRCLE_RADIUS_CELLS))
                    dx, dy = (xj - xi), (yj - yi)
                    d = math.hypot(dx, dy)
                    if d <= 0 or abs(d - (ri + rj)) > tol:
                        continue
                    nx, ny = dx / d, dy / d
                    cxg, cyg = (xi + nx * ri), (yi + ny * ri)
                    ux, uy = -ny, nx
                    vx, vy = nx, ny
                    hl, hw = length / 2.0, width_n / 2.0
                    p1g = (cxg + ux * hl + vx * hw, cyg + uy * hl + vy * hw)
                    p2g = (cxg + ux * hl - vx * hw, cyg + uy * hl - vy * hw)
                    p3g = (cxg - ux * hl - vx * hw, cyg - uy * hl - vy * hw)
                    p4g = (cxg - ux * hl + vx * hw, cyg - uy * hl + vy * hw)
                    p1 = _grid_to_px(p1g[0], p1g[1], w)
                    p2 = _grid_to_px(p2g[0], p2g[1], w)
                    p3 = _grid_to_px(p3g[0], p3g[1], w)
                    p4 = _grid_to_px(p4g[0], p4g[1], w)
                    _shadow_polygon([p1, p2, p3, p4], blur_px=max(1, int(cs * 0.1)), offset_px=max(1, int(cs * 0.05)))
                    draw.polygon([p1, p2, p3, p4], fill=cfill)
                    draw.line([p1, p2], fill=CIRCLE_BORDER_COLOR, width=stroke_w)
                    draw.line([p3, p4], fill=CIRCLE_BORDER_COLOR, width=stroke_w)

                    # Overlay connector type image strip if configured and image render mode
                    if render_mode == 'image':
                        try:
                            ctype_name = str(connector_opt.get('type') or '').strip()
                            if ctype_name:
                                tex = _get_connector_type_image(ctype_name)
                                if tex is not None:
                                    cx_px = int(round((p1[0] + p2[0] + p3[0] + p4[0]) / 4.0))
                                    cy_px = int(round((p1[1] + p2[1] + p3[1] + p4[1]) / 4.0))
                                    L_px = max(1, int(round(length * cs)))
                                    W_px = max(1, int(round(width_n * cs)))
                                    tex_scaled = tex.resize((L_px, L_px), resample=Image.BICUBIC)
                                    if W_px < L_px:
                                        top = max(0, (L_px - W_px) // 2)
                                        tex_strip = tex_scaled.crop((0, top, L_px, top + W_px))
                                    else:
                                        tex_strip = tex_scaled
                                    angle_ccw = math.degrees(math.atan2(-uy, ux))
                                    tex_rot = tex_strip.rotate(angle_ccw, resample=Image.BICUBIC, expand=True)
                                    tw, th = tex_rot.size
                                    # Build polygon mask
                                    poly_mask = Image.new('L', (w, h), 0)
                                    mdraw2 = ImageDraw.Draw(poly_mask)
                                    mdraw2.polygon([p1, p2, p3, p4], fill=255)
                                    overlay2 = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                                    overlay2.paste(tex_rot, (cx_px - tw // 2, cy_px - th // 2), tex_rot)
                                    # Constrain overlay by polygon mask
                                    alpha = overlay2.split()[3]
                                    combined = ImageChops.multiply(alpha, poly_mask)
                                    overlay2.putalpha(combined)
                                    img.alpha_composite(overlay2)
                        except Exception:
                            pass
        except Exception:
            pass

    for el in elements:
        if el.get('type') != 'text':
            continue
        x, y = el['x'], el['y']
        cx, cy = _grid_to_px(x, y, w)
        font = _load_font(max(12, int(cs * 0.5)))
        draw.text((cx + 1, cy + 1), el.get('text', ''), fill=(0, 0, 0, 80), font=font, anchor='mm')
        draw.text((cx, cy), el.get('text', ''), fill=TEXT_COLOR, font=font, anchor='mm')

    # connectors handled interleaved above per creation order
    return img


def _render_session_images(session_dir: str, state: Dict[str, Any]) -> Tuple[str, str | None]:
    step = state.get('step_index', 0)
    base = Image.new('RGBA', (IMG_BASE_SIZE, IMG_BASE_SIZE), (*BG_COLOR, 255))
    clean = _draw_elements(base, state.get('elements', []), with_grid=False, options=state.get('options') or {})
    clean_path = os.path.join(session_dir, f"{step}XG.png")
    clean.convert('RGB').save(clean_path)
    return clean_path, None


def _load_models_file() -> Dict[str, Any]:
    paths = [MODELS_FILE_PRIMARY, MODELS_FILE_FALLBACK]
    for p in paths:
        try:
            full = os.path.join(BASE_DIR, p)
            if os.path.exists(full):
                with open(full, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {
                            'models': data.get('models', []),
                            'default': data.get('default', (data.get('models', [""])[0] if data.get('models') else ""))
                        }
        except Exception as e:
            logging.warning(f"Models file '{p}' failed to load: {e}")
    return {'models': [], 'default': ''}


def _get_selected_model(preferred: str | None) -> str:
    cfg = _load_models_file()
    if preferred and isinstance(preferred, str) and preferred.strip():
        return preferred.strip()
    return cfg.get('default') or (cfg.get('models') or [''])[0]


def _or_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError('OPENROUTER_API_KEY not configured')
    return {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'HTTP-Referer': OPENROUTER_SITE_URL,
        'X-Title': OPENROUTER_SITE_NAME,
        'Content-Type': 'application/json'
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        try:
            return json.dumps(content)
        except Exception:
            return ''
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    out.append(part.get('text', ''))
                elif 'text' in part:
                    out.append(str(part.get('text')))
        return "\n".join(out)
    return ''


_FENCE_RE = re.compile(r"```(?:json|JSON|Json)?\s*([\s\S]*?)\s*```", re.DOTALL)


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*(\})", r"\1", re.sub(r",\s*(\])", r"\1", s))


def _find_top_object(s: str) -> str | None:
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return s[start:i+1]
    return None


def extract_json_flexible(text_response: str) -> Dict[str, Any] | None:
    if not text_response:
        return None
    s = text_response.strip().lstrip('\ufeff')
    for m in _FENCE_RE.finditer(s):
        content = m.group(1).strip()
        try:
            return json.loads(content)
        except Exception:
            inner = _find_top_object(content)
            if inner:
                try:
                    return json.loads(inner)
                except Exception:
                    cleaned = _remove_trailing_commas(inner)
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        pass
    inner = _find_top_object(s)
    if inner:
        try:
            return json.loads(inner)
        except Exception:
            cleaned = _remove_trailing_commas(inner)
            try:
                return json.loads(cleaned)
            except Exception:
                return None
    return None


# --- Helpers specific to circles ---
def _has_any_circle(elements: List[Dict[str, Any]]) -> bool:
    return any(el.get('type') == 'circle' for el in elements)


def _find_circle_by_id(elements: List[Dict[str, Any]], cid: str) -> Dict[str, Any] | None:
    for el in elements:
        if el.get('type') == 'circle' and el.get('id') == cid:
            return el
    return None


def _apply_actions_to_state(state: Dict[str, Any], actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    applied: List[Dict[str, Any]] = []
    elements = state['elements']
    counters = state['counters']

    if not _has_any_circle(elements):
        has_initial_place = any(a.get('type') == 'PlaceCircle' and isinstance(a.get('position'), list) for a in actions)
        only_option_edits = all(a.get('type') == 'EditOptions' for a in actions) if actions else False
        if not (has_initial_place or only_option_edits):
            errors.append('First element must be a circle placed with absolute position.')
            return applied, errors

    is_first_action_in_turn = True
    for action in actions:
        atype = action.get('type')
        try:
            if atype == 'PlaceCircle':
                pos = action.get('position')
                attach_to = action.get('attach_to_circle')
                angle_cw = action.get('angle_deg')
                fill_color = str(action.get('fill_color') or DEFAULT_FILL_COLOR)
                circle_type = str(action.get('circle_type') or '').strip()
                # If no explicit circle_type provided, fall back to first available type
                if not circle_type:
                    types_av = _available_circle_types()
                    if types_av:
                        circle_type = types_av[0]
                # Default rotation: use provided rotation (if any); else if attached, align with angle
                try:
                    rotation_deg = float(action.get('rotation_deg', 0.0) or 0.0)
                except Exception:
                    rotation_deg = 0.0
                circle_radius = action.get('circle_radius')
                try:
                    r_cells = float(circle_radius) if circle_radius is not None else DEFAULT_CIRCLE_RADIUS_CELLS
                except Exception:
                    r_cells = DEFAULT_CIRCLE_RADIUS_CELLS

                if isinstance(pos, list) and len(pos) == 2:
                    if _has_any_circle(elements):
                        errors.append('Absolute placement is only allowed for the very first circle.')
                        continue
                    x = float(pos[0]); y = float(pos[1])
                    if not (0.0 <= x <= GRID_SIZE and 0.0 <= y <= GRID_SIZE):
                        errors.append(f'PlaceCircle coordinates must be within [0.0, {GRID_SIZE}.0].')
                        continue
                    # Ensure first circle gets a default image type
                    if not circle_type:
                        types_av = _available_circle_types()
                        if types_av:
                            circle_type = types_av[0]
                else:
                    if not attach_to:
                        errors.append('PlaceCircle requires either position [x,y] (first circle) or attach_to_circle + angle_deg.')
                        continue
                    anchor = _find_circle_by_id(elements, str(attach_to))
                    if not anchor:
                        errors.append(f'attach_to_circle not found: {attach_to}')
                        continue
                    try:
                        angle_cw_val = float(angle_cw) % 360.0
                    except Exception:
                        errors.append('angle_deg must be a number in [0, 360).')
                        continue
                    # If rotation not provided, align image rotation to placement angle by default
                    if (action.get('rotation_deg') is None) or (isinstance(action.get('rotation_deg'), str) and not action.get('rotation_deg').strip()):
                        rotation_deg = angle_cw_val
                    r_anchor = float(anchor.get('radius', DEFAULT_CIRCLE_RADIUS_CELLS))
                    # Convert clockwise degrees to radians for math (which uses CCW): CW deg -> rad by negating
                    ang_rad = math.radians(-angle_cw_val)
                    center_dist = r_anchor + r_cells
                    x = float(anchor['x']) + center_dist * math.cos(ang_rad)
                    y = float(anchor['y']) + center_dist * math.sin(ang_rad)

                    # Bounds: ensure entire circle is inside grid
                    if (x - r_cells) < 0.0 or (x + r_cells) > GRID_SIZE or (y - r_cells) < 0.0 or (y + r_cells) > GRID_SIZE:
                        errors.append('Proposed placement is out of bounds. Choose a different angle, smaller radius, or different circle to place the new circle against')
                        continue

                    # Center-duplicate check only: allow overlaps but forbid identical centers
                    same_center_with: str | None = None
                    for other in elements:
                        if other.get('type') != 'circle':
                            continue
                        ox, oy = float(other['x']), float(other['y'])
                        if abs(x - ox) < 1e-9 and abs(y - oy) < 1e-9:
                            same_center_with = str(other.get('id') or 'unknown')
                            break
                    if same_center_with:
                        errors.append(f'Proposed placement center coincides with existing circle {same_center_with}. Choose a different angle or radius.')
                        continue

                cid = f"C{counters['C']}"; counters['C'] += 1
                circle = {
                    'type': 'circle',
                    'id': cid,
                    'x': x,
                    'y': y,
                    'radius': r_cells,
                    'fill_color': fill_color,
                    'circle_type': circle_type,
                    'rotation_deg': rotation_deg,
                    'attached_to': str(attach_to) if attach_to else '',
                    'placed_at_angle_deg': float(angle_cw) if angle_cw is not None else None,
                }
                elements.append(circle)
                applied.append({
                    'type': 'PlaceCircle',
                    'id': cid,
                    'position': [x, y],
                    'radius': r_cells,
                    'fill_color': fill_color,
                    'circle_type': circle_type,
                    'rotation_deg': rotation_deg,
                    'existing_circle': str(attach_to) if attach_to else '',
                    'angle_deg': float(angle_cw) if angle_cw is not None else None
                })
                is_first_action_in_turn = False
                if any(True for _ in actions[actions.index(action)+1:]):
                    errors.append('Single-action rule: Only one action is allowed per AI turn. Extra actions were ignored.')
                    break
            elif atype == 'PlaceText':
                pos = action.get('position')
                text = str(action.get('text') or '').strip()
                if not (isinstance(pos, list) and len(pos) == 2 and text):
                    errors.append('PlaceText requires position [x, y] and non-empty text.')
                    continue
                x, y = float(pos[0]), float(pos[1])
                if not (0.0 <= x <= GRID_SIZE and 0.0 <= y <= GRID_SIZE):
                    errors.append(f'PlaceText coordinates must be within [0.0, {GRID_SIZE}.0].')
                    continue
                tid = f"T{counters['T']}"; counters['T'] += 1
                elements.append({'type': 'text', 'id': tid, 'x': x, 'y': y, 'text': text})
                applied.append({'type': 'PlaceText', 'id': tid, 'position': [x, y], 'text': text})
            elif atype == 'EditCircle':
                cid = action.get('id')
                target = _find_circle_by_id(elements, str(cid))
                if not target:
                    errors.append(f'EditCircle target not found: {cid}')
                    continue
                if 'position' in action and isinstance(action.get('position'), list) and len(action['position']) == 2:
                    nx, ny = float(action['position'][0]), float(action['position'][1])
                    if 0.0 <= nx <= GRID_SIZE and 0.0 <= ny <= GRID_SIZE:
                        target['x'] = nx; target['y'] = ny
                if 'radius' in action:
                    try:
                        nr = float(action['radius'])
                        if 0.5 <= nr <= 15.0:
                            target['radius'] = nr
                    except Exception:
                        pass
                if 'fill_color' in action:
                    target['fill_color'] = str(action.get('fill_color') or DEFAULT_FILL_COLOR)
                if 'circle_type' in action:
                    target['circle_type'] = str(action.get('circle_type') or '').strip()
                if 'rotation_deg' in action:
                    try:
                        target['rotation_deg'] = float(action.get('rotation_deg') or 0.0)
                    except Exception:
                        pass
                applied.append({'type': 'EditCircle', 'id': target['id'], 'position': [target['x'], target['y']], 'radius': target.get('radius', DEFAULT_CIRCLE_RADIUS_CELLS), 'fill_color': target.get('fill_color', DEFAULT_FILL_COLOR), 'circle_type': target.get('circle_type', ''), 'rotation_deg': target.get('rotation_deg', 0.0)})
            elif atype == 'EditText':
                tid = action.get('id')
                target = next((e for e in elements if e.get('type') == 'text' and e.get('id') == tid), None)
                if not target:
                    errors.append(f'EditText target not found: {tid}')
                    continue
                if 'text' in action:
                    target['text'] = str(action.get('text') or '')
                if 'position' in action and isinstance(action.get('position'), list) and len(action['position']) == 2:
                    nx, ny = float(action['position'][0]), float(action['position'][1])
                    if 0.0 <= nx <= GRID_SIZE and 0.0 <= ny <= GRID_SIZE:
                        target['x'] = nx; target['y'] = ny
                applied.append({'type': 'EditText', 'id': tid, 'text': target.get('text', ''), 'position': [target.get('x'), target.get('y')]})
            elif atype == 'DeleteCircle':
                cid = str(action.get('id'))
                idx = next((i for i, e in enumerate(elements) if e.get('type') == 'circle' and str(e.get('id')) == cid), -1)
                if idx < 0:
                    errors.append(f'DeleteCircle target not found: {cid}')
                    continue
                removed = elements.pop(idx)
                applied.append({'type': 'DeleteCircle', 'id': removed.get('id')})
            elif atype == 'RunPattern':
                errors.append('RunPattern is no longer supported. Use single-step PlaceCircle only (via AI variations).')
            elif atype == 'EditOptions':
                opts = state.get('options') or {}
                # Saturation update
                if 'saturation' in action:
                    try:
                        sval = float(action.get('saturation'))
                        sval = max(0.0, min(100.0, sval))
                        opts['saturation'] = sval
                    except Exception:
                        pass
                # Connector updates
                if 'connector' in action and isinstance(action.get('connector'), dict):
                    conn = opts.get('connector') or {}
                    cd = action.get('connector')
                    if 'enabled' in cd:
                        conn['enabled'] = bool(cd.get('enabled'))
                    # accept length_m/width_m with back-compat width/height
                    if cd.get('length_m') is not None or cd.get('width') is not None:
                        try:
                            val = float(cd.get('length_m', cd.get('width')))
                            conn['length_m'] = val
                            conn['width'] = val
                        except Exception:
                            pass
                    if cd.get('width_m') is not None or cd.get('height') is not None:
                        try:
                            val = float(cd.get('width_m', cd.get('height')))
                            conn['width_m'] = val
                            conn['height'] = val
                        except Exception:
                            pass
                    if 'fill_color' in cd:
                        conn['fill_color'] = str(cd.get('fill_color') or '#ffffff')
                    if 'type' in cd:
                        conn['type'] = str(cd.get('type') or '').strip()
                    opts['connector'] = conn
                state['options'] = opts
                applied.append({'type': 'EditOptions', 'options': state.get('options')})
            else:
                errors.append(f'Unknown action type: {atype}')
        except Exception as e:
            logging.exception(f'Error applying action {atype}')
            errors.append(f'Error applying action {atype}: {e}')

    state['elements'] = elements
    state['counters'] = counters
    return applied, errors


# --- AI Prompt ---
AI_INSTRUCTION = """
You are an expert layout assistant for a circles-only composition. The scene contains filled circles on a {GRID_SIZE_TXT} grid. Each circle has:
- An ID C1, C2, C3, ... in creation order
- A center position in grid units, a radius in grid-cell units, and a fill color

PLACEMENT RULES:
- The user places the very first circle manually (absolute coordinates). From then on, you place exactly ONE new circle per turn.
- To place a new circle, choose an existing circle ID (e.g., C2) and a clockwise angle in [0, 360). The new circle must abut the existing one: its circumference touches the existing circle at the point on the existing circle's perimeter defined by that clockwise angle, and the tangent at the point of contact is perpendicular to the radius at that angle.
- The center of the new circle is placed along that chosen radius direction, at a distance equal to r_existing + r_new from the existing circle's center (where r_new is the new circle's radius).
- Angles are measured CLOCKWISE from +X and CAN BE ANY ANGLE (not just orthogonal angles).
- You must also provide the fill color of the new circle.

CREATIVE GUIDANCE:
- Vary placement angles between turns to create visual rhythm; do not restrict to multiples of 45° or 90°.
- Vary radii deliberately (small, medium, large) to create balance and hierarchy. Typical 2.0–6.0, but you may go smaller/larger occasionally when it improves composition.
- Avoid overlaps with any existing circles (except the chosen tangent contact). If a choice collides or goes out of bounds, adjust angle and/or radius.
- Explore new regions of the canvas while maintaining tangency to the chosen existing circle.
- You may attach to any prior circle (not only the most recently placed) to develop larger-scale patterns and balance.
- Using small circles (under 2.0 grid cells) can yield intricate, interesting patterns; use large circles sparingly.

STYLE VARIATION:
- Choose a circle_type (from available_circle_types) and a fill_color on every turn; vary them across turns for interest while maintaining cohesion.
- rotation_deg is optional; if omitted the app will align the texture to your placement angle.

CREATIVE SUGGESTIONS:
- Petal/flower arcs using small circles spaced by ~25°–45° around a hub
- Box-frame made of touching circles along each side
- Spirals or chains with gradually changing radii
- Balance heavy and light areas by alternating small clusters and occasional larger anchors

ADVANCED ACTIONS YOU MAY REQUEST (append one object in actions):
- DeleteCircle: remove a circle by ID if it's harmful or mistaken.
  { "type": "DeleteCircle", "id": "C#" }
- RunPattern: execute a deterministic sequence of placements, streamed by the app (still validated per step).
  Supported patterns and parameters:
  - floral: { "type":"RunPattern", "pattern":"floral", "anchor_circle":"C#", "count": N, "circle_radius": r.f, "start_deg": deg.f (optional), "fill_color": "#RRGGBB" (optional) }
  - fractal_chain: { "type":"RunPattern", "pattern":"fractal_chain", "anchor_circle":"C#", "depth": N, "circle_radius": r0.f, "scale": s.f (0..1], "angle_deg": a0.f, "delta_deg": da.f, "fill_color": "#RRGGBB" }
  - square_ring: { "type":"RunPattern", "pattern":"square_ring", "anchor_circle":"C#", "per_side": N, "circle_radius": r.f, "start_deg": deg.f (optional), "fill_color": "#RRGGBB" (optional) }

Constraints still apply: all placed circles are tangent to an existing circle, stay within the {GRID_SIZE_TXT} grid, and avoid duplicate centers.

VALIDITY CONSTRAINTS (hard requirements):
- The entire new circle must fit inside the {GRID_SIZE_TXT} grid.
- It is acceptable for circles to overlap visually; however, do NOT place a new circle whose center is exactly the same as any existing circle's center.
- Use only IDs that already exist for attach_to_circle.

OUTPUT FORMAT (STRICT):
Return ONLY ONE JSON object in ```json fences with these keys:
{
  "actions": [
    { "type": "PlaceCircle", "attach_to_circle": "C#", "angle_deg": cw_deg.f, "circle_radius": r.f (optional), "fill_color": "css-color-or-#RRGGBB", "circle_type": "type from available_circle_types (required)", "rotation_deg": cw_deg.f (optional) }
  ],
  "turn_note": "Brief rationale"
}

Notes:
- If you omit circle_radius it defaults to a sensible value; specifying it helps creative variation.
- Do NOT invent circle IDs that do not exist.
- Place exactly one action each turn.
 - Always include a circle_type chosen from available_circle_types. rotation_deg is optional; if omitted, the app aligns it to the placement angle by default.

EXAMPLES (for guidance only; still return ONLY the JSON object above in your answer):
```json
{
  "actions": [
    { "type": "PlaceCircle", "attach_to_circle": "C1", "angle_deg": 37, "circle_radius": 3.5, "fill_color": "#ffd166" }
  ],
  "turn_note": "Place a smaller circle at a non-orthogonal 37° from C1 to start a lively arc."
}
```
```json
{
  "actions": [
    { "type": "PlaceCircle", "attach_to_circle": "C2", "angle_deg": 205, "circle_radius": 6.0, "fill_color": "cornflowerblue" }
  ],
  "turn_note": "A larger circle at 205° from C2 for lower-left weight without collisions."
}
```
```json
{
  "actions": [
    { "type": "PlaceCircle", "attach_to_circle": "C3", "angle_deg": 122.5, "circle_radius": 2.75, "fill_color": "#84cc16" }
  ],
  "turn_note": "Compact accent at 122.5° from C3, safely in-bounds and non-overlapping."
}
```
"""


def _compute_forbidden_pairs_for_anchor(anchor: Dict[str, Any], elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ax, ay = float(anchor['x']), float(anchor['y'])
    r_anchor = float(anchor.get('radius', DEFAULT_CIRCLE_RADIUS_CELLS))
    pairs: List[Dict[str, Any]] = []
    for other in elements:
        if other.get('type') != 'circle' or other is anchor:
            continue
        ox, oy = float(other['x']), float(other['y'])
        dx, dy = (ox - ax), (oy - ay)
        d = math.hypot(dx, dy)
        if d <= 0:
            continue
        # angle clockwise from +X
        ccw = math.degrees(math.atan2(dy, dx))
        angle_cw = (-ccw) % 360.0
        radius_forbidden = max(0.0, d - r_anchor)
        pairs.append({
            'attach_to_circle': anchor.get('id'),
            'angle_deg_forbidden': round(angle_cw, 6),
            'radius_forbidden': round(radius_forbidden, 6),
            'coincides_with': other.get('id')
        })
    return pairs


def _build_ai_messages(clean_b64: str, state: Dict[str, Any], history_text: str, retry_feedback: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    els = state.get('elements', [])
    circles = [e for e in els if e.get('type') == 'circle']
    occupied_centers = [[float(c.get('x')), float(c.get('y'))] for c in circles]
    per_anchor_forbidden: Dict[str, List[Dict[str, Any]]] = {}
    for c in circles:
        per_anchor_forbidden[str(c.get('id'))] = _compute_forbidden_pairs_for_anchor(c, circles)

    hints: Dict[str, Any] = {
        'ids': {'circles': [c.get('id') for c in circles]},
        'circles': [
            {
                'id': c.get('id'),
                'center': [c.get('x'), c.get('y')],
                'radius': c.get('radius'),
                'fill_color': c.get('fill_color'),
                'circle_type': c.get('circle_type', ''),
                'rotation_deg': c.get('rotation_deg', 0.0)
            } for c in circles
        ],
        'occupied_centers': occupied_centers,
        'forbidden_angle_radius_pairs': per_anchor_forbidden,
        'recommended_radii': {
            'small_often_interesting': [0.5, 2.0],
            'typical': [2.0, 6.0],
            'large_use_sparingly': [6.0, 10.0]
        },
        'next_action': 'PlaceCircle (attached)',
        'available_circle_types': _available_circle_types(),
        'available_connector_types': _available_connector_types(),
        'connector_type': (state.get('options') or {}).get('connector', {}).get('type', '')
    }
    payload = {'grid_size': GRID_SIZE, 'elements': els, 'hints': hints}
    if retry_feedback:
        payload['retry_feedback'] = retry_feedback
    state_json = json.dumps(payload, ensure_ascii=False, indent=2)
    logging.info('AI prompt context (state + hints): %s', state_json)
    sys_text_template = AI_INSTRUCTION
    sys_text = sys_text_template.replace('{GRID_SIZE_TXT}', f'{GRID_SIZE}x{GRID_SIZE}')
    call_mode_note = "Mode: VARIATION generation. Propose EXACTLY one PlaceCircle with required circle_type." if not retry_feedback else "Mode: RETRY after invalid suggestion."
    sys_text += "\n\n" + call_mode_note + "\nFULL SESSION HISTORY (latest last):\n" + history_text + "\n\nCURRENT LAYOUT JSON WITH HINTS:\n" + state_json
    contents = [
        {'type': 'text', 'text': 'Propose the next action. Return ONLY the required JSON. Use an existing circle ID, a clockwise angle in [0,360), and a fill color.'},
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{clean_b64}', 'detail': 'high'}},
    ]
    if retry_feedback:
        contents.insert(0, {'type': 'text', 'text': f"Previous action was invalid: {retry_feedback.get('error_summary','')}. Avoid exact-center duplicates and out-of-bounds. Suggest a new angle/radius and consider creativity suggestions and forbidden pairs."})
    messages = [
        {'role': 'system', 'content': sys_text},
        {'role': 'user', 'content': contents}
    ]
    return messages


def _img_to_b64(img_path: str) -> str:
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# --- Variation utilities ---
def _variations_dir(session_dir: str) -> str:
    p = os.path.join(session_dir, 'variations')
    os.makedirs(p, exist_ok=True)
    return p


def _variation_state_path(session_dir: str) -> str:
    return os.path.join(session_dir, 'variations.json')


def _read_variation_state(session_dir: str) -> Dict[str, Any]:
    p = _variation_state_path(session_dir)
    if not os.path.exists(p):
        return {'active': False}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'active': False}


def _write_variation_state(session_dir: str, vstate: Dict[str, Any]) -> None:
    with open(_variation_state_path(session_dir), 'w', encoding='utf-8') as f:
        json.dump(vstate, f, indent=2)


def _render_variation_image(session_dir: str, elements: List[Dict[str, Any]], options: Dict[str, Any], step_num: int, var_index: int, size: int = 800) -> str:
    img = Image.new('RGBA', (size, size), (*BG_COLOR, 255))
    img = _draw_elements(img, elements, with_grid=False, options=options)
    vdir = _variations_dir(session_dir)
    path = os.path.join(vdir, f"{step_num}V{var_index}.png")
    img.convert('RGB').save(path)
    return path

# --- Routes ---
@app.route('/')
def root():
    if os.path.exists(os.path.join(BASE_DIR, 'Circles.html')):
        return send_from_directory(BASE_DIR, 'Circles.html')
    return 'Circles backend is running.'


# Quiet devtools discovery noise
@app.route('/.well-known/appspecific/com.chrome.devtools.json', methods=['GET'])
def quiet_devtools_probe():
    return jsonify({'ok': True})


# Serve asset images (circle types and connector types)
@app.route('/assets/circle-types/<path:filename>')
def serve_circle_type(filename: str):
    # filename expected to include extension (e.g., name.png)
    return send_from_directory(CIRCLE_TYPES_DIR, filename)


@app.route('/assets/connector-types/<path:filename>')
def serve_connector_type(filename: str):
    return send_from_directory(CONNECTOR_TYPES_DIR, filename)


# Serve generated variation preview images
@app.route('/runs/<session_id>/variations/<path:filename>')
def serve_variation_image(session_id: str, filename: str):
    sdir = os.path.join(RUNS_DIR, session_id)
    vdir = _variations_dir(sdir)
    return send_from_directory(vdir, filename)


@app.route('/api/ai/models', methods=['GET'])
def api_models():
    cfg = _load_models_file()
    return jsonify({
        'models': cfg.get('models', []),
        'default': cfg.get('default', ''),
        'circle_types': _available_circle_types(),
        'connector_types': _available_connector_types()
    })


@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    idx = _get_next_session_index()
    sdir = _session_dir_for_index(idx)
    os.makedirs(sdir, exist_ok=True)
    state = _read_state(sdir)
    _write_state(sdir, state)
    if not os.path.exists(_history_path(sdir)):
        with open(_history_path(sdir), 'w', encoding='utf-8') as f:
            f.write(f"Session H{idx} started at {_now_iso()}\n\n")
    return jsonify({'session_id': f'H{idx}', 'state': state})


@app.route('/api/session/state/<session_id>', methods=['GET'])
def api_get_state(session_id: str):
    try:
        sdir = os.path.join(RUNS_DIR, session_id)
        state = _read_state(sdir)
        return jsonify(state)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/session/history/<session_id>', methods=['GET'])
def api_get_history(session_id: str):
    try:
        sdir = os.path.join(RUNS_DIR, session_id)
        return jsonify({'text': _read_history_text(sdir)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/session/apply_actions', methods=['POST'])
def api_apply_actions():
    data = request.json or {}
    session_id = data.get('session_id')
    actions = data.get('actions') or []
    turn_note = data.get('turn_note', '')
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    state = _read_state(sdir)
    applied, errors = _apply_actions_to_state(state, actions)
    if errors:
        return jsonify({'error': ' | '.join(errors)}), 400
    state['step_index'] = int(state.get('step_index', 0)) + 1
    _write_state(sdir, state)
    _append_history(sdir, applied, turn_note)
    _render_session_images(sdir, state)
    return jsonify({'ok': True, 'state': state, 'applied': applied})


# --- Variation lifecycle endpoints ---
@app.route('/api/session/variations/start', methods=['POST'])
def api_variations_start():
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key is not configured.'}), 500
    data = request.json or {}
    session_id = data.get('session_id')
    model = _get_selected_model(data.get('model'))
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    state = _read_state(sdir)
    # Prepare base image and prompt context
    preview_state = dict(state)
    preview_state['step_index'] = int(state.get('step_index', 0)) + 1
    clean_path, _ = _render_session_images(sdir, preview_state)
    clean_b64 = _img_to_b64(clean_path)
    history_text = _read_history_text(sdir)
    vstate = {
        'active': True,
        'model': model,
        'num': int(data.get('num', VARIATIONS_PER_STEP)),
        'generated': 0,
        'step_index': int(state.get('step_index', 0)) + 1,
        'items': [],
        'base_image_path': clean_path,
        'base_image_b64': clean_b64,
        'history_text': history_text
    }
    _write_variation_state(sdir, vstate)
    return jsonify({'ok': True, 'variation': vstate})


@app.route('/api/session/variations/next', methods=['POST'])
def api_variations_next():
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key is not configured.'}), 500
    data = request.json or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    vstate = _read_variation_state(sdir)
    if not vstate.get('active'):
        return jsonify({'error': 'No active variation session.'}), 400
    if vstate.get('generated', 0) >= int(vstate.get('num', VARIATIONS_PER_STEP)):
        return jsonify({'done': True, 'variation': vstate})

    # Load current state fresh
    state = _read_state(sdir)
    # Build messages and call AI independently for this variation
    messages = _build_ai_messages(vstate['base_image_b64'], state, vstate['history_text'])
    try:
        r = requests.post(OPENROUTER_CHAT_URL, headers=_or_headers(), json={
            'model': vstate['model'],
            'messages': messages,
            'temperature': 0.15,
            'max_tokens': 1200,
            'stream': False,
            'provider': {'allow_fallbacks': False}
        }, timeout=300)
        r.raise_for_status()
        resp = r.json()
        msg = resp['choices'][0]['message']
        text = _message_content_to_text(msg.get('content', ''))
        parsed = extract_json_flexible(text)
        if not (isinstance(parsed, dict) and isinstance(parsed.get('actions'), list)):
            return jsonify({'error': 'AI response malformed; missing actions list.'}), 200
        acts = parsed['actions']
        # Ensure each variation carries a human-readable caption for selection
        note = str(parsed.get('turn_note', 'Variation candidate'))

        # Simulate apply and render preview image, then revert (not changing real state)
        temp_state = json.loads(json.dumps(state))
        applied, errors = _apply_actions_to_state(temp_state, acts)
        if errors:
            vstate['items'].append({'applied': [], 'errors': errors, 'note': note})
        else:
            img_path = _render_variation_image(sdir, temp_state.get('elements', []), temp_state.get('options') or {}, vstate['step_index'], vstate.get('generated', 0))
            img_name = os.path.basename(img_path)
            img_url = f"/runs/{os.path.basename(sdir)}/variations/{img_name}"
            vstate['items'].append({
                'applied': applied,
                'proposal_actions': acts,
                'note': note,
                'img_path': img_path,
                'img_url': img_url
            })
        vstate['generated'] = int(vstate.get('generated', 0)) + 1
        # Append concise record to session history as a non-committing note
        try:
            _append_history(sdir, [{'type': 'VariationPreview', 'index': vstate['generated'], 'proposal_actions': acts, 'applied_preview': vstate['items'][-1].get('applied', []), 'note': note}], 'variation_preview')
        except Exception:
            pass
        _write_variation_state(sdir, vstate)
        return jsonify({'ok': True, 'variation': vstate})
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'AI request failed: {e}'}), 502
    except Exception as e:
        logging.exception('Variation next failed')
        return jsonify({'error': f'Variation next failed: {e}'}), 500


@app.route('/api/session/variations/status', methods=['GET'])
def api_variations_status():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    vstate = _read_variation_state(sdir)
    # backfill URLs if missing
    for it in vstate.get('items', []) or []:
        if it.get('img_path') and not it.get('img_url'):
            try:
                img_name = os.path.basename(it['img_path'])
                it['img_url'] = f"/runs/{session_id}/variations/{img_name}"
            except Exception:
                pass
    return jsonify({'variation': vstate})


@app.route('/api/session/variations/choose', methods=['POST'])
def api_variations_choose():
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key is not configured.'}), 500
    data = request.json or {}
    session_id = data.get('session_id')
    pick = int(data.get('pick', 0))
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    state = _read_state(sdir)
    vstate = _read_variation_state(sdir)
    if not (vstate.get('active') and vstate.get('items')):
        return jsonify({'error': 'No completed variation session to choose from.'}), 400
    items = [it for it in vstate.get('items', []) if it.get('applied')]
    if not items:
        vstate['active'] = False
        _write_variation_state(sdir, vstate)
        return jsonify({'error': 'No valid variations to choose.'}), 200
    pick = max(0, min(pick, len(items) - 1))
    chosen = items[pick]

    applied = chosen['applied']
    turn_note = chosen.get('note', '')
    real_applied, errors = _apply_actions_to_state(state, [a for a in applied])
    if errors:
        return jsonify({'error': ' | '.join(errors)}), 200
    state['step_index'] = int(state.get('step_index', 0)) + 1
    _write_state(sdir, state)
    _append_history(sdir, real_applied, turn_note)
    _render_session_images(sdir, state)

    # Close the variation session
    vstate['active'] = False
    _write_variation_state(sdir, vstate)
    return jsonify({'ok': True, 'applied': real_applied, 'turn_note': turn_note, 'state': state})


@app.route('/api/session/variations/auto_choose', methods=['POST'])
def api_variations_auto_choose():
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key is not configured.'}), 500
    data = request.json or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    vstate = _read_variation_state(sdir)
    if not (vstate.get('active') and vstate.get('items')):
        return jsonify({'error': 'No completed variation session to choose from.'}), 400
    # Collect valid items
    items = [it for it in vstate.get('items', []) if it.get('applied')]
    if not items:
        vstate['active'] = False
        _write_variation_state(sdir, vstate)
        return jsonify({'error': 'No valid variations to choose.'}), 200

    # Build chooser prompt with images
    # Build chooser prompt with ONLY captions (no images) to reduce context and avoid VLM issues
    chooser_messages = [
        {'role': 'system', 'content': 'Mode: CHOICE selection. From the candidate captions, pick the best next action that adds exactly one new circle to some existing circle with a valid clockwise angle in [0,360). Return ONLY {"pick": index_int, "reason": "..."}.'}
    ]
    chooser_lines = ["Candidates:"]
    for i, it in enumerate(items):
        caption = it.get('note', f'Variation {i}')
        chooser_lines.append(f"{i}: {caption}")
    chooser_lines.append('Pick the single best index now.')
    chooser_messages.append({'role': 'user', 'content': [{'type':'text','text':'\n'.join(chooser_lines)}]})

    try:
        r = requests.post(OPENROUTER_CHAT_URL, headers=_or_headers(), json={
            'model': vstate['model'],
            'messages': chooser_messages,
            'temperature': 0.0,
            'max_tokens': 300,
            'stream': False,
            'provider': {'allow_fallbacks': False}
        }, timeout=300)
        r.raise_for_status()
        resp = r.json()
        msg = resp['choices'][0]['message']
        text = _message_content_to_text(msg.get('content', ''))
        parsed = extract_json_flexible(text) or {}
        pick = int(parsed.get('pick', 0)) if isinstance(parsed, dict) else 0
        pick = max(0, min(pick, len(items) - 1))
    except Exception as e:
        logging.warning(f"Auto choose failed, defaulting to 0: {e}")
        pick = 0

    # Apply chosen
    state = _read_state(sdir)
    chosen = items[pick]
    # Commit the ORIGINAL proposed actions, not the preview-applied summary
    proposed_actions = chosen.get('proposal_actions') or []
    turn_note = chosen.get('note', '')
    if not isinstance(proposed_actions, list) or not proposed_actions:
        return jsonify({'error': 'Chosen variation missing proposal actions.'}), 200
    real_applied, errors = _apply_actions_to_state(state, proposed_actions)
    if errors:
        return jsonify({'error': ' | '.join(errors)}), 200
    state['step_index'] = int(state.get('step_index', 0)) + 1
    _write_state(sdir, state)
    _append_history(sdir, real_applied, turn_note)
    _render_session_images(sdir, state)
    vstate['active'] = False
    _write_variation_state(sdir, vstate)
    return jsonify({'ok': True, 'applied': real_applied, 'turn_note': turn_note, 'state': state, 'picked': pick})


@app.route('/api/session/ai_step', methods=['POST'])
def api_ai_step():
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OpenRouter API key is not configured.'}), 500
    data = request.json or {}
    session_id = data.get('session_id')
    model = _get_selected_model(data.get('model'))
    if not session_id:
        return jsonify({'error': 'Missing session_id'}), 400
    sdir = os.path.join(RUNS_DIR, session_id)
    state = _read_state(sdir)

    preview_state = dict(state)
    preview_state['step_index'] = int(state.get('step_index', 0)) + 1
    clean_path, _ = _render_session_images(sdir, preview_state)
    clean_b64 = _img_to_b64(clean_path)
    history_text = _read_history_text(sdir)

    messages = _build_ai_messages(clean_b64, state, history_text)

    try:
        # Single attempt + one automatic retry with error feedback
        def call_ai(msgs):
            r = requests.post(OPENROUTER_CHAT_URL, headers=_or_headers(), json={
                'model': model,
                'messages': msgs,
                'temperature': 0.15,
                'max_tokens': 1200,
                'stream': False,
                'provider': {'allow_fallbacks': False}
            }, timeout=300)
            r.raise_for_status();
            return r.json()

        dataj = call_ai(messages)
        msg = dataj['choices'][0]['message']
        text = _message_content_to_text(msg.get('content', ''))
        parsed = extract_json_flexible(text)
        if not (isinstance(parsed, dict) and isinstance(parsed.get('actions'), list)):
            return jsonify({'error': 'AI response malformed; missing actions list.'}), 200
        actions = parsed['actions']
        turn_note = str(parsed.get('turn_note', ''))
        applied, errors = _apply_actions_to_state(state, actions)
        if errors:
            # retry once with feedback
            retry_feedback = {'error_summary': ' | '.join(errors)}
            messages_retry = _build_ai_messages(clean_b64, state, history_text, retry_feedback=retry_feedback)
            dataj2 = call_ai(messages_retry)
            msg2 = dataj2['choices'][0]['message']
            text2 = _message_content_to_text(msg2.get('content', ''))
            parsed2 = extract_json_flexible(text2)
            if not (isinstance(parsed2, dict) and isinstance(parsed2.get('actions'), list)):
                return jsonify({'error': 'AI response malformed after retry; missing actions list.'}), 200
            actions2 = parsed2['actions']
            turn_note = str(parsed2.get('turn_note', ''))
            applied, errors = _apply_actions_to_state(state, actions2)
            if errors:
                return jsonify({'error': ' | '.join(errors)}), 200

        state['step_index'] = int(state.get('step_index', 0)) + 1
        _write_state(sdir, state)
        _append_history(sdir, applied, turn_note)
        _render_session_images(sdir, state)
        return jsonify({'ok': True, 'applied': applied, 'turn_note': turn_note, 'state': state})
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'AI request failed: {e}'}), 502
    except Exception as e:
        logging.exception('AI step failed')
        return jsonify({'error': f'AI step failed: {e}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5001'))
    app.run(debug=False, host='127.0.0.1', port=port, use_reloader=False)


