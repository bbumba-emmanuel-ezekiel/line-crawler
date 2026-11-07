import os
import io
import json
import math
import base64
import logging
import requests
import traceback
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
from dotenv import load_dotenv
import glob

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Gracefully handle CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, 'RUNS')
BG_DIR = os.path.join(BASE_DIR, 'Background Image')
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(BG_DIR, exist_ok=True)

# Configuration defaults
DEFAULT_GRID_SIZE = 55.0
DEFAULT_CREATURE_DIAMETER = 0.6
IMG_BASE_SIZE = 1200
BG_COLOR = (255, 255, 255)
CREATURE_COLOR = (0, 120, 255)
CREATURE_FILL = (0, 120, 255, 100) # Translucent blue
LINE_COLOR = (50, 50, 50, 255)

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_SITE_URL = os.getenv('OPENROUTER_SITE_URL', 'http://localhost')
OPENROUTER_SITE_NAME = os.getenv('OPENROUTER_SITE_NAME', 'Line-Crawler')
OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions'
FAL_KEY = os.getenv('FAL_KEY')

# Load Config
def get_config():
    config_path = os.path.join(BASE_DIR, 'Config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "objective": "Explore and modify the environment.",
        "grid_size": DEFAULT_GRID_SIZE,
        "creature_diameter": DEFAULT_CREATURE_DIAMETER
    }

def get_models_config():
    paths = ['Models.json', 'json.models', 'models.json']
    for p in paths:
        full = os.path.join(BASE_DIR, p)
        if os.path.exists(full):
            with open(full, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'models': data.get('models', []),
                    'default': data.get('default', data.get('models', [''])[0] if data.get('models') else '')
                }
    return {'models': [], 'default': ''}

def get_image_models_config():
    paths = [os.path.join(BASE_DIR, 'image-generation.json')]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    return {
        "default": "google/nano-banana-lite/edit",
        "models": [
            {"id": "google/nano-banana-lite/edit", "name": "Gemini 3.1 Flash Lite Image (Nano Banana Lite)"},
            {"id": "fal-ai/nano-banana-2/edit", "name": "Gemini 3.1 Flash Image (Nano Banana 2)"}
        ]
    }

def _call_fal_nano_banana(prompt: str, model: str = "google/nano-banana-lite/edit", image_b64: str = None) -> str | None:
    if not FAL_KEY:
        logging.warning("FAL_KEY is not set.")
        return None
    headers = {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": "auto",
        "output_format": "png",
        "safety_tolerance": "4",
        "limit_generations": True
    }
    if image_b64:
        payload["image_urls"] = [f"data:image/png;base64,{image_b64}"]

    try:
        logging.info(f"Submitting fal.ai image generation with model {model}...")
        resp = requests.post(f"https://fal.run/{model}", headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            res_data = resp.json()
            images = res_data.get("images", [])
            if images and len(images) > 0:
                img_url = images[0].get("url")
                logging.info(f"fal.ai image generated successfully: {img_url}")
                return img_url
        elif resp.status_code in (201, 202):
            queue_data = resp.json()
            req_id = queue_data.get("request_id")
            if req_id:
                for _ in range(30):
                    time.sleep(1)
                    st_resp = requests.get(f"https://queue.fal.run/{model}/requests/{req_id}/status", headers=headers, timeout=10)
                    if st_resp.status_code == 200:
                        st = st_resp.json().get("status")
                        if st == "COMPLETED":
                            r_resp = requests.get(f"https://queue.fal.run/{model}/requests/{req_id}", headers=headers, timeout=10)
                            if r_resp.status_code == 200:
                                imgs = r_resp.json().get("images", [])
                                if imgs:
                                    return imgs[0].get("url")
                            break
                        elif st in ("FAILED", "CANCELLED"):
                            break
    except Exception as e:
        logging.error(f"fal.ai request exception: {e}")
    return None

def get_selected_model():
    cfg = get_models_config()
    return cfg.get('default')

def _now_iso(): return datetime.now(timezone.utc).isoformat()

def _session_dir_for_index(index: int) -> str:
    return os.path.join(RUNS_DIR, f'S{index}')

def _get_next_session_index() -> int:
    existing = [d for d in os.listdir(RUNS_DIR) if d.startswith('S') and os.path.isdir(os.path.join(RUNS_DIR, d))]
    max_i = 0
    for name in existing:
        try:
            i = int(name[1:])
            max_i = max(max_i, i)
        except Exception:
            pass
    return max_i + 1

def _read_state(session_dir: str):
    p = os.path.join(session_dir, 'state.json')
    cfg = get_config()
    if not os.path.exists(p):
        return {
            'grid_size': cfg.get('grid_size', DEFAULT_GRID_SIZE),
            'creature': { 'x': cfg.get('grid_size', DEFAULT_GRID_SIZE)/2, 'y': cfg.get('grid_size', DEFAULT_GRID_SIZE)/2, 'radius': cfg.get('creature_diameter', DEFAULT_CREATURE_DIAMETER)/2 },
            'vertices': [], # List of dicts: {id, x, y}
            'polylines': [], # List of lists representing connected vertex IDs e.g. [1, 2, 4, 5]
            'shapes': [], # Generated shapes
            'step_index': 0,
            'vertex_id_counter': 1,
            'shape_id_counter': 1,
            'zoom_bounds': None # None or [min_x, min_y, max_x, max_y]
        }
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def _write_state(session_dir: str, state):
    with open(os.path.join(session_dir, 'state.json'), 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

def _append_history(session_dir: str, action, note, prompt_context="", raw_response="", is_ai=True):
    # AI Reference History
    hp = os.path.join(session_dir, 'History.json')
    history = []
    if os.path.exists(hp):
        with open(hp, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append({'action': action, 'note': note, 'is_ai': is_ai, 'ts': _now_iso()})
    with open(hp, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Developer Full Logs
    lp = os.path.join(session_dir, 'Logs.txt')
    with open(lp, 'a', encoding='utf-8') as f:
        f.write(f"--- STEP {len(history)} [{_now_iso()}] ---\n")
        f.write(f"PROMPT CONTEXT:\n{prompt_context}\n\n")
        f.write(f"RAW AI RESPONSE:\n{raw_response}\n")
        f.write(f"{'='*40}\n\n")

def _read_history(session_dir: str, max_items=5):
    hp = os.path.join(session_dir, 'History.json')
    if not os.path.exists(hp): return ''
    with open(hp, 'r', encoding='utf-8') as f:
        history = json.load(f)
        
    if not history: return ''
    
    # Take the last max_items
    history = history[-max_items:]
    
    formatted = ""
    for idx, item in enumerate(reversed(history)):
        moves_ago = idx + 1
        
        prefix = ""
        if moves_ago == 1:
            prefix = "In the most recent step this is the move you made"
        elif moves_ago == 2:
            prefix = "In the move before the most recent step this is the move you made"
        elif moves_ago == 3:
            prefix = "In the move before the move before the most recent step this is the move you made"
        else:
            prefix = f"In the move {moves_ago} steps ago this is the move you made"
            
        formatted_str = f"{prefix}:\n{json.dumps(item['action'], indent=2)}\nAnd this was your reasoning: {item['note']}\n\n"
        formatted = formatted_str + formatted
        
    return formatted

# Math and Raycasting
def ray_intersect_segment(rx, ry, rdx, rdy, p1x, p1y, p2x, p2y):
    # Ray: rx + t * rdx, ry + t * rdy, t >= 0
    # Segment: p1 + u * (p2 - p1), 0 <= u <= 1
    v1x = rx - p1x
    v1y = ry - p1y
    v2x = p2x - p1x
    v2y = p2y - p1y
    v3x = -rdy
    v3y = rdx
    
    dot = v2x * v3x + v2y * v3y
    if abs(dot) < 0.000001: return None # parallel
    
    t1 = (v2x * v1y - v2y * v1x) / dot
    t2 = (v1x * v3x + v1y * v3y) / dot
    
    if t1 >= 0.0 and (0.0 <= t2 <= 1.0):
        return t1
    return None

def calculate_raycasts(state):
    cx, cy = state['creature']['x'], state['creature']['y']
    vertices = state.get('vertices', [])
    polylines = state.get('polylines', [])
    
    # Convert polylines dict to actual segments for raycasting
    segments = []
    # Build lookup table for O(1)
    v_map = {v['id']: v for v in vertices}
    for polyline in polylines:
        for i in range(len(polyline) - 1):
            id1, id2 = polyline[i], polyline[i+1]
            if id1 in v_map and id2 in v_map:
                segments.append((v_map[id1], v_map[id2]))
                
    # Standard 10-degree rays
    angles_deg = list(range(0, 360, 10))
        
    rays_data = []
    
    # Process standard rays against polylines
    for angle in angles_deg:
        rad = math.radians(angle)
        rdx, rdy = math.cos(rad), math.sin(rad)
        min_dist = 10.0 # max distance 10m
        intersected_obj = 'None'
        for seg in segments:
            t = ray_intersect_segment(cx, cy, rdx, rdy, seg[0]['x'], seg[0]['y'], seg[1]['x'], seg[1]['y'])
            if t is not None and t < min_dist:
                min_dist = t
                intersected_obj = f"PolylineSegment"
        rays_data.append({'angle_deg': angle, 'distance': round(min_dist, 2), 'hit': intersected_obj, 'type': 'standard'})

    # Process special vertex rays
    tol = 0.01
    for v in vertices:
        vx, vy = v['x'] - cx, v['y'] - cy
        dist = math.hypot(vx, vy)
        if dist > 15.0 or dist == 0: continue # See vertices up to 15m away
        angle = math.degrees(math.atan2(vy, vx)) % 360
        # check if this is occluded
        rdx, rdy = vx/dist, vy/dist
        min_dist = dist
        intersected_obj = f"Vertex_{v['id']}"
        for seg in segments:
            t = ray_intersect_segment(cx, cy, rdx, rdy, seg[0]['x'], seg[0]['y'], seg[1]['x'], seg[1]['y'])
            if t is not None and t < min_dist - tol:
                min_dist = t
                intersected_obj = f"Vertex_{v['id']}_Occluded"
        if not 'Occluded' in intersected_obj:
            rays_data.append({'angle_deg': round(angle, 1), 'distance': round(dist, 2), 'hit': intersected_obj, 'target_pos': [v['x'], v['y']], 'type': 'special'})
            
    return rays_data

# Rendering
def _get_bg_image():
    # First image in Background Image folder
    exts = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(BG_DIR, ext)))
    if files: return files[0]
    return None

def _grid_to_px(x, y, grid_size, img_size, zoom_bounds=None):
    if zoom_bounds:
        min_x, min_y, max_x, max_y = zoom_bounds
        z_w = max_x - min_x
        z_h = max_y - min_y
        scale = img_size / max(z_w, z_h)
        # Center in view if not square
        pad_x = (img_size - z_w * scale) / 2
        pad_y = (img_size - z_h * scale) / 2
        px = (x - min_x) * scale + pad_x
        py = img_size - ((y - min_y) * scale + pad_y) 
        return px, py
    else:
        scale = img_size / grid_size
        return x * scale, img_size - (y * scale)

def _render_state(session_dir: str, state):
    step = state.get('step_index', 0)
    grid_size = state.get('grid_size', DEFAULT_GRID_SIZE)
    zb = state.get('zoom_bounds')
    
    img = Image.new('RGBA', (IMG_BASE_SIZE, IMG_BASE_SIZE), (*BG_COLOR, 255))
    
    # Background image (map its height to grid_size, maintaining aspect ratio)
    bg_path = _get_bg_image()
    if bg_path:
        try:
            bgimg = Image.open(bg_path).convert('RGBA')
            bw, bh = bgimg.size
            # scale so height represents grid_size
            # if we have zoom_bounds, background is also affected
            if zb:
                min_x, min_y, max_x, max_y = zb
                z_w = max_x - min_x
                z_h = max_y - min_y
                px_scale = IMG_BASE_SIZE / max(z_w, z_h)
                # coordinates in grid space: (0,0) to (grid_size, grid_size)
                # Background occupies x: 0 to (bw/bh)*grid_size, y: 0 to grid_size
                bg_grid_w = (bw/bh) * grid_size
                bg_grid_h = grid_size
                
                # Transform background corners to pixel space
                x0, y0 = _grid_to_px(0, 0, grid_size, IMG_BASE_SIZE, zb)
                x1, y1 = _grid_to_px(bg_grid_w, bg_grid_h, grid_size, IMG_BASE_SIZE, zb)
                # y1 is top, y0 is bottom (in image space y goes down, so y1 < y0)
                px_w = int(abs(x1 - x0))
                px_h = int(abs(y0 - y1))
                if px_w > 0 and px_h > 0:
                    bg_resized = bgimg.resize((px_w, px_h), resample=Image.BICUBIC)
                    img.paste(bg_resized, (int(x0), int(min(y0, y1))), bg_resized)
            else:
                scale = IMG_BASE_SIZE / grid_size
                bg_w = int((bw/bh) * grid_size * scale)
                bg_h = int(grid_size * scale)
                bg_resized = bgimg.resize((bg_w, bg_h), resample=Image.BICUBIC)
                img.paste(bg_resized, (0, IMG_BASE_SIZE - bg_h), bg_resized)
        except Exception as e:
            logging.error(f"Error drawing bg: {e}")

    # Draw shapes
    for shape in state.get('shapes', []):
        try:
            sp = os.path.join(session_dir, f"Shape_{shape['id']}.png")
            if os.path.exists(sp):
                si = Image.open(sp).convert('RGBA')
                sx, sy = shape['x'], shape['y']
                sw, sh = shape['w'], shape['h']
                # Transform corners
                x0, y0 = _grid_to_px(sx - sw/2, sy - sh/2, grid_size, IMG_BASE_SIZE, zb)
                x1, y1 = _grid_to_px(sx + sw/2, sy + sh/2, grid_size, IMG_BASE_SIZE, zb)
                pw, ph = int(abs(x1-x0)), int(abs(y1-y0))
                if pw>0 and ph>0:
                    si_res = si.resize((pw, ph), resample=Image.BICUBIC)
                    # Support rotation if we wanted, but let's stick to placing for now
                    img.paste(si_res, (int(min(x0,x1)), int(min(y0,y1))), si_res)
        except Exception:
            pass

    draw = ImageDraw.Draw(img)
    
    # Larger fonts for clarity
    try: font_large = ImageFont.truetype('arial.ttf', 24)
    except: font_large = ImageFont.load_default()
    try: font_grid = ImageFont.truetype('arial.ttf', 20)
    except: font_grid = ImageFont.load_default()
    
    # Draw Polylines
    polylines = state.get('polylines', [])
    vertices = state.get('vertices', [])
    v_map = {v['id']: v for v in vertices}
    
    for polyline in polylines:
        if len(polyline) < 2: continue
        points = []
        for vid in polyline:
            if vid in v_map:
                v = v_map[vid]
                px, py = _grid_to_px(v['x'], v['y'], grid_size, IMG_BASE_SIZE, zb)
                points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=(0, 255, 0, 255), width=4, joint='curve') # Green polylines
            
    # Draw Vertices
    for v in vertices:
        vx, vy = _grid_to_px(v['x'], v['y'], grid_size, IMG_BASE_SIZE, zb)
        r = 6 # radius of vertex dot
        draw.ellipse([vx-r, vy-r, vx+r, vy+r], fill=(255, 255, 0), outline=(0, 0, 0), width=2) # Yellow dot
        # Vertex ID boldly beside it
        draw.text((vx + 10, vy - 10), str(v['id']), fill=(255,0,0,255), font=font_large)

    # Draw AI Creature
    c = state.get('creature')
    if c:
        cx, cy = _grid_to_px(c['x'], c['y'], grid_size, IMG_BASE_SIZE, zb)
        # radius in pixels
        if zb:
            rpx = c['radius'] * (IMG_BASE_SIZE / max(zb[2]-zb[0], zb[3]-zb[1]))
        else:
            rpx = c['radius'] * (IMG_BASE_SIZE / grid_size)
            
        circ_bbox = [cx - rpx, cy - rpx, cx + rpx, cy + rpx]
        # Translucent fill
        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        odraw = ImageDraw.Draw(overlay)
        odraw.ellipse(circ_bbox, fill=CREATURE_FILL, outline=CREATURE_COLOR, width=3)
        img.alpha_composite(overlay)
        
        # Center anchor
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(0,0,0), outline=(255,255,255))
        
    # Draw pink grid
    grid_color = (255, 105, 180, 150) # Pink, slightly transparent
    if zb:
        # Dynamic grid spacing based on zoom level
        min_x, min_y, max_x, max_y = zb
        z_w = max_x - min_x
        z_h = max_y - min_y
        max_dim = max(z_w, z_h)
        if max_dim <= 5: spacing = 0.5
        elif max_dim <= 15: spacing = 1.0
        elif max_dim <= 30: spacing = 1.0
        else: spacing = 5.0
        
        start_x = math.floor(min_x / spacing) * spacing
        start_y = math.floor(min_y / spacing) * spacing
        
        # Vertical lines
        x = start_x
        while x <= max_x:
            px, _ = _grid_to_px(x, 0, grid_size, IMG_BASE_SIZE, zb)
            draw.line([(px, 0), (px, IMG_BASE_SIZE)], fill=grid_color, width=1)
            draw.text((px + 2, IMG_BASE_SIZE - 25), f"{x}m", fill=grid_color, font=font_grid)
            x += spacing
            
        # Horizontal lines
        y = start_y
        while y <= max_y:
            _, py = _grid_to_px(0, y, grid_size, IMG_BASE_SIZE, zb)
            draw.line([(0, py), (IMG_BASE_SIZE, py)], fill=grid_color, width=1)
            draw.text((2, py + 2), f"{y}m", fill=grid_color, font=font_grid)
            y += spacing
    else:
        # Default 55x55 grid, lines every 1m since user requested metre-level grid sizes
        spacing = 1.0
        for i in range(0, int(grid_size)+1, int(spacing)):
            px, py = _grid_to_px(i, 0, grid_size, IMG_BASE_SIZE, zb)
            # Make every 5m slightly bolder
            lw = 2 if i % 5 == 0 else 1
            draw.line([(px, 0), (px, IMG_BASE_SIZE)], fill=grid_color, width=lw)
            if i % 5 == 0:
                draw.text((px + 2, IMG_BASE_SIZE - 25), f"{i}m", fill=grid_color, font=font_grid)
            px2, py2 = _grid_to_px(0, i, grid_size, IMG_BASE_SIZE, zb)
            draw.line([(0, py2), (IMG_BASE_SIZE, py2)], fill=grid_color, width=lw)
            if i % 5 == 0:
                draw.text((2, py2 + 2), f"{i}m", fill=grid_color, font=font_grid)
            
    out_path = os.path.join(session_dir, f"{step}X.png")
    img.convert('RGB').save(out_path)
    return out_path

def get_system_prompt():
    sp_path = os.path.join(BASE_DIR, 'SystemPrompt.txt')
    if os.path.exists(sp_path):
        with open(sp_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "You are an AI."

def extract_json(text):
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            code = parts[1].split("```")[0].strip()
            try: return json.loads(code)
            except: pass
    try: return json.loads(text)
    except: return None

@app.route('/api/session/start', methods=['POST'])
def session_start():
    idx = _get_next_session_index()
    sdir = _session_dir_for_index(idx)
    os.makedirs(sdir, exist_ok=True)
    state = _read_state(sdir)
    _write_state(sdir, state)
    out_path = _render_state(sdir, state)
    with open(out_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    return jsonify({
        'session_id': f'S{idx}',
        'state': state,
        'image': f'data:image/png;base64,{img_b64}'
    })

@app.route('/api/session/state/<session_id>', methods=['GET'])
def get_state(session_id):
    sdir = os.path.join(RUNS_DIR, session_id)
    if not os.path.exists(sdir): return jsonify({'error': 'Session not found'}), 404
    state = _read_state(sdir)
    step = state.get('step_index', 0)
    out_path = os.path.join(sdir, f"{step}X.png")
    if not os.path.exists(out_path): out_path = _render_state(sdir, state)
    with open(out_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    hist = _read_history(sdir)
    rays = calculate_raycasts(state)
    return jsonify({
        'state': state,
        'image': f'data:image/png;base64,{img_b64}',
        'history': hist,
        'rays': rays
    })

def apply_action(state, sdir, action_obj):
    action = action_obj.get('action', {})
    atype = action.get('type')
    
    if atype == 'MoveCreature':
        tp = action.get('target_position')
        if tp and len(tp) == 2:
            state['creature']['x'] = float(tp[0])
            state['creature']['y'] = float(tp[1])
    elif atype == 'MoveVertexExact':
        vid = action.get('vertex_id')
        vn = action.get('vertex_new')
        if vid is not None and vn is not None and len(vn) == 2:
            for v in state['vertices']:
                if v['id'] == vid:
                    v['x'], v['y'] = float(vn[0]), float(vn[1])
                    state['creature']['x'], state['creature']['y'] = v['x'], v['y']
                    break
    elif atype == 'DisplaceVertex':
        vid = action.get('vertex_id')
        dx = action.get('dx', 0.0)
        dy = action.get('dy', 0.0)
        if vid is not None:
            for v in state['vertices']:
                if v['id'] == vid:
                    v['x'] += float(dx)
                    v['y'] += float(dy)
                    state['creature']['x'], state['creature']['y'] = v['x'], v['y']
                    break
    elif atype == 'DeleteVertex':
        vid = action.get('vertex_id')
        state['vertices'] = [v for v in state['vertices'] if v['id'] != vid]
        # Clean up any simple polylines that referenced this vertex (optional but good practice)
        new_polylines = []
        for p in state.get('polylines', []):
            if vid not in p:
                new_polylines.append(p)
        state['polylines'] = new_polylines
    elif atype == 'GenerateVertex':
        angle = action.get('angle_deg', 0)
        length = action.get('length', 1.0)
        rad = math.radians(angle)
        cx, cy = state['creature']['x'], state['creature']['y']
        nx, ny = cx + length * math.cos(rad), cy + length * math.sin(rad)
        vid = state.get('vertex_id_counter', 1)
        state['vertices'].append({
            'id': vid,
            'x': float(nx),
            'y': float(ny)
        })
        state['vertex_id_counter'] = vid + 1
        # Move creature to the new vertex
        state['creature']['x'], state['creature']['y'] = float(nx), float(ny)
    elif atype == 'GeneratePolyline':
        vids = action.get('vertex_ids', [])
        if isinstance(vids, list) and len(vids) >= 2:
            if 'polylines' not in state: state['polylines'] = []
            state['polylines'].append(vids)
    elif atype == 'Zoom':
        zb = action.get('bounds')
        if zb and len(zb) == 4:
            state['zoom_bounds'] = zb
        else:
            state['zoom_bounds'] = None
    elif atype == 'GenerateShape':
        sid = state['shape_id_counter']
        state['shape_id_counter'] += 1
        prompt = action.get('prompt', 'Spatial architectural diagram')
        model = action.get('image_model', 'google/nano-banana-lite/edit')
        img_url = _call_fal_nano_banana(prompt, model=model)
        
        sp = os.path.join(sdir, f"Shape_{sid}.png")
        if img_url:
            try:
                r = requests.get(img_url, timeout=15)
                if r.status_code == 200:
                    with open(sp, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                logging.warning(f"Could not download fal image: {e}")
        
        if not os.path.exists(sp):
            si = Image.new('RGBA', (200, 200), (58, 150, 221, 120))
            sd = ImageDraw.Draw(si)
            sd.text((45, 90), f"SHAPE {sid}", fill=(255, 255, 255))
            si.save(sp)
        
        state['shapes'].append({
            'id': sid,
            'x': action.get('x', 15),
            'y': action.get('y', 15),
            'w': action.get('w', 10),
            'h': action.get('h', 10),
            'url': img_url
        })
    elif atype == 'ModifyShape':
        sid = action.get('shape_id')
        for s in state['shapes']:
            if s['id'] == sid:
                s['x'] = action.get('x', s['x'])
                s['y'] = action.get('y', s['y'])
                s['w'] = action.get('w', s['w'])
                s['h'] = action.get('h', s['h'])
                
    state['step_index'] += 1
    
@app.route('/')
def root():
    if os.path.exists(os.path.join(BASE_DIR, 'Interface.html')):
        return send_from_directory(BASE_DIR, 'Interface.html')
    return 'Line Crawler backend is running.'

@app.route('/.well-known/appspecific/com.chrome.devtools.json', methods=['GET'])
def quiet_devtools_probe():
    return jsonify({'ok': True})

@app.route('/api/session/step', methods=['POST'])
def ai_step():
    data = request.json or {}
    session_id = data.get('session_id')
    sdir = os.path.join(RUNS_DIR, session_id)
    if not os.path.exists(sdir): return jsonify({'error': 'Session not found'}), 404
    
    state = _read_state(sdir)
    cfg = get_config()
    history_depth = cfg.get('history_depth', 5)
    
    rays = calculate_raycasts(state)
    hist = _read_history(sdir, max_items=history_depth)
    
    out_path = _render_state(sdir, state)
    with open(out_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
    # Filter rays to only send relevant data to the AI (hits and special)
    filtered_rays = [r for r in rays if r.get('distance', 10.0) < 10.0 or r.get('type') == 'special']
    
    prompt_context = f"As you can see in the LATEST image of the world space, this is your current state:\n"
    prompt_context += f"OBJECTIVE: {cfg.get('objective')}\n"
    prompt_context += f"CREATURE STATE: Position [{state['creature']['x']}, {state['creature']['y']}]\n"
    prompt_context += f"RAYS DATA (Filtered for hits/special): {json.dumps(filtered_rays)}\n"
    prompt_context += f"RECENT MOVES:\n{hist}\n"
    prompt_context += "Assess the environment and reason about what to do next to fulfill the objective. Respond with exactly ONE required JSON action."
    
    messages = [
        {'role': 'system', 'content': get_system_prompt()},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': prompt_context},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}}
        ]}
    ]
    
    model = get_selected_model()
    if not OPENROUTER_API_KEY:
        return jsonify({'error': 'OPENROUTER_API_KEY is missing in .env'}), 500
        
    try:
        logging.info(f"Calling OpenRouter with model: {model}")
        r = requests.post(OPENROUTER_CHAT_URL, headers={
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'HTTP-Referer': OPENROUTER_SITE_URL,
            'X-Title': OPENROUTER_SITE_NAME,
            'Content-Type': 'application/json'
        }, json={
            'model': model,
            'messages': messages,
            'temperature': 0.3
        }, timeout=120)
        
        if r.status_code != 200:
            logging.error(f"OpenRouter Error {r.status_code}: {r.text}")
            return jsonify({'error': f"OpenRouter Error {r.status_code}", 'details': r.text}), 500
            
        resp = r.json()
        raw_text = resp['choices'][0]['message'].get('content', '')
        action_obj = extract_json(raw_text)
        
        if not action_obj:
            logging.error(f"Failed to parse JSON from AI: {raw_text}")
            return jsonify({'error': 'Failed to parse JSON from AI response', 'raw': raw_text}), 500
            
        apply_action(state, sdir, action_obj)
        _write_state(sdir, state)
        _append_history(sdir, action_obj.get('action'), action_obj.get('reasoning'), prompt_context=prompt_context, raw_response=raw_text)
        
        out_path = _render_state(sdir, state)
        with open(out_path, 'rb') as f:
            new_img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
        return jsonify({
            'ok': True,
            'action': action_obj,
            'state': state,
            'image': f'data:image/png;base64,{new_img_b64}',
            'rays': calculate_raycasts(state),
            'history': _read_history(sdir)
        })
        
    except requests.exceptions.RequestException as re:
        logging.error(f"OpenRouter Connection Error: {re}")
        return jsonify({'error': 'Connection to OpenRouter failed', 'details': str(re)}), 500
    except Exception as e:
        trace = traceback.format_exc()
        logging.error(f"Internal Server Error: {e}\n{trace}")
@app.route('/api/models', methods=['GET'])
@app.route('/api/ai/models', methods=['GET'])
def api_get_models():
    return jsonify(get_models_config())

@app.route('/api/image-models', methods=['GET'])
def api_get_image_models():
    return jsonify(get_image_models_config())

@app.route('/api/generate-image', methods=['POST'])
def api_direct_generate_image():
    body = request.json or {}
    prompt = body.get('prompt', 'Abstract spatial architectural composition')
    model = body.get('model', 'google/nano-banana-lite/edit')
    url = _call_fal_nano_banana(prompt, model=model)
    if url:
        return jsonify({'ok': True, 'url': url})
    return jsonify({'error': 'Image generation failed or FAL_KEY not configured.'}), 500

# --- Blueprint Factory for Root Core.py ---
def create_blueprint():
    """Return a Flask Blueprint for mounting under a parent Core.py server at /api/line-crawler.

    Routes are stripped of the /api/ prefix — Core.py adds it at registration.
    """
    from flask import Blueprint
    bp = Blueprint('line_crawler', __name__, url_prefix='')
    for rule in app.url_map.iter_rules():
        orig = rule.rule
        if rule.endpoint == 'static':
            continue
        if orig.startswith('/api/'):
            new_rule = orig[4:]  # strip '/api'
        elif orig.startswith('/runs/') or orig.startswith('/assets/') or orig.startswith('/Background'):
            new_rule = orig
        else:
            continue
        view_func = app.view_functions.get(rule.endpoint)
        if view_func:
            bp.add_url_rule(new_rule, rule.endpoint, view_func, methods=list(rule.methods))
    return bp


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(debug=False, host='127.0.0.1', port=port, use_reloader=False)
