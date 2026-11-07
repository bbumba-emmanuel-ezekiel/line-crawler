"""
LINE-CRAWLER — Autonomous AI Architectural Line Draughtsman (Backend)
======================================================================
Iterative draughtsman engine where the AI traces, micro-fixes, and refines
meaningful architectural lines over an uploaded reference image.

Key features:
- 20×20 decimal coordinate grid reference system (0.0 to 20.0) with visible
  numeric labels and ticks rendered on the reference image canvas.
- Line object management with persistent IDs and micro-fixing actions:
    * "Add-line": Create a new line segment [x1, y1] to [x2, y2]
    * "Modify-line": Micro-adjust an existing line by ID to correct positioning
    * "Delete-line": Remove a misplaced line by ID
- Visual feedback loop: Canvas with dimmed reference image + grid ticks +
  drawn lines + line ID badges is rendered via Pillow and sent to multimodal AI.
- Multi-turn memory: Retains what was done in previous steps vs current step.
- Full session downloads: PNG, SVG vector overlay, and ZIP of iteration states.
"""

import os
import io
import json
import logging
import math
import base64
import re
import uuid
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from flask import Flask, jsonify, request, Blueprint, send_file, send_from_directory

# ─── Paths & env ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")
load_dotenv(dotenv_path=BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("LINE-CRAWLER")

# ─── Grid & Drawing Configuration ─────────────────────────────────────────────
GRID_SIZE = 20                         # Coordinate range: 0.0 to 20.0
IMG_SIZE = 840                         # Rendered PNG size in pixels
INTERNAL_MARGIN = 30                   # Margin for grid tick numbers
GRID_AREA_PX = IMG_SIZE - 2 * INTERNAL_MARGIN  # 780px
CELL_SIZE = GRID_AREA_PX / GRID_SIZE   # 39px per unit

LINE_PALETTE = {
    "structural": {"color": (20, 20, 20), "width": 3, "svg": "#141414"},
    "outline": {"color": (45, 55, 72), "width": 2, "svg": "#2d3748"},
    "detail": {"color": (113, 128, 150), "width": 1, "svg": "#718096"},
    "contour": {"color": (58, 150, 221), "width": 2, "svg": "#3a96dd"},
    "annotation": {"color": (217, 119, 6), "width": 1, "svg": "#d97706"},
    "default": {"color": (30, 30, 30), "width": 2, "svg": "#1e1e1e"}
}

# ─── Session store ─────────────────────────────────────────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}

def load_models_config() -> dict:
    for p in [BASE_DIR / "models.json", ROOT_DIR / "models.json"]:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {
        "default": "google/gemini-2.5-flash-lite-preview-06-17",
        "models": [
            "google/gemini-2.5-flash-lite-preview-06-17",
            "google/gemini-2.5-flash",
            "anthropic/claude-sonnet-4-5",
            "openai/gpt-4o-mini"
        ]
    }

def _grid_to_px(coord: float) -> float:
    return INTERNAL_MARGIN + (coord * CELL_SIZE)

def _load_system_font(size: int):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()

# ─── Pillow Rendering Engine with Visual Grid Reference ────────────────────────
def _render_composite_canvas(
    original_b64: Optional[str],
    lines_dict: Dict[int, Dict[str, Any]],
    show_grid: bool = True,
    show_labels: bool = True,
    dim_photo: float = 0.55
) -> bytes:
    """
    Renders the composite image:
    1. White background + Dimmed reference photo scaled to 780x780 grid area
    2. Subtle grid reference lines with coordinate numbers 0-20
    3. Active line segments with color coding
    4. Numeric ID badges near line midpoints for micro-fixing reference
    """
    canvas = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (255, 255, 255, 255))
    
    # 1. Overlay reference photo if provided
    if original_b64:
        try:
            photo_bytes = base64.b64decode(original_b64)
            photo = Image.open(io.BytesIO(photo_bytes)).convert("RGBA")
            photo_fitted = photo.resize((int(GRID_AREA_PX), int(GRID_AREA_PX)), Image.Resampling.LANCZOS)
            
            # Dim photo slightly with white layer
            dim_layer = Image.new("RGBA", photo_fitted.size, (255, 255, 255, int(255 * (1.0 - dim_photo))))
            photo_dimmed = Image.alpha_composite(photo_fitted, dim_layer)
            
            canvas.paste(photo_dimmed, (INTERNAL_MARGIN, INTERNAL_MARGIN), photo_dimmed)
        except Exception as e:
            log.warning(f"Failed to composite photo: {e}")

    draw = ImageDraw.Draw(canvas)
    axis_font = _load_system_font(max(9, int(CELL_SIZE * 0.35)))
    badge_font = _load_system_font(max(8, int(CELL_SIZE * 0.28)))

    # 2. Grid lines & labels
    if show_grid:
        grid_color = (180, 190, 205, 120)
        border_color = (100, 110, 125, 200)
        
        for i in range(GRID_SIZE + 1):
            px = _grid_to_px(i)
            # Vertical line
            draw.line([(px, INTERNAL_MARGIN), (px, IMG_SIZE - INTERNAL_MARGIN)], fill=grid_color, width=1)
            # Horizontal line
            draw.line([(INTERNAL_MARGIN, px), (IMG_SIZE - INTERNAL_MARGIN, px)], fill=grid_color, width=1)

        # Draw outer boundary
        draw.rectangle(
            [INTERNAL_MARGIN, INTERNAL_MARGIN, IMG_SIZE - INTERNAL_MARGIN, IMG_SIZE - INTERNAL_MARGIN],
            outline=border_color, width=2
        )

        # Coordinate numbers on Top (X) and Left (Y)
        for i in range(GRID_SIZE + 1):
            cx = _grid_to_px(i)
            # Top axis
            draw.text((cx, INTERNAL_MARGIN - 14), str(i), fill=(60, 70, 85, 255), font=axis_font, anchor="mm")
            # Left axis
            draw.text((INTERNAL_MARGIN - 14, cx), str(i), fill=(60, 70, 85, 255), font=axis_font, anchor="mm")

    # 3. Draw active line segments
    for line_id, line in sorted(lines_dict.items()):
        try:
            x1, y1 = float(line["start"][0]), float(line["start"][1])
            x2, y2 = float(line["end"][0]), float(line["end"][1])
            ltype = str(line.get("type", "outline")).lower()
            style = LINE_PALETTE.get(ltype, LINE_PALETTE["default"])
            
            px1, py1 = _grid_to_px(x1), _grid_to_px(y1)
            px2, py2 = _grid_to_px(x2), _grid_to_px(y2)
            lw = style["width"]
            color = style["color"]
            
            # Shadow for contrast
            draw.line([(px1, py1), (px2, py2)], fill=(255, 255, 255, 200), width=lw + 2)
            # Primary line
            draw.line([(px1, py1), (px2, py2)], fill=(*color, 255), width=lw)
            
            # Draw midpoint ID tag for micro-fixing awareness
            if show_labels:
                mx, my = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                tag_text = str(line_id)
                tag_w = max(14, len(tag_text) * 7 + 6)
                tag_h = 13
                # Small white pill
                draw.rectangle([mx - tag_w/2, my - tag_h/2, mx + tag_w/2, my + tag_h/2], fill=(255, 255, 255, 230), outline=color, width=1)
                draw.text((mx, my), tag_text, fill=(20, 20, 20, 255), font=badge_font, anchor="mm")
        except Exception as e:
            log.warning(f"Error rendering line {line_id}: {e}")

    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ─── AI Draughtsman Prompt Template with Micro-Fixing Protocol ─────────────────
CRAWL_PROMPT_TEMPLATE = """
INSTRUCTIONS:
You are an expert architectural draughtsman tracing precise vector line segments over a reference drawing/image.
You have access to a 20x20 Cartesian grid (X: 0.0 to 20.0, Y: 0.0 to 20.0) with visible numeric ticks and grid lines.

COORDINATE SYSTEM:
- Coordinates are floating-point numbers in the range [0.0, 20.0].
- X ranges from 0.0 (left) to 20.0 (right).
- Y ranges from 0.0 (top) to 20.0 (bottom).
- You can use high decimal precision (e.g., [4.35, 11.20]) to position lines accurately.

DRAWING & MICRO-FIXING METHODOLOGY:
1. Examine the current state image showing the reference photo with existing lines and their ID numbers.
2. Determine what architectural features to trace next (road edges, building footprints, walls, outlines, curbs, contour lines).
3. If an existing line is slightly misplaced or misaligned, you can MODIFY it using its ID to micro-fix its start or end coordinates.
4. If a line is erroneous or unneeded, you can DELETE it by its ID.
5. Trace clean, continuous segments (5 to 15 actions per turn). Connect adjoining lines at shared vertex coordinates.

AVAILABLE LINE TYPES:
- "structural": Major structural outlines/curbs/footprints (bold black, width 3)
- "outline": Primary boundaries and exterior edges (dark slate, width 2)
- "detail": Secondary details, divisions, markings (medium grey, width 1)
- "contour": Curving contours, paths, natural boundaries (blueprint blue, width 2)
- "annotation": Guides and reference lines (amber, width 1)

RESPONSE FORMAT:
Respond with ONLY a single valid JSON object enclosed in ```json ... ``` markers:
```json
{{
  "Actions": [
    {{
      "Action": "Add-line",
      "Start": [4.20, 8.50],
      "End": [6.80, 8.50],
      "Type": "structural",
      "Note": "Tracing south curb edge of roadway"
    }},
    {{
      "Action": "Modify-line",
      "Id": 2,
      "Start": [6.80, 8.50],
      "End": [9.10, 8.90],
      "Note": "Micro-adjusting line 2 to smoothly match the road curvature"
    }},
    {{
      "Action": "Delete-line",
      "Id": 5,
      "Note": "Removing misplaced stray line"
    }}
  ],
  "Turn-Note": "Detailed description of what you observed, what was newly traced, and any micro-fixes performed."
}}
```

ACTIVE LINES ON CANVAS ({active_lines_count} total):
{active_lines_summary}

SESSION HISTORY (Recent Turns):
{action_history}

Now analyze the reference image and coordinate grid, and formulate your next set of line drafting and micro-fixing actions in JSON.
"""

def _extract_json_block(text: str) -> Optional[dict]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None

def _call_openrouter(prompt: str, model: str, screenshot_b64: str) -> Optional[dict]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        return None

    user_content = [
        {"type": "text", "text": "Analyze the reference image with grid overlay and active line ID tags. Execute precise drafting and micro-fixing actions."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}", "detail": "high"}}
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 3000,
        "temperature": 0.35
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "LINE-CRAWLER"
    }

    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return _extract_json_block(raw)
    except Exception as e:
        log.error(f"OpenRouter call failed: {e}")
        return None

# ─── Action Processing (Add, Modify, Delete) ───────────────────────────────────
def _apply_crawl_actions(lines_dict: Dict[int, Dict[str, Any]], next_id: int, actions: list) -> Tuple[int, int, List[str]]:
    """Applies Add-line, Modify-line, and Delete-line actions. Returns (applied, next_id, errors)."""
    applied = 0
    errors = []

    for idx, act in enumerate(actions):
        action_name = str(act.get("Action") or act.get("type") or "").strip().lower()
        
        # 1. Add line
        if action_name in ("add-line", "add", "placeline"):
            start = act.get("Start") or act.get("start")
            end = act.get("End") or act.get("end")
            if not (isinstance(start, (list, tuple)) and len(start) == 2 and isinstance(end, (list, tuple)) and len(end) == 2):
                # Check flat x1, y1, x2, y2 format
                if all(k in act for k in ("x1", "y1", "x2", "y2")):
                    start = [float(act["x1"]), float(act["y1"])]
                    end = [float(act["x2"]), float(act["y2"])]
                else:
                    errors.append(f"Action {idx+1}: Missing Start [x1,y1] or End [x2,y2]")
                    continue
            try:
                x1, y1 = float(start[0]), float(start[1])
                x2, y2 = float(end[0]), float(end[1])
            except Exception:
                errors.append(f"Action {idx+1}: Non-numeric coordinates")
                continue
            
            # Clamp bounds
            x1, y1 = max(0.0, min(float(GRID_SIZE), x1)), max(0.0, min(float(GRID_SIZE), y1))
            x2, y2 = max(0.0, min(float(GRID_SIZE), x2)), max(0.0, min(float(GRID_SIZE), y2))
            
            ltype = str(act.get("Type") or act.get("stroke") or "outline").lower()
            lid = next_id
            next_id += 1
            lines_dict[lid] = {
                "id": lid,
                "start": [round(x1, 2), round(y1, 2)],
                "end": [round(x2, 2), round(y2, 2)],
                "type": ltype,
                "note": str(act.get("Note", ""))
            }
            applied += 1

        # 2. Modify line
        elif action_name in ("modify-line", "modify", "edit", "editline"):
            target_id = act.get("Id") or act.get("id")
            try:
                target_id = int(target_id)
            except Exception:
                target_id = None
            if target_id is None or target_id not in lines_dict:
                errors.append(f"Action {idx+1}: Line ID {target_id} not found to modify")
                continue
            
            start = act.get("Start") or act.get("start")
            end = act.get("End") or act.get("end")
            if start and len(start) == 2:
                lines_dict[target_id]["start"] = [round(float(start[0]), 2), round(float(start[1]), 2)]
            if end and len(end) == 2:
                lines_dict[target_id]["end"] = [round(float(end[0]), 2), round(float(end[1]), 2)]
            if "Type" in act:
                lines_dict[target_id]["type"] = str(act["Type"]).lower()
            applied += 1

        # 3. Delete line
        elif action_name in ("delete-line", "delete", "remove"):
            target_id = act.get("Id") or act.get("id")
            try:
                target_id = int(target_id)
            except Exception:
                target_id = None
            if target_id is not None and target_id in lines_dict:
                del lines_dict[target_id]
                applied += 1
            else:
                errors.append(f"Action {idx+1}: Line ID {target_id} not found to delete")

    return applied, next_id, errors


# ─── Session Helpers ───────────────────────────────────────────────────────────
def _make_session(sid: str, model: str) -> Dict[str, Any]:
    return {
        "session_id": sid,
        "model": model,
        "original_b64": None,
        "original_filename": "",
        "lines": {},                   # Dict[int, Dict]: line_id -> line object
        "next_line_id": 1,
        "iteration_count": 0,
        "screenshots": [],             # List of base64 composite images
        "history": [],                 # List of turn summaries
        "turn_notes": []
    }

def _session_response(sess: Dict[str, Any], composite_b64: str) -> Dict[str, Any]:
    lines_list = list(sess["lines"].values())
    return {
        "session_id": sess["session_id"],
        "model": sess["model"],
        "iteration_count": sess["iteration_count"],
        "line_count": len(sess["lines"]),
        "has_image": sess["original_b64"] is not None,
        "original_filename": sess["original_filename"],
        "composite_b64": composite_b64,
        "turn_note": sess["turn_notes"][-1] if sess["turn_notes"] else "",
        "lines": lines_list
    }


# ─── Route Handlers ────────────────────────────────────────────────────────────
def _register_routes(bp_or_app):

    @bp_or_app.route("/models", methods=["GET"])
    @bp_or_app.route("/ai/models", methods=["GET"])
    @bp_or_app.route("/image-models", methods=["GET"])
    def lc_models():
        return jsonify(load_models_config())

    @bp_or_app.route("/session/start", methods=["POST"])
    def lc_session_start():
        body = request.get_json(silent=True) or {}
        sid = uuid.uuid4().hex[:12]
        model = body.get("model") or load_models_config().get("default", "google/gemini-2.5-flash")
        sess = _make_session(sid, model)
        _sessions[sid] = sess
        
        # Initial empty composite
        composite_bytes = _render_composite_canvas(None, sess["lines"])
        b64 = base64.b64encode(composite_bytes).decode("utf-8")
        sess["screenshots"].append(b64)
        
        log.info(f"LINE-CRAWLER session started: {sid}")
        return jsonify(_session_response(sess, b64))

    @bp_or_app.route("/session/upload", methods=["POST"])
    def lc_upload():
        sid = request.form.get("session_id") or (list(_sessions.keys())[-1] if _sessions else None)
        if not sid or sid not in _sessions:
            sid = uuid.uuid4().hex[:12]
            model = load_models_config().get("default", "google/gemini-2.5-flash")
            _sessions[sid] = _make_session(sid, model)

        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            original_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            sess = _sessions[sid]
            sess["original_b64"] = original_b64
            sess["original_filename"] = file.filename
            sess["lines"] = {}
            sess["next_line_id"] = 1
            sess["iteration_count"] = 0
            sess["screenshots"] = []
            sess["turn_notes"] = []
            sess["history"] = ["Reference image loaded."]

            # Render initial composite with grid overlay
            composite_bytes = _render_composite_canvas(original_b64, sess["lines"])
            composite_b64 = base64.b64encode(composite_bytes).decode("utf-8")
            sess["screenshots"].append(composite_b64)

            log.info(f"LINE-CRAWLER session {sid}: loaded image '{file.filename}'")
            return jsonify(_session_response(sess, composite_b64))
        except Exception as e:
            log.error(f"Upload failed: {e}")
            return jsonify({"error": str(e)}), 500

    @bp_or_app.route("/session/step", methods=["POST"])
    def lc_step():
        body = request.get_json(silent=True) or {}
        sid = body.get("session_id")
        if not sid or sid not in _sessions:
            if _sessions:
                sid = list(_sessions.keys())[-1]
            else:
                return jsonify({"error": "No active session. Please start and upload an image first."}), 400

        sess = _sessions[sid]
        sess["iteration_count"] += 1
        model = body.get("model") or sess.get("model") or load_models_config().get("default", "google/gemini-2.5-flash")
        sess["model"] = model

        # 1. Render current state image for AI vision
        composite_bytes = _render_composite_canvas(sess["original_b64"], sess["lines"])
        current_b64 = base64.b64encode(composite_bytes).decode("utf-8")

        # 2. Build prompt context
        lines_summary = []
        for lid, l in sorted(sess["lines"].items()):
            lines_summary.append(f"  Line #{lid} ({l.get('type', 'outline')}): Start={l['start']}, End={l['end']}")
        active_summary_str = "\n".join(lines_summary) if lines_summary else "  (None yet — begin by drafting primary structural outlines)"

        history_str = "\n".join(sess["history"][-5:]) if sess["history"] else "  (Session started)"

        prompt = CRAWL_PROMPT_TEMPLATE.format(
            active_lines_count=len(sess["lines"]),
            active_lines_summary=active_summary_str,
            action_history=history_str
        )

        # 3. Call AI
        log.info(f"LINE-CRAWLER step {sess['iteration_count']} calling AI ({model})...")
        ai_res = _call_openrouter(prompt, model, current_b64)

        if not ai_res or not (isinstance(ai_res.get("Actions"), list) or isinstance(ai_res.get("lines"), list)):
            return jsonify({
                "error": "AI model did not return a valid drafting action set.",
                "iteration_count": sess["iteration_count"],
                "composite_b64": current_b64
            }), 500

        actions = ai_res.get("Actions") or ai_res.get("lines") or []
        turn_note = str(ai_res.get("Turn-Note") or ai_res.get("note") or "Executed line drafting pass.")

        # 4. Apply actions (Add / Modify / Delete)
        applied_count, sess["next_line_id"], errors = _apply_crawl_actions(
            sess["lines"], sess["next_line_id"], actions
        )

        sess["turn_notes"].append(turn_note)
        sess["history"].append(f"Step {sess['iteration_count']}: {applied_count} actions applied. {turn_note}")

        # 5. Render updated canvas
        new_composite_bytes = _render_composite_canvas(sess["original_b64"], sess["lines"])
        new_b64 = base64.b64encode(new_composite_bytes).decode("utf-8")
        sess["screenshots"].append(new_b64)

        log.info(f"LINE-CRAWLER step {sess['iteration_count']}: {applied_count} actions applied, {len(sess['lines'])} active lines.")
        
        resp = _session_response(sess, new_b64)
        resp["actions_applied"] = applied_count
        resp["action_errors"] = errors
        return jsonify(resp)

    @bp_or_app.route("/session/composite", methods=["GET"])
    def lc_composite():
        sid = request.args.get("session_id")
        if not sid or sid not in _sessions:
            if _sessions:
                sid = list(_sessions.keys())[-1]
            else:
                return jsonify({"error": "No active session"}), 404
        sess = _sessions[sid]
        png_bytes = _render_composite_canvas(sess["original_b64"], sess["lines"])
        return send_file(io.BytesIO(png_bytes), mimetype="image/png", as_attachment=False, download_name="linecrawler.png")

    @bp_or_app.route("/session/download-svg", methods=["GET"])
    def lc_download_svg():
        sid = request.args.get("session_id")
        if not sid or sid not in _sessions:
            if _sessions:
                sid = list(_sessions.keys())[-1]
            else:
                return jsonify({"error": "No active session"}), 404
        sess = _sessions[sid]
        
        svg_elements = []
        for lid, l in sess["lines"].items():
            stype = l.get("type", "outline")
            style = LINE_PALETTE.get(stype, LINE_PALETTE["default"])
            color = style["svg"]
            w = style["width"]
            x1, y1 = l["start"][0], l["start"][1]
            x2, y2 = l["end"][0], l["end"][1]
            svg_elements.append(f'  <line id="line_{lid}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w*0.08:.3f}" stroke-linecap="round"/>')

        svg = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {GRID_SIZE} {GRID_SIZE}" width="800" height="800">\n'
            f'  <rect width="{GRID_SIZE}" height="{GRID_SIZE}" fill="#ffffff"/>\n'
            + "\n".join(svg_elements)
            + "\n</svg>\n"
        )
        return send_file(io.BytesIO(svg.encode("utf-8")), mimetype="image/svg+xml", as_attachment=True, download_name=f"LineCrawler_{sid}.svg")

    @bp_or_app.route("/session/download-zip", methods=["GET"])
    @bp_or_app.route("/session/download-zip/<session_id>", methods=["GET"])
    def lc_download_zip(session_id=None):
        sid = session_id or request.args.get("session_id")
        if not sid or sid not in _sessions:
            if _sessions:
                sid = list(_sessions.keys())[-1]
            else:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("info.txt", "No session active.")
                zip_buf.seek(0)
                return send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name="LineCrawler_session.zip")
        sess = _sessions[sid]

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, b64 in enumerate(sess.get("screenshots", [])):
                png_bytes = base64.b64decode(b64)
                label = "initial" if idx == 0 else f"step_{idx:03d}"
                zf.writestr(f"LineCrawler_{sid}_{label}.png", png_bytes)
            
            lines_json = json.dumps({"session_id": sid, "lines": list(sess["lines"].values())}, indent=2)
            zf.writestr(f"LineCrawler_{sid}_lines.json", lines_json)

        zip_buf.seek(0)
        return send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name=f"LineCrawler_{sid}.zip")

    @bp_or_app.route("/session/reset", methods=["POST"])
    def lc_reset():
        body = request.get_json(silent=True) or {}
        sid = body.get("session_id")
        if not sid or sid not in _sessions:
            return jsonify({"error": "Session not found"}), 404
        orig = _sessions[sid].get("original_b64")
        fname = _sessions[sid].get("original_filename", "")
        model = _sessions[sid].get("model")
        _sessions[sid] = _make_session(sid, model)
        _sessions[sid]["original_b64"] = orig
        _sessions[sid]["original_filename"] = fname
        
        composite_bytes = _render_composite_canvas(orig, _sessions[sid]["lines"])
        b64 = base64.b64encode(composite_bytes).decode("utf-8")
        _sessions[sid]["screenshots"].append(b64)
        return jsonify(_session_response(_sessions[sid], b64))

    @bp_or_app.route("/health", methods=["GET"])
    def lc_health():
        return jsonify({"status": "ok", "sessions": len(_sessions)})


def create_blueprint():
    bp = Blueprint("line_crawler", __name__)
    _register_routes(bp)
    return bp

app = Flask(__name__)
_register_routes(app)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5004"))
    log.info(f"Starting LINE-CRAWLER standalone on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
