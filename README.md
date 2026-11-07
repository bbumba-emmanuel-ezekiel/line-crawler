# Line Crawler

### A spatial reasoning study

**Bbumba Emmanuel Ezekiel** · Spaxzyte Limited · Kampala, Uganda

---

## Abstract

An embodied agent navigating and tracing two-dimensional plan environments. Spatial understanding is modelled as traversal rather than as inspection from above.

This repository is one of six independent studies pursuing a common question:
what is a functional digital unit for the interpretation of space? Each study
isolates one candidate answer and is published as its own artefact, with its own
documentation, so that it can be read and cited on its own terms.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your own keys
python app.py
```

The interface served is `Interface.html`. Dependencies are declared in
`requirements.txt`; the file is the superset for the collection, so not every
package is required for this study.

## Configuration

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | language-model calls |
| `FAL_KEY` | image generation routes |
| `FLASK_SECRET_KEY` | any random string; Flask session signing |

No credentials are stored in this repository. `.env` is excluded by
`.gitignore`.

## Licence and citation

MIT — see [`LICENSE`](./LICENSE). Citation metadata is in
[`CITATION.cff`](./CITATION.cff).

---

# Detailed documentation

The module documentation below is the study's own working record, reproduced
unchanged.

# LINE-CRAWLER — AI Spatial Navigation & Tracing

**An embodied AI agent that navigates a 2D plan world, interprets its surroundings through raycast vision, and traces features — vertices, polylines, and shapes — onto the environment from a background image.**

> **Research domain:** Embodied AI · Spatial intelligence · Agentic computer vision · Autonomous mapping & tracing

---

## Table of Contents

- [Research Concept](#research-concept)
- [Features](#features)
- [How the AI Works](#how-the-ai-works)
- [Architecture & File Structure](#architecture--file-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Raycast Data Format](#raycast-data-format)
- [API Endpoints](#api-endpoints)
- [Session Management](#session-management)
- [Deployment Notes](#deployment-notes)
- [Research Credits](#research-credits)

---

## Research Concept

Line-Crawler investigates a core question in artificial spatial intelligence: **Can a vision-capable language model function as an embodied spatial agent?**

The system places a simulated "creature" inside a continuous 2D Cartesian world. The creature does not receive an abstract symbolic map of its environment. Instead, it sees the world through:

1. A **rendered scene** (an overhead image of the world, with a translucent blue creature marker, labelled grid, vertices, polylines, and a background image).
2. A **raycast data array** — 36 radial beams (10° increments) that report the distance to the nearest polyline wall, plus special rays that detect visible vertices by ID.

The AI must fuse these two modalities — visual context plus structured numeric rays — into a coherent spatial understanding, then **choose and execute a single action per step** to progressively trace, map, and modify the environment.

This is a minimal yet complete instantiation of the **perception → reasoning → action** loop, making Line-Crawler a testbed for studying how foundation models handle continuous-space navigation, memory of past placements, and long-horizon spatial reasoning.

---

## Features

| Feature | Description |
|---|---|
| **Raycast Vision** | 36 radial beams (10° spacing) from the creature's centre, each reporting distance-to-hit (up to 10 m) and any intersected object. Special rays additionally detect visible vertices by ID. |
| **Vertex Generation** | Creature can shoot a new vertex at a given angle and distance from its current position. |
| **Polyline Tracing** | Connect any sequence of existing vertices into a persistent green polyline. |
| **Shape Generation** | Create 2D shapes (currently a placeholder; supports FAL AI image generation when configured). |
| **Vertex Editing** | Move, displace, or delete existing vertices with sub-metre precision. |
| **Zoom** | Dynamic camera bounds allow the creature to zoom into any region for micro-adjustments. |
| **Background Image Tracing** | Place a satellite photo or floor plan in `Background Image/`; the AI can trace roads, boundaries, or features from it. |
| **AI System Prompt** | A detailed behavioural prompt (`SystemPrompt.txt`) governs the creature's reasoning, action format, and two-step editing protocol. |
| **Session Management** | Each run creates an isolated session directory with state snapshots, full action logs, and rendered step images. |
| **Human-in-the-Loop Interface** | Web UI at `Interface.html` with Start Session, Step AI, and Auto Run controls. |

---

## How the AI Works

The loop is driven entirely by the AI model, accessible through **OpenRouter** (supporting Gemini, GPT, Claude, Qwen, Llama, and others).

### Step-by-step

1. The backend reads the current session `state.json` (creature position, vertices, polylines, shapes, zoom bounds).
2. It renders the world as a 1200×1200 PNG:
   - Background image (scaled from the `Background Image/` folder).
   - All shapes (loaded from session-specific `Shape_*.png` files).
   - All polylines (green, 4 px, curve-joined).
   - All vertices (yellow dots with large red ID labels).
   - The creature (translucent blue circle with centre anchor).
   - A labelled pink grid (1 m spacing, 5 m emphasis).
3. It computes **raycast data**: 36 standard rays plus special vertex-directed rays.
4. It constructs a **multimodal prompt** for the AI:
   - **Text portion:** current creature coordinates, objective string, filtered raycast data, recent history (last N moves).
   - **Image portion:** the rendered scene PNG.
5. The AI returns **exactly one JSON action** (wrapped in ` ```json `) with a `reasoning` field.
6. The backend applies the action to the state, writes new state and history, re-renders, and returns the updated view.

### Action Types

The AI may choose any one of the following per step:

| Action | Purpose |
|---|---|
| `MoveCreature` | Move to an absolute `[x, y]` coordinate |
| `MoveVertexExact` | Relocate a vertex to an exact new position (requires prior MoveCreature to that vertex) |
| `DisplaceVertex` | Micro-adjust a vertex by `dx`, `dy` offset |
| `DeleteVertex` | Remove a vertex (and any polylines referencing it) |
| `GenerateVertex` | Shoot a new vertex at a given `angle_deg` and `length` from current centre |
| `GeneratePolyline` | Connect vertex IDs into a polyline |
| `Zoom` | Set camera bounds `[min_x, min_y, max_x, max_y]` or `null` to reset |
| `GenerateShape` | Create a shape at `[x, y]` with dimensions `w × h` (uses FAL AI if configured) |
| `ModifyShape` | Reposition or rescale an existing shape |

### Two-Step Editing Protocol

The system prompt enforces a strict protocol for vertex adjustments:

1. **Step 1:** `MoveCreature` to the exact coordinates of the target vertex.
2. **Step 2:** `DisplaceVertex` or `MoveVertexExact` to correct its position.

This ensures the creature is always positioned over the vertex it is editing, enabling precise micro-adjustments informed by the zoomed view.

---

## Architecture & File Structure

```
line-crawler/
├── Core.py                 # Flask backend — API server, rendering, raycasting, AI orchestration
├── Interface.html          # Web UI — session controls, canvas display, raycast log
├── Config.json             # World parameters & objective string
├── SystemPrompt.txt        # AI behavioural prompt with action templates
├── models.json             # Available OpenRouter models & default selection
├── image-generation.json   # Shape-generation engine configuration (FAL / OpenRouter)
├── .env                    # API keys (OPENROUTER_API_KEY, FAL_KEY, etc.)
├── LOGIC-BOOK.txt          # Internal design notes (Circles & Line-Crawler)
├── Background Image/       # Place background images (satellite, floor plan, etc.)
│   ├── .gitkeep
│   └── 2.png               # Example background
└── RUNS/                   # Session data (auto-created)
    └── S{n}/               # Each session
        ├── state.json      # Current world state
        ├── History.json    # AI action history (last N moves)
        ├── Logs.txt        # Full developer logs (prompts + raw responses)
        ├── {n}X.png        # Rendered scene image at each step
        └── Shape_{n}.png   # Generated shape images
```

### Key Files at a Glance

| File | Role |
|---|---|
| `Core.py` | Single-file Flask server. Handles session lifecycle, rendering, raycasting, OpenRouter calls, action application, and CORS. |
| `Interface.html` | Single-page web app. Uses vanilla JavaScript with no build step. Communicates with `http://127.0.0.1:5000/api/session/*`. |
| `Config.json` | JSON dictionary defining `objective` (the AI's mission), `grid_size`, `creature_diameter`, and `history_depth`. |
| `SystemPrompt.txt` | The system message sent to the AI on every step. Defines the creature's persona, abilities, action JSON schemas, and behavioural constraints. |
| `models.json` | Lists OpenRouter model slugs and sets the `default` model. The UI uses this when calling the `/api/session/step` endpoint. |

---

## Setup

### 1. Install Dependencies

```bash
pip install flask flask-cors pillow requests python-dotenv
```

Python 3.10+ recommended.

### 2. Configure Environment

Create a `.env` file (or update the existing one) with at least:

```env
OPENROUTER_API_KEY=sk-or-v1-...   # Required for AI inference
FAL_KEY=...                        # Required only for shape generation via FAL AI
```

Optional keys (not required for core functionality):

```env
GOOGLE_GEMINI_KEY=...   # Alternative if calling Gemini directly
ELEVEN_LABS_KEY=...     # Legacy / not used by core loop
HUGGINGFACE_TOKEN=...   # Legacy / not used by core loop
```

### 3. Place a Background Image

Add a PNG, JPG, or WebP image to the `Background Image/` folder. The first image found (alphabetically by extension) is used as the world's background. Satellite images work well for road-tracing tasks.

### 4. (Optional) Select a Model

Edit `models.json` to choose your preferred model. The default is:

```json
{ "default": "google/gemini-3-flash-preview" }
```

Vision-capable models are required — the AI receives the rendered scene as an image input.

### 5. Start the Server

```bash
python Core.py
```

The server starts on `http://127.0.0.1:5000`. Opening this URL in a browser serves `Interface.html`.

---

## Configuration

### `Config.json`

```json
{
  "objective": "Navigate the space and use your abilities to trace out the roads in the background satellite image...",
  "grid_size": 55.0,
  "creature_diameter": 0.6,
  "history_depth": 3
}
```

| Key | Default | Description |
|---|---|---|
| `objective` | — | The mission string injected into every AI prompt. This is the agent's primary task. |
| `grid_size` | `55.0` | The world extent in metres (square). Affects coordinate-space and the default rendering scale. |
| `creature_diameter` | `0.6` | Diameter of the AI creature in metres (approx. 600 mm — roughly human-scale at the waist). |
| `history_depth` | `3` | Number of previous moves included in the AI prompt for temporal context. Higher values increase token usage. |

### `SystemPrompt.txt`

A plain-text file containing the full system message. It defines:

- The creature's identity and scale ("600 mm wide, 1700 mm high").
- How to interpret the rendered image, grid, and raycast data.
- The exact JSON schemas for each of the 9 action types.
- The strict two-step editing protocol.
- The instruction to output exactly one ` ```json ` block per response.

Edit this file to change the AI's behaviour, add new action types, or adjust the editing protocol.

### `models.json`

```json
{
  "default": "google/gemini-3-flash-preview",
  "models": ["google/gemini-3-flash-preview", "openai/gpt-5", ...]
}
```

- `default` — the model slug used for all AI steps.
- `models` — the available list (for future UI model-switching).

---

## Raycast Data Format

The `calculate_raycasts()` function in `Core.py` produces an array of ray objects returned to the frontend and included in the AI prompt context.

### Standard Rays

36 rays equally spaced by 10°, with a maximum range of 10 m.

```json
{
  "angle_deg": 0,
  "distance": 3.42,
  "hit": "PolylineSegment",
  "type": "standard"
}
```

| Field | Description |
|---|---|
| `angle_deg` | Direction of the ray in degrees (0 = right/east, 90 = up/north). |
| `distance` | Distance in metres to the nearest polyline intersection. `10.0` if no hit. |
| `hit` | `"PolylineSegment"` if the ray intersected a polyline; `"None"` otherwise. |
| `type` | Always `"standard"`. |

### Special Vertex Rays

For every vertex within 15 m of the creature, a special ray is computed toward that vertex. If the line-of-sight is unobstructed by polylines, the ray is included.

```json
{
  "angle_deg": 45.0,
  "distance": 2.15,
  "hit": "Vertex_3",
  "target_pos": [18.4, 12.7],
  "type": "special"
}
```

| Field | Description |
|---|---|
| `angle_deg` | Exact angle to the vertex (may not fall on 10° increments). |
| `distance` | Direct Euclidean distance to the vertex in metres. |
| `hit` | `"Vertex_{id}"` if visible; `"Vertex_{id}_Occluded"` if behind a polyline segment (filtered out from prompt). |
| `target_pos` | The vertex's `[x, y]` world coordinates. |
| `type` | Always `"special"`. |

### Filtering for the AI

Before being sent to the AI model, the ray list is filtered to only include rays where `distance < 10.0` (i.e. actual hits) or `type === "special"`. This reduces token usage while preserving the most informative spatial data.

---

## API Endpoints

All endpoints are served by the Flask backend at `http://127.0.0.1:5000`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves `Interface.html` (the web UI). |
| `POST` | `/api/session/start` | Create a new session. Resets the creature to the centre of the grid with no vertices or polylines. Returns `session_id`, initial `state`, and rendered `image`. |
| `GET` | `/api/session/state/<session_id>` | Fetch the current state, rendered image, history summary, and full raycast data for an existing session. |
| `POST` | `/api/session/step` | Advance the AI by one step. Accepts `{ "session_id": "S{id}" }`. Fires the full perception → reasoning → action cycle. Returns the updated state, new image, action taken, history summary, and raycast data. |

### Request/Response Examples

**Start Session:**
```
POST /api/session/start
→ { "session_id": "S1", "state": { ... }, "image": "data:image/png;base64,..." }
```

**Step AI:**
```
POST /api/session/step
{ "session_id": "S1" }
→ {
    "ok": true,
    "action": { "action": { "type": "GenerateVertex", "angle_deg": 45, "length": 5.0 }, "reasoning": "..." },
    "state": { ... },
    "image": "data:image/png;base64,...",
    "rays": [ ... ],
    "history": "..."
  }
```

---

## Session Management

Each call to `POST /api/session/start` creates a new numbered session directory under `RUNS/S{n}/`:

```
RUNS/
├── S1/
│   ├── state.json       # Serializable world state
│   ├── History.json     # Array of { action, note, is_ai, ts }
│   ├── Logs.txt         # Human-readable developer log (prompts + raw responses)
│   ├── 0X.png           # Initial rendered scene
│   ├── 1X.png           # Step 1 rendered scene
│   ├── 2X.png           # Step 2 rendered scene
│   └── ...
├── S2/
└── ...
```

- **Sessions are isolated and persistent.** Restarting the server does not lose existing sessions.
- The **History.json** file stores the last N actions (controlled by `history_depth`) and is injected into the AI prompt each step as temporal context.
- The **Logs.txt** file records every full prompt context and raw AI response — useful for debugging, prompt engineering, and research analysis.

---

## Deployment Notes

### Standalone Server

```bash
python Core.py
```

Binds to `127.0.0.1:5000` by default. Set the `PORT` environment variable to change the port:

```bash
export PORT=8080
python Core.py
```

### As a Blueprint (Embedded Mode)

`Core.py` exposes `create_blueprint()` which returns a Flask Blueprint. This allows Line-Crawler to be mounted under a parent Flask application — for example, a multi-agent research platform:

```python
from flask import Flask
from Core import create_blueprint

app = Flask(__name__)
app.register_blueprint(create_blueprint(), url_prefix='/api/line-crawler')
```

All routes from `Core.py` are preserved; the prefix can be adjusted as needed.

### Production Considerations

- The server runs in **debug=False** mode by default and disables the reloader. For production, consider a WSGI wrapper (Gunicorn, Waitress, or uWSGI).
- Session images and logs accumulate on disk indefinitely. For long-running deployments, add a log-rotation or archival strategy under `RUNS/`.
- The OpenRouter API call has a **120-second timeout**. Long-running model responses may be interrupted; adjust `timeout` on line 635 of `Core.py` if needed.
- API keys in `.env` should be kept out of version control. The `.env` file is listed in `.gitignore` (if configured) — verify it is not tracked.

---

## Research Credits

Line-Crawler was developed as part of the **Consolidated Simple Spatial System Research** project — an ongoing investigation into functional digital units for the interpretation of space.

### Research Foundations

- **Embodied AI & Spatial Reasoning:** The creature-in-grid paradigm draws on ideas from active perception, where an agent must move through its environment to gather information rather than receiving a complete world model a priori.
- **Vision-Language Models as Spatial Agents:** By feeding a rendered scene directly into a vision-capable LLM, the system tests whether these models can perform geometric reasoning — estimating distances, angles, and occlusion relationships from pixel data augmented with structured raycast measurements.
- **Raycast Fusion:** Combining structured numeric data (ray distances) with visual context (the rendered scene) mirrors sensor-fusion approaches in robotics, adapted here for a purely digital agent.
- **Iterative Tracing & Mapping:** The vertex-polyline-shape workflow is a minimal representation of how an autonomous agent might progressively build a structured map of an unknown environment.

### Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/), [Pillow](https://python-pillow.org/), and [OpenRouter](https://openrouter.ai/) for model inference.
- Inspired by research in autonomous navigation, spatial intelligence, and LLM-based agent architectures.
- Part of a broader research programme exploring how AI systems can perceive, reason about, and act upon continuous spatial representations.

---

> **Line-Crawler** — *Tracing the geometry of intelligence, one vertex at a time.*
