# ZoneScore

**A living quality scorer that rates geographic zones worldwide on a scale of 0–100 using a Graph Attention Network (GAT).**

## Architecture

| Layer | Technology |
|---|---|
| Backend | Django 4.x + Django REST Framework |
| Frontend | Django templates + Leaflet.js + H3-js |
| Geo processing | GeoPandas, XArray, Shapely |
| GNN | PyTorch Geometric — GATConv |
| Spatial indexing | H3 resolution 7 hexagons |
| Chat assistant | Optional Google Gemini function calling, with a deterministic local mock |

## Quick Start

### 1. Create and activate a virtual environment
```bash
cd zonescorer
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r backend/requirements.txt
```

> **Note:** For PyTorch Geometric, you may need to install wheels matching your CUDA version.
> See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 3. Set up environment variables
Create a `.env` file in the `zonescorer/` project directory and add only the credentials and settings you need.
The chat works without a provider key in mock mode; see [Configure the LLM](#configure-the-llm).

### 4. Train the GAT model
```bash
cd backend
python gnn/train.py
```
This generates `backend/gnn/model.pt` using synthetic data.

### 5. Run the server
```bash
cd backend
python manage.py runserver
```
Open **http://localhost:8000** in your browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health/` | GET | Liveness probe |
| `/api/criteria/` | GET | Returns criteria names and weights |
| `/api/score/` | POST | Scores H3 hexagons for a bounding box |
| `/api/chat/` | POST | Conversational search and ranking of Tunisian zones |

### POST `/api/score/` example
```json
{
  "bbox": [2.28, 48.81, 2.41, 48.90]
}
```
Returns a GeoJSON `FeatureCollection` with `score` and per-criterion values per hexagon.

## LLM Neighborhood Agent

The chat assistant turns a free-text request into a search over Tunisian zones.
The model interprets the request and can ask for missing details, but it does
not calculate or make up zone scores. The backend validates the model's
`search_zones` tool arguments and computes the ranking from the zone data.

### Request flow

1. The chat UI in `frontend/templates/index.html` collects a message. The
   JavaScript sends it as JSON to `POST /api/chat/`, including a `session_id`
   to continue the conversation.
2. `backend/api/chat_views.py` keeps conversation history and the extracted
   preference profile in in-memory session storage. It builds the system
   prompt and exposes the `search_zones` function schema to the LLM client.
3. `backend/api/llm_client.py` selects the local mock or Gemini. The mock
   matches multilingual criterion keywords and location aliases; Gemini can
   use function calling to request a search. Both return either text, a tool
   call, or an error to the chat endpoint.
4. When a tool call is returned, `chat_views.py` validates the criterion IDs,
   importance levels, minimum scores, and result count. It then invokes
   `backend/api/ranking.py`.
5. The ranking engine resolves the location, applies hard minimums, computes
   weighted match scores, sorts the candidates, and returns result details.
   The endpoint sends those actual results back to the LLM for a grounded
   explanation, then returns the reply, zones, and current profile to the UI.
6. The frontend displays the reply and ranked zone cards, updates preference
   chips, and places result markers on the Leaflet map.

Conversation state and rate-limit counters are currently held in process
memory. They are lost when the server restarts and are not shared across
multiple server workers.

### Integration files

| File | Utility in the LLM integration |
|---|---|
| `backend/api/chat_views.py` | Django chat endpoint and agent orchestrator. Defines the `search_zones` tool schema, validates tool arguments, maintains session/profile state, runs the LLM → tool → LLM loop, and returns JSON results. |
| `backend/api/llm_client.py` | Provider adapter. Defines the common response shape, local mock behavior, Gemini API request/function-call parsing, and reads the LLM configuration. |
| `backend/api/system_prompt.py` | Builds the agent instructions and inserts criterion descriptions and multilingual cue words. Sets language, follow-up, tool-use, and no-invented-scores rules. |
| `backend/api/criteria_config.py` | Canonical criterion IDs, names, descriptions, French/Arabic/English synonyms, importance-to-weight mapping, defaults, and the `must_have` score threshold. |
| `backend/api/ranking.py` | Executes `search_zones`: applies location and hard-score filters, calculates weighted compatibility scores, sorts results, and creates explanation facts. |
| `backend/api/location_resolver.py` | Normalizes and fuzzy-matches a user-supplied place against zone, delegation, governorate, and Arabic aliases. |
| `backend/api/zone_data.py` | Provides the `Zone` data model, current zone records and lookup indexes consumed by location resolution and ranking. These records and their criterion scores are synthetic demo data, not live LLM output or GNN inference results. |
| `backend/api/urls.py` | Registers the `/api/chat/` route alongside the existing API endpoints. |
| `backend/zonescorer/urls.py` | Includes the API routes under `/api/` and serves the main frontend template at `/`. |
| `backend/zonescorer/env.py` | Loads configuration from the project `.env` file or environment and parses boolean settings. |
| `backend/zonescorer/settings.py` | Configures Django, REST framework, and the frontend template directory; also uses the shared environment loader for application settings. |
| `backend/requirements.txt` | Declares Django and the project's Python dependencies, including `python-decouple` for configuration. Gemini requests use Python's standard-library HTTP client. |
| `frontend/templates/index.html` | Defines the chat panel, message list, input, preference panel, and map UI. It currently contains the assembled inline chat JavaScript as well as the rest of the map application. |
| `frontend/chat_js_part1.js` | Chat UI helpers for session state, opening/closing the panel, rendering messages, zone cards, and quick replies. |
| `frontend/chat_js_part2.js` | Sends chat requests, updates preference chips, displays zone markers, and wires the send/reset controls. |
| `frontend/inject_chat_js.py` | Assembly helper that inserts the two JavaScript parts into `index.html`. Run it after editing either source part to refresh the inline script in the template. |
| `backend/test_chat.py` | Manual Django client smoke checks for a greeting/follow-up, a French location-and-preference request, and an Arabic request. |
| `backend/test_ar.py` | Small manual diagnostic for Arabic text normalization and location matching. |

The chat ranking data in `backend/api/zone_data.py` is separate from the
geographic H3 scoring API and the trained GNN in `backend/gnn/`. In particular,
the chat agent currently ranks the in-code demo zone records; it does not call
GNN inference to produce those zone-level scores.

### Configure the LLM

The chat works without a provider key in mock mode. The configuration is read
from the `zonescorer/.env` file (the repository's `.gitignore` excludes it) or
from environment variables. Do not commit provider credentials.

```dotenv
ZONESCORER_LLM_MOCK=true
GOOGLE_API_KEY=
ZONESCORER_LLM_MODEL=gemini-1.5-flash
ZONESCORER_LLM_TEMPERATURE=0.3
```

To use Gemini, set `ZONESCORER_LLM_MOCK=false` and provide a valid
`GOOGLE_API_KEY`. The default model is `gemini-1.5-flash`; model and
temperature can be overridden as shown. If mock mode is enabled or the API key
is missing, the client uses the local mock instead.

## Criteria

| # | Criterion | Weight | Source |
|---|---|---|---|
| 1 | Greenness (NDVI) | 20% | Sentinel-2 via GEE |
| 2 | Climate Comfort | 15% | ERA5-Land via CDS |
| 3 | Sky View Factor | 10% | Overture Maps Buildings |
| 4 | Air Quality | 15% | CAMS / Sentinel-5P via GEE |
| 5 | Heat / LST | 15% | Landsat 8/9 via GEE |
| 6 | Accessibility | 15% | OpenStreetMap via osmnx |
| 7 | Transit Access | 10% | Transitland API |

## Credentials (optional — mock mode works without them)

Copy `.env.example` to `.env` and fill in:

```
GEE_SERVICE_ACCOUNT=your-service-account@project.iam.gserviceaccount.com
GEE_KEY_FILE=path/to/key.json
CDS_API_KEY=your-copernicus-key
TRANSITLAND_API_KEY=your-transitland-key
```

## Model Architecture

```
ZoneGAT
  Input:   [N, 7]   (7 criteria per H3 hexagon)
  Layer 1: GATConv(7→32, heads=4, concat=True)  → [N, 128]
  Layer 2: GATConv(128→16, heads=4, concat=True) → [N, 64]
  Layer 3: GATConv(64→1, heads=1, concat=False)  → [N, 1]
  Output:  Sigmoid × 100  →  score ∈ [0, 100]
```

## Project Structure

```
zonescorer/
├── backend/
│   ├── manage.py
│   ├── zonescorer/          # Django project settings
│   ├── api/                 # DRF endpoints and chat-agent modules
│   │   ├── chat_views.py
│   │   ├── llm_client.py
│   │   ├── system_prompt.py
│   │   ├── criteria_config.py
│   │   ├── ranking.py
│   │   ├── location_resolver.py
│   │   └── zone_data.py
│   ├── gnn/                 # GAT model, train, inference
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── model.pt         # generated by train.py
│   ├── preprocessing/       # 7 data criterion modules
│   └── requirements.txt
├── frontend/
│   ├── templates/index.html # App UI, map and assembled chat interface
│   ├── chat_js_part1.js     # Chat message and card rendering
│   ├── chat_js_part2.js     # Chat API, profile and map behavior
│   └── inject_chat_js.py    # Assembles chat JS into the template
├── .env                     # Local configuration; do not commit
└── README.md
```
