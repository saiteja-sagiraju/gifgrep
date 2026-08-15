# gifgrep — System Design & Product Requirements Document

## 1. Product Vision & Constraints

**gifgrep** is a local, highly-optimized semantic GIF search engine. It bridges the "modality gap" by fusing deterministic relational metadata, temporal visual embeddings, and OCR text into a single Reciprocal Rank Fusion (RRF) search pipeline entirely within the database engine.

### Core Architectural Constraints

| Constraint | Detail |
|---|---|
| **Hardware Boundaries** | Must operate safely within an 8GB VRAM limit. GPU workloads (VideoCLIP) must be strictly isolated from CPU workloads (EasyOCR) and human-sourced contextual data. |
| **Dependency-Averse** | Zero distributed over-engineering. No Redis, no Celery, no cloud vector databases, no Jinja2 templating. |
| **Decoupled State** | FastAPI handles asynchronous HTTP routing, Python threadpools handle blocking PyTorch inference, and Turso (libSQL) natively handles all state, indexing, and relevance ranking math. |

---

## 2. User Experience & Inputs

The system relies on human crowdsourcing for deterministic context, trading automated VLM compute for manual user input to protect GPU memory.

### The Upload Flow (Data Ingestion)

When a user uploads a new GIF, the frontend requires three explicit inputs before submission:

1. **The Media Payload** — the raw `.gif` file.
2. **The Entity Mapping** — a relational dropdown selecting the Character and Show/Movie (e.g., "Han", "2 Broke Girls"). This populates the deterministic identity.
3. **The Context Description** — a mandatory text area where the user inputs a single, literal sentence describing the visual action and setting (e.g., "Janice gasping and covering her mouth in shock").

### The Search Flow (Data Retrieval)

- **The Interface**: a single, native text bar. No dropdowns, no advanced filters.
- **The Query**: the user types a natural language query (e.g., "Megatron I smell u boy" or "Janice laughing").
- **UI State**: during uploads, the UI uses a non-blocking `setInterval` JavaScript loop to poll the backend (`GET /api/status/{gif_id}`) and display a loading placeholder until PyTorch processing completes.

---

## 3. Data Schema & Indexing

By leveraging Turso's native Tantivy integration, gifgrep bypasses legacy SQLite FTS5 virtual tables and triggers. Full-text search and vector search exist concurrently on the same core table.

Because FTS operates strictly on row data, the backend concatenates the relational entities, user description, and OCR text into a single denormalized `search_tags` column upon insertion.

```sql
-- 1. Relational Entities (The Knowledge Base)
CREATE TABLE shows (id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE characters (id INTEGER PRIMARY KEY, name TEXT, show_id INTEGER REFERENCES shows(id));

-- 2. Semantic & Storage Table
CREATE TABLE gifs (
    id TEXT PRIMARY KEY,
    usage_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'processing', -- States: processing, ready, failed
    search_tags TEXT,                 -- Denormalized string: "Megatron Transformers I smell u boy robot pointing"
    embedding F32_BLOB(512)           -- VideoCLIP temporal vector
);

-- 3. The Many-to-Many Link
CREATE TABLE gif_characters (
    gif_id TEXT REFERENCES gifs(id),
    character_id INTEGER REFERENCES characters(id),
    PRIMARY KEY (gif_id, character_id)
);

-- 4. Turso Native Indexes
-- Vector Index (optional depending on row count; brute-force kNN is fine for PoC)
CREATE INDEX idx_gifs_vec ON gifs (vector_distance_cos(embedding));

-- Tantivy Full-Text Search Index (no virtual tables needed)
CREATE INDEX idx_gifs_fts ON gifs USING fts (search_tags) WITH (tokenizer='ngram');
```

---

## 4. The Ingestion Pipeline

To prevent FastAPI's asynchronous event loop from freezing, heavy extraction tasks are offloaded.

1. **Request Reception** — the user submits the file, entity IDs, and context description. The server generates a UUID, returns `{"status": "processing", "gif_id": uuid}`, and pushes the payload to FastAPI `BackgroundTasks`.
2. **Threadpool Isolation** — the background task immediately wraps the processing function in `asyncio.to_thread(worker_func)` to move it off the main event loop.
3. **CPU Action (OCR)** — EasyOCR scans the GIF frames, extracts burned-in text, and deduplicates the strings via a Python hash-set.
4. **GPU Action (VideoCLIP)** — PyTorch loads 8–16 frames into the 8GB VRAM buffer, outputting the float32 temporal action array.
5. **Payload Compilation** — the Python worker concatenates the user description, character name, franchise name, and OCR text into one large string.
6. **Commit** — the worker inserts the row into `gifs`, binds the relational IDs in `gif_characters`, and flips `status` to `'ready'`. Turso automatically indexes the text via Tantivy.

---

## 5. The Hybrid Search Pipeline

When the user submits a text query, the system executes Reciprocal Rank Fusion (RRF) entirely inside Turso.

$$RRF = \frac{1}{k + r_{fts}} + \frac{1}{k + r_{vec}}$$

1. **Text Embedding** — the raw string is passed to the VideoCLIP text encoder (wrapped in `@functools.lru_cache` to instantly return frequent queries).
2. **Database Execution** — the Python backend sends the following CTE to Turso, merging the Tantivy BM25 score with cosine vector distance:

```sql
WITH fts_search AS (
    -- Execute Tantivy native search. fts_score ranks relevance.
    SELECT
        id,
        row_number() OVER (ORDER BY fts_score(search_tags, '[QUERY_STRING]') DESC) as fts_rank
    FROM gifs
    WHERE status = 'ready' AND fts_match(search_tags, '[QUERY_STRING]')
    LIMIT 50
),
vec_search AS (
    -- Execute Exact/Approximate Nearest Neighbor search
    SELECT
        id,
        row_number() OVER (ORDER BY vector_distance_cos(embedding, '[<encoded_512_array>]') ASC) as vec_rank
    FROM gifs
    WHERE status = 'ready'
    LIMIT 50
)
SELECT
    COALESCE(f.id, v.id) AS final_gif_id,
    -- RRF math executed in database
    COALESCE(1.0 / (60 + f.fts_rank), 0.0) + COALESCE(1.0 / (60 + v.vec_rank), 0.0) AS rrf_score
FROM fts_search f
FULL OUTER JOIN vec_search v ON f.id = v.id
ORDER BY rrf_score DESC
LIMIT 20;
```

---

## 6. Implementation & QA Breakdown

### Phase 1 — Database & Environment
- Initialize local Turso instance and execute the schema (`CREATE INDEX ... USING fts`).
- Create a setup script to pre-seed the Knowledge Base (`shows` and `characters` tables).

### Phase 2 — Extraction & ML Models
- Implement `VideoCLIP` class (ensure it loads entirely into VRAM).
- Implement EasyOCR CPU sidecar with text deduplication.
- **Stress Test**: batch process 3 GIFs concurrently to ensure strict thread-locking prevents VRAM OOM crashes.

### Phase 3 — The Application API
- Build `POST /api/upload` endpoint using `BackgroundTasks` + `asyncio.to_thread`.
- Build `GET /api/status/{id}` for UI polling.
- Build `GET /api/search` with the embedded LRU cache and dynamic RRF SQL query generation.
- **Stress Test**: trigger a heavy GIF upload and immediately spam `/api/search` to verify the event loop is not starved.

### Phase 4 — Client Interface
- Mount `StaticFiles(directory="frontend", html=True)` as the final FastAPI route.
- Build the frontend layout with a grid layout and the manual upload form.
- Implement the `setInterval` JavaScript polling logic for resolving the placeholder UI.