# gifgrep — TODO

Tracking checklist derived from the Implementation & QA Breakdown in the design doc.

## Phase 1 — Database & Environment
- [X] ~~Initialize local Turso instance~~
- [X] ~~Execute schema: `shows`, `characters`, `gifs`, `gif_characters` tables~~
- [X] ~~Create `idx_gifs_vec` vector index~~
- [X] ~~Create `idx_gifs_fts` Tantivy FTS index (`ngram` tokenizer)~~
- [X] ~~Write setup script to pre-seed Knowledge Base (`shows` + `characters`)~~

## Phase 2 — Extraction & ML Models
- [X] ~~Implement `VideoCLIP` class (confirm full VRAM load)~~
- [X] ~~Implement EasyOCR CPU sidecar~~
- [X] ~~Add OCR text deduplication (Python hash-set)~~
- [X] ~~**Stress test**: batch process 3 GIFs concurrently, confirm thread-locking prevents VRAM OOM~~

## Phase 3 — The Application API
- [ ] Build `POST /api/upload` (`BackgroundTasks` + `asyncio.to_thread`)
- [ ] Build `GET /api/status/{id}` for UI polling
- [ ] Build `GET /api/search` (RRF SQL generation + `functools.lru_cache` for query embeddings)
- [ ] **Stress test**: trigger heavy upload, spam `/api/search` concurrently, confirm event loop isn't starved

## Phase 4 — Client Interface
- [ ] Mount `StaticFiles(directory="frontend", html=True)` as final FastAPI route
- [ ] Build frontend grid layout
- [ ] Build manual upload form (file + character/show dropdown + description text area)
- [ ] Implement `setInterval` polling logic for upload placeholder UI

## Open Questions / Follow-ups
- [ ] Decide when to switch from brute-force kNN to an approximate vector index (row-count threshold)
- [ ] Define `k` value for RRF (doc uses 60 — confirm/tune)
- [ ] Define failure/retry behavior for `status = 'failed'` GIFs