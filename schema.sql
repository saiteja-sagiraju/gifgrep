-- 1. Relational Entities (For UI Dropdowns & Filtering)
CREATE TABLE shows (id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE characters (id INTEGER PRIMARY KEY, name TEXT, show_id INTEGER REFERENCES shows(id));

-- 2. Main Storage Table
CREATE TABLE gifs (
    id TEXT PRIMARY KEY,
    usage_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'processing', -- processing, ready, failed
    
    -- Denormalized text block populated by Python worker
    -- Example: "Megatron Transformers I smell u boy Giant robot pointing"
    search_tags TEXT, 
    
    -- VideoCLIP temporal vector
    embedding F32_BLOB(512) 
);

-- 3. The Many-to-Many Link (For strict relational queries if needed)
CREATE TABLE gif_characters (
    gif_id TEXT REFERENCES gifs(id),
    character_id INTEGER REFERENCES characters(id),
    PRIMARY KEY (gif_id, character_id)
);

-- 4. Turso Native Tantivy Index
-- This automatically indexes the search_tags column inside the SQLite B-tree.
-- No triggers required.
CREATE INDEX idx_gif_search ON gifs USING fts (search_tags);