import turso

conn = turso.connect("sqlite.db", experimental_features="index_method")
cur = conn.cursor()

shows = [
    (1, "The Simpsons"),
    (2, "Breaking Bad"),
    (3, "Stranger Things"),
    (4, "Game of Thrones"),
    (5, "The Office"),
]

characters = [
    (1, "Homer Simpson", 1),
    (2, "Marge Simpson", 1),
    (3, "Walter White", 2),
    (4, "Jesse Pinkman", 2),
    (5, "Eleven", 3),
    (6, "Mike Wheeler", 3),
    (7, "Jon Snow", 4),
    (8, "Daenerys Targaryen", 4),
    (9, "Michael Scott", 5),
    (10, "Jim Halpert", 5),
]

cur.executescript("""
-- 1. Relational Entities (For UI Dropdowns & Filtering)
CREATE TABLE IF NOT EXISTS shows (id INTEGER PRIMARY KEY, title TEXT);
CREATE TABLE IF NOT EXISTS characters (id INTEGER PRIMARY KEY, name TEXT, show_id INTEGER REFERENCES shows(id));

-- 2. Main Storage Table
CREATE TABLE IF NOT EXISTS gifs (
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
CREATE TABLE IF NOT EXISTS gif_characters (
    gif_id TEXT REFERENCES gifs(id),
    character_id INTEGER REFERENCES characters(id),
    PRIMARY KEY (gif_id, character_id)
);

-- 4. Turso Native Tantivy Index
-- This automatically indexes the search_tags column inside the SQLite B-tree.
-- No triggers required.
CREATE INDEX IF NOT EXISTS idx_gif_search ON gifs USING fts (search_tags);
""")

cur.execute("SELECT COUNT(*) FROM shows")
if cur.fetchone()[0] == 0:
    cur.executemany("INSERT INTO shows (id, title) VALUES (?, ?)", shows)
    cur.executemany("INSERT INTO characters (id, name, show_id) VALUES (?, ?, ?)", characters)

conn.commit()
print(f"Inserted {len(shows)} shows and {len(characters)} characters")