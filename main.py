from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import turso
from uuid import uuid4
import asyncio
import json
import logging
from extractor.pipeline import GIFExtractionPipeline
from typing import Union, cast

logger = logging.getLogger(__name__)

pipeline = GIFExtractionPipeline()

conn: turso.Connection = cast(turso.Connection, None)
cursor: turso.Cursor = cast(turso.Cursor, None)


async def extract_gif(gif_path_or_bytes: Union[str, bytes], id: str):
    try:
        result = await asyncio.to_thread(pipeline.process_gif, gif_path_or_bytes)
        ocr_text = result.get("ocr_text", "")
        embedding = result.get("embedding")
        embedding_json = json.dumps(embedding) if embedding is not None else None

        if embedding_json:
            cursor.execute(
                "UPDATE gifs SET status = ?, search_tags = ?, embedding = vector32(?) WHERE id = ?",
                ("ready", ocr_text, embedding_json, id),
            )
        else:
            cursor.execute(
                "UPDATE gifs SET status = ?, search_tags = ? WHERE id = ?",
                ("ready", ocr_text, id),
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Error processing GIF {id}: {e}", exc_info=True)
        cursor.execute("UPDATE gifs SET status = ? WHERE id = ?", ("failed", id))
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global conn, cursor

    conn = turso.connect("sqlite.db", experimental_features="index_method")
    cursor = conn.cursor()

    yield

    cursor.close()
    conn.close()


app = FastAPI(lifespan=lifespan)


@app.get("/api")
def home():
    return {"message": "Hello"}


@app.get("/api/shows-and-characters")
def fetch_shows_and_characters():
    cursor.execute("SELECT * FROM shows")
    shows = cursor.fetchall()

    cursor.execute("SELECT * FROM characters")
    characters = cursor.fetchall()

    return {"shows": shows, "characters": characters}


@app.post("/api/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_name = file.filename

    if file_name and not file_name.lower().endswith(".gif"):
        raise HTTPException(status_code=422, detail="Only .gif files are allowed.")

    gif_file = await file.read()
    file_id = str(uuid4())

    cursor.execute("INSERT INTO gifs (id) VALUES (?)", (file_id,))
    conn.commit()

    background_tasks.add_task(extract_gif, gif_file, file_id)

    return {"file_id": file_id}


@app.get("/api/status/{id}")
def get_status(id: str):
    cursor.execute("SELECT status FROM gifs WHERE id = ?", (id,))
    result = cursor.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="GIF not found")

    return {"status": result[0]}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
