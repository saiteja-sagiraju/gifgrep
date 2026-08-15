from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import turso

@asynccontextmanager
async def lifespan(app: FastAPI):
    global conn, cursor

    conn = turso.connect("sqlite.db", experimental_features="index_method")
    cursor = conn.cursor()

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/api")
def home():
    return {"message":"Hello"}

@app.get("/api/shows-and-characters")
def fetch_shows_and_characters():
    cursor.execute("SELECT * FROM shows")
    shows = cursor.fetchall()

    cursor.execute("SELECT * FROM CHARACTERS")
    characters = cursor.fetchall()

    return {"shows":shows, "characters":characters}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
