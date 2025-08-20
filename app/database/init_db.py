import sqlite3

def init_db(DB_PATH, SCHEMA_PATH):
    """Initialise an SQLite database schema to record solar forecasts."""
    
    with sqlite3.connect(DB_PATH) as conn, open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    print(f"Initialized SQLite at {DB_PATH}")

