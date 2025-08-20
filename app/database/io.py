import sqlite3

def insert_forecast(
    db_path: str,
    *,
    forecast_time: str,
    harpnum: str,
    flare_class: str,
    intensity: float,
    std_error: float
): 
    """Write a forecast to the DB from the model output JSON file."""
    
    sql = """
    INSERT INTO forecasts (forecast_time, harpnum, flare_class, intensity, std_error)
    VALUES (?, ?, ?, ?, ?)
    """
    params = (forecast_time, harpnum, flare_class, float(intensity), float(std_error))

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
