-- Create forecast table in the app.db to record forecast data.

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_time TEXT NOT NULL,
    harpnum TEXT NOT NULL,
    flare_class TEXT NOT NULL,
    intensity REAL NOT NULL,
    std_error REAL NOT NULL
);
