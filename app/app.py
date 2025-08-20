from flask import Flask, render_template, request
import os
import sqlite3
from app.config.config import load_config
from app.database.init_db import init_db

# Load filepaths required to render forecast from config.
config = load_config()
forecast_file = config['paths']['forecast_file']
database_file = config['paths']['database_file']
database_schema = config['paths']['database_schema']

# Create database if it does not already exist.
if not os.path.exists(database_file):
    init_db(database_file, database_schema)

# Create named app instance and route homepage.
app = Flask(__name__)
@app.route("/")


def index():
    """Render the forecast table filtered by date (if provided)"""
    
    # Use GET request to get date from form.
    date_filter = request.args.get('date')

    # create DB connection and access rows by name.
    conn = sqlite3.connect(database_file)
    conn.row_factory = sqlite3.Row

    # Fetch data using filter.
    if date_filter:
        rows = conn.execute(
            "SELECT * FROM forecasts WHERE DATE(forecast_time) = ? ORDER BY forecast_time DESC",
            (date_filter,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM forecasts ORDER BY forecast_time DESC LIMIT 50"
        ).fetchall()

    # Close connection and render GUI from template.
    conn.close()
    return render_template('index.html', forecasts=rows, selected_date=date_filter)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
