from flask import Flask, render_template
import json
import os
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

app = Flask(__name__)
@app.route('/')

def index():
    """Render the current forecast.json file using index.html template"""

    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)
    
    return render_template('index.html', forecast=forecast_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
