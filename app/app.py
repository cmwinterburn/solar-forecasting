from flask import Flask, render_template
import json
import yaml
from config.config import load_config

config = load_config()
forecast_file = config['paths']['forecast_file']

app = Flask(__name__)
    # Accessing file paths

@app.route('/')
def index():
    # Read the JSON file
    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)
    
    return render_template('index.html', forecast=forecast_data)

if __name__ == "__main__":
    app.run(debug=True)
