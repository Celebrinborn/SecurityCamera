from flask import render_template, jsonify
from api import app

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the camera view page
@app.route('/camera')
def camera_view():
    return render_template('camera.html')

# Route for the status API
@app.route('/status')
def status():
    status_dict = {
        'camera_status': app.cm.status(),
        'file_status': app.fm.status()
    }
    return jsonify(status_dict)
