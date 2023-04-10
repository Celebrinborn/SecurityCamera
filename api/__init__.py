# from flask import Flask
# import logging

# logger = logging.getLogger()
# def create_app():
#     logger.info('creating app')
#     app = Flask(__name__)

#     # Register the views Blueprint
#     from .views import views
#     app.register_blueprint(views)
    
#     return app


from flask import Flask
from camera.camera import Camera
from camera.filemanager import FileManager

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

with Camera() as cm, FileManager() as fm:
    app.cm = cm
    app.fm = fm

    from api.routes import *

    # Register blueprints here
    from api.camera_blueprint import camera_bp
    app.register_blueprint(camera_bp)

    if __name__ == '__main__':
        app.run()