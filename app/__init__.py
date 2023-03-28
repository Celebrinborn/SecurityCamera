from flask import Flask
import logging

logger = logging.getLogger()
def create_app():
    logger.info('creating app')
    app = Flask(__name__)

    # Register the views Blueprint
    from .views import views
    app.register_blueprint(views)
    
    return app
