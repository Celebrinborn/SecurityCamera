from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register the views Blueprint
    from .views import views
    app.register_blueprint(views)
    
    return app
