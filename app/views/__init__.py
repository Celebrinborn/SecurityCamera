from flask import Blueprint

# create a Blueprint for the views
views = Blueprint('views', __name__)

# import the view functions
from .index import index
from .camera import *
from .stats import stats

# register the view functions with the Blueprint
views.add_url_rule('/', view_func=index)
views.add_url_rule('/camera', view_func=camera)
views.add_url_rule('/stats', view_func=stats)
views.add_url_rule('/video_feed', view_func=video_feed)
