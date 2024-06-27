
from flask import Flask
def create_app():

    app=Flask(__name__, static_url_path='/website/static')
    app.config['SECRET_KEY']='vigrno'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024

    from.Views import views
    from.plots import hist
    from.ml import ml
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(hist, url_prefix='/')
    app.register_blueprint(ml, url_prefix='/')
    return app








