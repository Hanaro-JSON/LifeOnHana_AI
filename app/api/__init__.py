from .article_batch_api import article_batch_bp

def init_api(app):
    app.register_blueprint(article_batch_bp) 