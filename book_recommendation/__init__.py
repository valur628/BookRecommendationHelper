from flask import Flask
from .model import bookrec, State

def create_app():
    app = Flask(__name__)

    new_rec = bookrec()
    app.state = State(new_rec.df['하위 장르'].unique().tolist(), new_rec.names, new_rec.df['중위 장르'].unique().tolist(), new_rec)

    from book_recommendation import routes
    app.register_blueprint(routes.bp)

    return app