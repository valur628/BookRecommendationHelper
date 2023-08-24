from flask import Flask, g
from .model import bookrec
import os
import pandas as pd
from . import routes

def create_app():
    app = Flask(__name__)

    app.register_blueprint(routes.bp)

    genre_files = os.listdir('data')
    mid_genres = [file.split('-')[1].split('.')[0] for file in genre_files if file.count('-') == 1 and file.endswith('.csv')]

    app.mid_genres = mid_genres

    return app