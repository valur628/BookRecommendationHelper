from flask import Blueprint, render_template, request
from book_recommendation import create_app
import random

bp = Blueprint('main', __name__)

def random_books(state, num=10):
    return random.sample(state.names, num)

@bp.route('/', methods=['GET', 'POST'])
def main_page():
    state = create_app().state
    if request.method == 'POST':
        genre = request.form.get("genre")
        selected_books = request.form.getlist("books")
        try:
            recommendations = state.rec_model.book_recommend(genre, selected_books)
        except Exception as e:
            recommendations = [str(e)]
        return render_template('index.html', genre=state.genre, names=random_books(state), recommendations=recommendations)
    return render_template('index.html', genre=state.genre, names=random_books(state))