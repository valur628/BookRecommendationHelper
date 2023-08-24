from flask import Blueprint, render_template, request, flash, g
from .model import State, bookrec
import random

bp = Blueprint('main', __name__)

def random_books(state, num=10):
    return random.sample(state.names, num)

@bp.route('/', methods=['GET', 'POST'])
def select_genre():
    if 'mid_genres' not in g:
        from . import create_app
        g.mid_genres = create_app().mid_genres
    mid_genres = g.mid_genres

    if request.method == 'POST':
        mid_genre = request.form.get("mid_genre")
        if mid_genre:
            new_rec = bookrec(mid_genre)
            g.state = State(new_rec.genre, new_rec.names, new_rec)
            return render_template('books.html', genre=g.state.genre, names=random_books(g.state))
    return render_template('index.html', mid_genres=mid_genres)

@bp.route('/books', methods=['GET', 'POST'])
def select_books():
    state = g.get('state')
    if request.method == 'POST':
        genre = request.form.get("genre")
        selected_books = request.form.getlist("books")
        if genre and selected_books:
            recommendations = state.rec_model.book_recommend(genre, selected_books)
        else:
            recommendations = []
            flash('하위 장르와 책 선택은 필수입니다.')
    else:
        recommendations = []
    return render_template('books.html', genre=state.genre, names=random_books(state), recommendations=recommendations)