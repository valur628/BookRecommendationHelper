from flask import Blueprint, render_template, request, flash, g, redirect, url_for
from .model import State, bookrec
import random
from flask import session
import traceback

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
            session['mid_genre'] = mid_genre
            return redirect(url_for('main.select_books'))
    return render_template('index.html', mid_genres=mid_genres)

@bp.route('/books', methods=['GET', 'POST'])
def select_books():
    print("@@@ Calling select_books function @@@")
    mid_genre = session.get('mid_genre', None)
    if mid_genre is None:
        return redirect(url_for('main.select_genre'))
    rec = bookrec(mid_genre)
    state = State(rec.names, rec)
    print(f"@@@ Current state: {state} @@@")
    if state is not None:
        if request.method == 'POST':
            print("@@@ POST method in select_books function @@@")
            selected_books = request.form.getlist("books")
            try:
                print(f"@@@ Selected books: {selected_books} @@@")
                recommendations = state.rec_model.book_recommend(selected_books)
                print(f"@@@ Recommendations: {recommendations} @@@")
            except:
                recommendations = [traceback.format_exc()]
                print("@@@ Error during Recommendation: @@@")
                print(recommendations)
            return render_template('books.html', names=random_books(state), recommendations=recommendations)
        else:
            print("@@@ GET method in select_books function @@@")
            return render_template('books.html', names=random_books(state))
    else:
        print("@@@ State is None in select_books function @@@")
        return redirect(url_for('main.select_genre'))