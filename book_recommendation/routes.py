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
    print("@@@ select_books 함수 호출 중... @@@")
    mid_genre = session.get('mid_genre', None)
    if mid_genre is None:
        return redirect(url_for('main.select_genre'))
    rec = bookrec(mid_genre)
    state = State(rec.names, rec)
    print(f"@@@ 현 상태: {state} @@@")
    if state is not None:
        if request.method == 'POST':
            print("@@@ select_books 함수의 POST 메서드 시작 @@@")
            selected_books = request.form.getlist("books")
            try:
                print(f"@@@ 선택된 책: {selected_books} @@@")
                recommendations, scores = state.rec_model.book_recommend(selected_books)
                print(f"@@@ 책 추천: {recommendations}, 점수: {scores} @@@")
            except:
                recommendations = [traceback.format_exc()]
                print("@@@ 추천 중 오류: @@@")
                print(recommendations)
            return render_template('books.html', names=random_books(state), recommendations=recommendations)
        else:
            print("@@@ select_books 함수의 GET 메서드 시작 @@@")
            return render_template('books.html', names=random_books(state))
    else:
        print("@@@ select_books 함수의 상태는 None입니다. @@@")
        return redirect(url_for('main.select_genre'))