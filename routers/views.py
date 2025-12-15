from flask import Blueprint, render_template

bp = Blueprint('views', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')