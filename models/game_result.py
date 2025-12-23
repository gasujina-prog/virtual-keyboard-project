from datetime import datetime
from core.database import db

class GameResult(db.Model):
    __tablename__ = 'game_result'

    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Integer, nullable=False)        # 게임 점수 (예: 1500)
    accuracy = db.Column(db.Float, nullable=False)       # 정확도 (예: 95.5)
    mode = db.Column(db.String(20), nullable=False)      # 게임 모드 (Easy, Normal, Hard)
    create_date = db.Column(db.DateTime(), nullable=False, default=datetime.now) # 게임 한 날짜

    # 어떤 유저의 기록인지 연결 (Foreign Key)
    # user 테이블의 id와 연결됩니다. 유저가 삭제되면 기록도 같이 삭제(CASCADE)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    user = db.relationship('User', backref=db.backref('game_results', cascade='all, delete-orphan'))