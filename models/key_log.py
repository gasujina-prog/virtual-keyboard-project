from datetime import datetime
from core.database import db

class KeyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key_name = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    # user.id는 user 테이블의 id 컬럼을 의미합니다.
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=True)
    user = db.relationship('User', backref=db.backref('keylogs'))