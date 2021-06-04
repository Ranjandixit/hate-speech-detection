import sqlalchemy
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer,String,DateTime,ForeignKey
from sqlalchemy.ext import declarative
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()

class Message(Base):
    __tablename__ ='messages'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    uploader = Column(String,default='admin')
    created_on = Column(DateTime, default=datetime.now)

    def __str__(self):
        return self.text


class Predictor(Base):
    __tablename__ ='prediction'

    id = Column(Integer, primary_key=True)
    output = Column(String)
    msg_id = Column(Integer,ForeignKey(Message.id))
    created_on = Column(DateTime, default=datetime.now)
    msg = relationship("Message",backref='Preditor.msg_id')
    def __str__(self):
        return self.output

if __name__ == "__main__":
    engine = create_engine('sqlite:///db.sqlite3')
    Base.metadata.create_all(engine)