from sqlalchemy import Column, Integer, String, LargeBinary, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True, index=True)
    base_url = Column(String, unique=True, index=True)
    file_data = Column(LargeBinary)
    file_name = Column(String)

    pages = relationship("Page", back_populates="site")

class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"))
    url = Column(String)
    text_line = Column(Text)

    site = relationship("Site", back_populates="pages")
