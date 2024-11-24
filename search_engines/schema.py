from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SearchRecord(BaseModel):
    title:str
    date:datetime
    url:str
    snippet:str
    content:Optional[str] = None
    
    def set_content(self, content:str):
        self.content = content
    