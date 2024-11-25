import logging

logging.basicConfig(level=logging.INFO)
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import pydantic
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from retry import retry

from text_utils import process_web_content, html_to_text

from .schema import SearchRecord

BING_NEWS_URL:str = "https://api.bing.microsoft.com/v7.0/news/search"
BING_TEXT_URL:str = "https://api.bing.microsoft.com/v7.0/search"

def load_bing_subscription_key():
    load_dotenv(Path.home() / ".env")
    
    
load_bing_subscription_key()

def p2d(p:BaseModel):
    """Serialize pydantic model to dict, excluding None values, note that this may change according to different pydantic versions."""
    
    # 兼容pydantic 1.x 和 2.x
    if pydantic.__version__.startswith("1"):
        return p.dict(exclude_none=True)
    
    return p.model_dump(exclude_none=True)



class BingRequest(BaseModel):
    """
    Request Data Object for Bing V7 API.
    
    You can simply pass query to `1` parameter and get results.
    Or use classmethod `chinese` to get a default request object.   
    
    Request body as descripted in
    https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters.
    
    Note on textFormat:
    Currently only 2 formats are supported, "HTML" and "Raw", HTML is better than Raw, so we choose HTML.
    
    Note on freshness:
    To get articles discovered by Bing during a specific timeframe, specify a date range in the form, YYYY-MM-DD..YYYY-MM-DD. 
    For example, &freshness=2019-02-01..2019-05-30. 
    To limit the results to a single date, set this parameter to a specific date. For example, &freshness=2019-02-04.
    """
    q:str = Field(..., description="The user's search query term.")
    answerCount:Optional[int] = Field(default = None, description="The number of answers that you want the response to include.")
    cc:Optional[str] = Field(default = None, description="A 2-character country code of the country where the results come from. See https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes")
    count:Optional[int] = Field(default = None, description = "The number of search results to return in the response. The default is 10 and the maximum value is 50. The actual number delivered may be less than requested.")
    freshness:Optional[str] = Field(default = None, description = "Filter search results by the following case-insensitive age values ...")
    mkt:Optional[str] = Field(default = None, description="The market where the results come from. Typically, mkt is the country where the user is making the request from.")
    offset:Optional[int] = Field(default = None, description="The zero-based offset that indicates the number of search results to skip before returning results. The default is 0.")
    promote:Optional[str] = Field(default = None, description="A comma-delimited list of answers that you want the response to include regardless of their ranking.")
    responseFilter:Optional[str] = Field(default = None, description="A comma-delimited list of answers to include in the response.")
    safeSearch:Optional[str] = Field(default = None, description="Used to filter webpages, images, and videos for adult content.")
    setLang:Optional[str] = Field(default = None, description="The language to use for user interface strings.")
    textDecorations:Optional[bool] = Field(default = None, description="A Boolean value that determines whether display strings in the results should contain decoration markers such as hit highlighting characters.")
    textFormat:Optional[str] = Field(default = None, description="The type of markers to use for text decorations.")
    
    
    @classmethod
    def chinese(cls, q:str, count:int = 10):
        """this is a default initializing method incase you don't know how to set the parameters."""
        return cls(
            q=q,
            textDecorations=True, 
            textFormat="HTML", 
            cc = "zh-CN", 
            mkt= 'zh-CN', 
            count = count) 
     
def post_process(data: List[dict]) -> List[dict]:
    return [
        {
        "title": html_to_text(d.get('name', "Unnamed")),
        "date": datetime.strptime(d['datePublished'][:19], "%Y-%m-%dT%H:%M:%S") 
            if 'datePublished' in d else datetime(1970,1,1,0,0,0),
        "url": d.get('url', "No URL"),    
        "snippet": process_web_content(d.get("snippet", ""))
        } for d in data]


class BaseBingSearch(ABC):
    
    def __init__(self, rewriter:Callable[[str], str] = None):
        # it is better to used a small model like llama3.1 8B or Qwen2.5 7B to speed up the whole process.
        self.rewriter = rewriter
        
    def rewrite(self, q:str):
        """Rewrite the original query before sending to bing api."""
        if self.rewriter:
            return self.rewriter.rewrite(q)
        return q
    
    def post_process(self, data:List[dict]) -> List[dict]:
        """Post process the raw data from bing api."""
        return post_process(data)
    
    @abstractmethod    
    def search(self, req:BingRequest) -> List[dict]:
        """Search and return a list of raw data."""
        raise NotImplementedError("You must implement this method.")
    
    def search_normalize(self, req:BingRequest) -> List[SearchRecord]:
        """Search and return a list of SearchRecord."""
        data = self.post_process(self.search(req))
        return [SearchRecord(**d) for d in data]
    
class BingTextSearch(BaseBingSearch):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @retry(tries=3, delay=0.1)
    def search(self, req:BingRequest) -> List[dict]:
        logging.info(f"Calling bing text api on query {req.q}")
        headers = {"Ocp-Apim-Subscription-Key": os.getenv("BING_SUBSCRIPTION_KEY")}
        params = p2d(req)
        params['q'] = self.rewrite(req.q)
        print("Using query: ", params['q'])
        response = requests.get(BING_TEXT_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()['webPages']['value']
        return data
    
class BingNewsSearch(BaseBingSearch):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @retry(tries=3, delay=0.1)
    def search(self, req:BingRequest) -> List[dict]:
        logging.info(f"Calling bing news api on query {req.q}")
        headers = {"Ocp-Apim-Subscription-Key": os.getenv("BING_SUBSCRIPTION_KEY")}
        params = p2d(req)
        response = requests.get(BING_NEWS_URL, headers=headers, params=params)
        response.raise_for_status()
        
        try:
            # for news call we only have to return the value field
            data = response.json()['value']
            # change the key name from description to snippet
            for d in data:
                d['snippet'] = d['description']
                del d['description']
                
        except Exception as e:
            raise ValueError(f"Error in response for bing news api: {response.text}")
        return data