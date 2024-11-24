import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import aiohttp

from text_utils import process_web_content


async def get_web_content(url:str, encoding:str = 'utf-8', timeout:int = 10) -> str:
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout = timeout_obj) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            html = await response.text(encoding=encoding)
    return html


class BasePageReader(ABC):
    
    @abstractmethod
    async def get(self, url:str) -> str:
        ...
        
    def transform(self, text:str) -> str:
        return process_web_content(text)
    
    async def aread(self, url:str) -> str:
        logging.info("[WebRetriever] Start retrieving content from " + url)
        start = time.time()
        html = await self.get(url)
        result = self.transform(html)
        during = time.time() - start
        logging.info("[WebRetriever] Finished retrieving content from " + url + f" in {during:.2f} seconds.")
        return result
    
    def read(self, url:str) -> str:
        return asyncio.run(self.aread(url))
    
class SimplePageReader(BasePageReader):
    
    def __init__(self, timeout:int = 10, encoding = 'utf-8'):
        self.timeout = timeout
        self.encoding = encoding
    
    async def get(self, url:str) -> str:
        try:
            return await get_web_content(url, encoding = self.encoding, timeout = self.timeout)
        except:
            logging.info(f"[SimplePageReader] Error retrieving from {url}...")
            return ""
    
@dataclass
class ConcurrentWrapper:
    """A concurrent retriever has a BaseWebRetriever and extracts web content concurrently."""
    
    reader:BasePageReader = None
    
    def __post_init__(self):
        if self.reader is None:
            self.reader = SimplePageReader()
            
    def process(self, results:list) -> list:
        """Post process retrieve results."""
        return results
    
    async def aread(self, urls: list)-> list:
        """Retrieve web contents from multiple urls concurrently."""
        
        logging.info(f"[ConcurrentReader] Retrieving contents from {json.dumps(urls)}")
        start = time.time()
        tasks = [self.reader.aread(url) for url in urls]
        results = await asyncio.gather(*tasks)
        results = self.process(results)
        end = time.time()
        logging.info("[ConcurrentReader] Finished Concurrent web retrieving job in " + f"{end - start:.2f} seconds.")
        return results
    
    def read(self, urls:list) -> list:
        """Synchronously retrieve web contents from multiple urls."""
        return asyncio.run(self.aread(urls))
    
    
