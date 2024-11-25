## RAG-SE Tools

Execute a Search pipeline, [query_rewrite] -> [search engine] -> [web scraper] -> [content chunking] -> [content reranking]
```python
import nest_asyncio

nest_asyncio.apply()

import numpy as np

from page_reader import ConcurrentWrapper, SimplePageReader
from rerankers import Reranker
from rewriters import QwenRewriter
from search_engines.bing import BingRequest, BingTextSearch
from splitters import split_search_records


def test_search_pipeline(q:str = "柯南和基德的关系是？", topn:int = 5):
    # 构建请求
    text_req = BingRequest.chinese(q)
    
    # 执行搜索
    rewriter = QwenRewriter()
    text_searcher = BingTextSearch(rewriter = rewriter)
    search_records = text_searcher.search_normalize(text_req)
    
    # 详情爬虫
    page_reader = ConcurrentWrapper(reader = SimplePageReader())
    urls = [r.url for r in search_records]
    contents = page_reader.read(urls)
    for i, s in enumerate(search_records):
        s.set_content(contents[i])
       
    # 内容分割 
    chunked_records = split_search_records(search_records)
    
    # 内容精排
    reranker = Reranker()
    
    docs = [s.content for s in chunked_records]
    scores = reranker.rerank(q, docs)
    top_records = np.array(chunked_records)[np.argsort(scores)[::-1]][:topn].tolist()
        
    return top_records

search_records = test_search_pipeline()
```

```bash
INFO:root:Calling bing text api on query 柯南和基德的关系是？
INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
Using query:  柯南和基德之间的关系是什么？
INFO:root:[ConcurrentReader] Retrieving contents from ["https://www.163.com/dy/article/IK844A8B05530518.html", "https://baijiahao.baidu.com/s?id=1607158420381923663", "https://baijiahao.baidu.com/s?id=1716137486522592196", "https://zhidao.baidu.com/question/426520131.html", "https://zhidao.baidu.com/question/83572472.html", "https://zhidao.baidu.com/question/1890910250807144228.html", "https://baijiahao.baidu.com/s?id=1762875763606340277", "https://www.gamersky.com/handbook/202404/1731994.shtml", "https://k.sina.com.cn/article_6433828208_17f7c6d7000100blqj.html", "https://www.zhihu.com/question/416564528"]
INFO:root:[WebRetriever] Start retrieving content from https://www.163.com/dy/article/IK844A8B05530518.html
INFO:root:[WebRetriever] Start retrieving content from https://baijiahao.baidu.com/s?id=1607158420381923663
INFO:root:[WebRetriever] Start retrieving content from https://baijiahao.baidu.com/s?id=1716137486522592196
INFO:root:[WebRetriever] Start retrieving content from https://zhidao.baidu.com/question/426520131.html
INFO:root:[WebRetriever] Start retrieving content from https://zhidao.baidu.com/question/83572472.html
INFO:root:[WebRetriever] Start retrieving content from https://zhidao.baidu.com/question/1890910250807144228.html
INFO:root:[WebRetriever] Start retrieving content from https://baijiahao.baidu.com/s?id=1762875763606340277
INFO:root:[WebRetriever] Start retrieving content from https://www.gamersky.com/handbook/202404/1731994.shtml
INFO:root:[WebRetriever] Start retrieving content from https://k.sina.com.cn/article_6433828208_17f7c6d7000100blqj.html
INFO:root:[WebRetriever] Start retrieving content from https://www.zhihu.com/question/416564528
INFO:root:[WebRetriever] Finished retrieving content from https://www.gamersky.com/handbook/202404/1731994.shtml in 0.05 seconds.
INFO:root:[WebRetriever] Finished retrieving content from https://k.sina.com.cn/article_6433828208_17f7c6d7000100blqj.html in 0.15 seconds.
INFO:root:[SimplePageReader] Error retrieving from https://www.zhihu.com/question/416564528...
INFO:root:[WebRetriever] Finished retrieving content from https://www.zhihu.com/question/416564528 in 0.16 seconds.
INFO:root:[WebRetriever] Finished retrieving content from https://www.163.com/dy/article/IK844A8B05530518.html in 0.22 seconds.
INFO:root:[WebRetriever] Finished retrieving content from https://baijiahao.baidu.com/s?id=1607158420381923663 in 0.33 seconds.
INFO:root:[WebRetriever] Finished retrieving content from https://baijiahao.baidu.com/s?id=1762875763606340277 in 0.33 seconds.
INFO:root:[WebRetriever] Finished retrieving content from https://baijiahao.baidu.com/s?id=1716137486522592196 in 0.36 seconds.
INFO:root:[SimplePageReader] Error retrieving from https://zhidao.baidu.com/question/1890910250807144228.html...
INFO:root:[WebRetriever] Finished retrieving content from https://zhidao.baidu.com/question/1890910250807144228.html in 0.72 seconds.
INFO:root:[SimplePageReader] Error retrieving from https://zhidao.baidu.com/question/426520131.html...
INFO:root:[WebRetriever] Finished retrieving content from https://zhidao.baidu.com/question/426520131.html in 0.72 seconds.
INFO:root:[SimplePageReader] Error retrieving from https://zhidao.baidu.com/question/83572472.html...
INFO:root:[WebRetriever] Finished retrieving content from https://zhidao.baidu.com/question/83572472.html in 2.34 seconds.
INFO:root:[ConcurrentReader] Finished Concurrent web retrieving job in 2.35 seconds.
INFO:root:Loading huggingface model BAAI/bge-reranker-base
INFO:root:Model BAAI/bge-reranker-base loaded.
Reranking 21 documents...
Reranking taking 0.28 seconds
```