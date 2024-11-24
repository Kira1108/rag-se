## RAG-SE Tools

We need to build a search and retrieve pipeline for our project.
The following is a simple pipeline search -> retrieve.
```python
from search_engines.bing import BingRequest,  BingTextSearch
from page_reader import SimplePageReader, ConcurrentWrapper
from rewrite import QwenRewriter

q = "question here"
text_req = BingRequest.chinese(q)

rewriter = QwenRewriter()

text_searcher = BingTextSearch(rewriter = rewriter)
search_records = text_searcher.search_normalize(text_req)

page_reader = ConcurrentWrapper(reader = SimplePageReader())
urls = [r.url for r in search_records]
contents = page_reader.read(urls)

for i, s in enumerate(search_records):
    s.set_content(contents[i])
```