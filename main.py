import json
from pathlib import Path
from search_engines.bing import (
    BingRequest, 
    BingTextSearch, 
    BingNewsSearch,
    p2d
)

from rewriters import QwenRewriter

from page_reader import SimplePageReader, ConcurrentWrapper

def test_call_bing_raw():
        
    save_path = Path(__file__).parent / "data" / "raw_bing"

    # send a text request
    text_req = BingRequest.chinese("柯南和怪盗基德是什么关系？")
    text_searcher = BingTextSearch()
    news_searcher = BingNewsSearch()
    
    text_data = text_searcher.search(text_req)
    with open(save_path / "bing_text.json", "w") as f:
        json.dump(text_data, f, ensure_ascii=False, indent=4)

    # send a news request
    new_req = BingRequest.chinese("朝鲜和韩国最近的关系如何？")
    news_data = news_searcher.search(new_req)    
    with open(save_path / "bing_news.json", "w") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)
        
def test_readers():
    reader = SimplePageReader()
    creader = ConcurrentWrapper(reader = reader)
    # contents = reader.read("https://www.tsinghua.edu.cn")
    # print(contents)
    
    contents = creader.read(['https://www.tsinghua.edu.cn', 'https://www.pku.edu.cn/about.html'])
    print(contents)
    
    
def test_search_pipeline(q:str = "柯南和基德的关系是？"):
    # 构建请求
    text_req = BingRequest.chinese(q)
    
    # 执行搜索
    rewriter = QwenRewriter()
    text_searcher = BingTextSearch(rewriter = rewriter)
    search_records = text_searcher.search_normalize(text_req)
    
    # 详情获取
    page_reader = ConcurrentWrapper(reader = SimplePageReader())
    urls = [r.url for r in search_records]
    contents = page_reader.read(urls)
    for i, s in enumerate(search_records):
        s.set_content(contents[i])
    return search_records
    


if __name__ == "__main__":
    # test_call_bing_raw()
    # test_readers()
    
    search_records = test_search_pipeline()
    print(search_records)
    
