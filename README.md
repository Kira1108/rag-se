## RAG-SE Tools

**本地能跑的小模型RAG**

Execute a Search pipeline, [query_rewrite] -> [search engine] -> [web scraper] -> [content chunking] -> [content reranking]
And finally use an React Agent to answer the question.
```python
import logging
logging.basicConfig(level=logging.ERROR)
import nest_asyncio
nest_asyncio.apply()

import numpy as np

from page_reader import ConcurrentWrapper, SimplePageReader
from rerankers import Reranker
from rewriters import QwenRewriter
from search_engines.bing import BingRequest, BingTextSearch
from splitters import split_search_records
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama


def format_information(search_records):
    knowledge_template = """
    <Search Record>
    Title: {title}
    URL: {url}
    Content: {content}
    </Search Record>
    """.strip()

    return "\n\n".join([knowledge_template.format(
        title = s.title.strip(),
        url = s.url.strip(), 
        content = s.content.strip()) 
    for s in search_records])


def execute_search_engine(q:str, topn:int = 5):
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


def search_tool(query:str, topn:int = 5) -> str:
    """Search microsoft bing against a query, rerank results and return topn results as a knowledge string."""
    search_records = execute_search_engine(query, topn)
    return format_information(search_records)


bing_tool = FunctionTool.from_defaults(
    search_tool
)

agent = ReActAgent.from_tools(
    tools = [bing_tool], 
    llm=Ollama(model="qwen2.5:14b"),
    verbose=True
)
```

```bash
> Running step 1176a73b-c1cc-4f30-b428-c77f2678c3a6. Step input: 新疆有一个叫做五家渠的地方么？
[1;3;38;5;200mThought: The current language of the user is: zh. I need to use a tool to answer the question.
Action: search_tool
Action Input: {'query': '新疆 五家渠', 'topn': 5}
[0mUsing query:  五家渠 新疆 旅游景点
Reranking 71 documents...
Reranking taking 0.74 seconds
[1;3;34mObservation: <Search Record>
    Title: 新疆五家渠的哪些景点值得安利给游客 (新疆五家渠旅游景点)
    URL: https://www.liantu.cn/gonglue/k96669.html
    Content: 联途 > 旅游攻略 > 新疆五家渠的哪些景点值得安利给游客(新疆五家渠旅游景点)
新疆五家渠是位于中国新疆维吾尔自治区的一个城市，虽然不像其他一些新疆城市那样闻名遐迩，但这里依然拥有独特的自然风光和文化遗产。以下是一些值得游客前往的五家渠景点：  
这是一个集观光、休闲、教育和科研于一体的综合性湿地公园。公园内有丰富的水生植物和鸟类资源，是观鸟爱好者的理想之地。在这里，游客可以体验宁静的自然风光，感受大自然的恬静与和谐。
    </Search Record>

<Search Record>
    Title: 2024五家渠旅游攻略,五家渠自由行攻略,马蜂窝五家渠出游 ...
    URL: https://www.mafengwo.cn/travel-scenic-spot/mafengwo/32748.html
    Content: 五家渠市位于新疆维吾尔自治区中部天山山脉北麓、准噶尔盆地东南缘，与昌吉市、乌鲁木齐市相接。
该市是新疆维吾尔自治区天山北坡经济腹心地带，也是从乌鲁木齐到古尔班... 更多>> wangpinga
“太偏了，打车一点都不方便，还有就是周围没有餐馆，酒店餐厅服务很好，因为早上赶飞机，特意吩咐餐厅阿姨帮我提前准备了一份早餐，感动。 永远不满..
“已偕同家人入住五家渠迎宾馆好多次了（自2014年10.1小长假开始）。 本次由于只提前了一天预订，房间为1楼，临近停车场比较吵。 早餐一直觉...” 鹿徒
“还可以，休闲逛逛的人挺多的。 鹿徒 “去看郁金香的时候去过，人还挺多的，烤鱼不错。 具体时间官方还没有发布，不过肯定是在4月中旬到5月上旬之间。
    </Search Record>

<Search Record>
    Title: 新疆五家渠的哪些景点值得安利给游客 (新疆五家渠旅游景点)
    URL: https://www.liantu.cn/gonglue/k96669.html
    Content: 古尔班通古特沙漠 - 这片位于五家渠市境内的沙漠以其独特的自然景观吸引着众多游客。在这里，游客可以体验沙漠探险、骑骆驼穿越沙丘等活动，感受大漠的壮阔与神秘。  
总之，五家渠虽然不像其他新疆城市那样知名，但它独有的自然风光和文化底蕴同样值得游客前来探索。无论是喜欢自然风光的旅行者，还是对历史文化感兴趣的游客，都能在五家渠找到属于自己的旅行乐趣。
    </Search Record>

<Search Record>
    Title: 新疆五家渠的哪些景点值得安利给游客 (新疆五家渠旅游景点)
    URL: https://www.liantu.cn/gonglue/k96669.html
    Content: 五家渠地处古丝绸之路北道，郊区有“唐朝路”、“新渠故城”、“烽火台”等古迹。邻近有天池、北庭故城等名胜景点；有硅化木、风剥岩等地质奇观；垦区有具观光价值的大条田、大喷灌、机械化绿洲大农业。辖区内有平原水库4座，水面30余平方千米，具有兵团军垦特色的青格达湖景区。
    </Search Record>

<Search Record>
    Title: 新疆五家渠的哪些景点值得安利给游客 (新疆五家渠旅游景点)
    URL: https://www.liantu.cn/gonglue/k96669.html
    Content: 五家渠好玩的地方有青格达湖、一万泉旅游度假区、脱库孜萨来古城、三台沟原始森林景区、古尔班通古特沙漠等。
五家渠青格达湖旅游区，以猛进水库为依托，背靠天山。烟波浩淼，碧水连天；映天山雪峰，展大漠风情；花香树绿，宛如江南；天然修饰与人工雕凿相得益彰。三十公里长的和平渠湍湍流水；尤如一条闪光的彩带。9.5公里大坝巍巍雄壮，逶迤弯曲的防浪墙，将旅游区点缀得更加美丽。
    </Search Record>
[0m> Running step 6b88f24b-092c-4c19-ae18-1521378a7b28. Step input: None
[1;3;38;5;200mThought: I can answer without using any more tools. I'll use the user's language to answer.
Answer: 新疆确实有一个叫做五家渠的地方。它位于中国新疆维吾尔自治区中部天山山脉北麓、准噶尔盆地东南缘，是一个集观光、休闲、教育和科研于一体的综合性城市。五家渠拥有丰富的自然风光和文化遗产，如青格达湖风景区、古尔班通古特沙漠等旅游景点都值得一游。

您有其他问题吗？
[0m
```