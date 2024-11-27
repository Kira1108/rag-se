from llama_index.llms.ollama import Ollama
from datetime import datetime

class BaseQueryRewriter:
    
    def rewrite(self, query: str) -> str:
        """Rewrite the original query before sending to bing api."""
        return query
    
    
REWRITE_PROMPT = """
Rewrite the query, make it more suitable for search engine.
Make the intent of the query more clear. add necessary components to the query.
Do not include any explaination, just output the rewritten query.
The output language should be the same as the input language.
Current datetime is {current_datetime}. Note if the query is time sensitive, you should include a date in the query.
You can refer to today's date to inject the date into the query.

Input query:{query}
Rewritten query:
""".strip()
    
class QwenRewriter(BaseQueryRewriter):
    
    def __init__(self, model_name:str = "qwen2.5:14b"):
        self.llm = Ollama(model=model_name)
        
    def rewrite(self, query:str) -> str:
        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            p = REWRITE_PROMPT.format(
                current_datetime = current_datetime,
                query = query)
            rq = self.llm.complete(p)
            return rq.text.strip()
        except Exception as e:
            print("Error rewriting query:", e)
            return query
        
    