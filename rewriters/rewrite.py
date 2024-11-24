from llama_index.llms.ollama import Ollama

class BaseQueryRewriter:
    
    def rewrite(self, query: str) -> str:
        """Rewrite the original query before sending to bing api."""
        return query
    
    
    
REWRITE_PROMPT = """
Rewrite the query, make it more suitable for search engine.
Make the intent of the query more clear. add necessary components to the query.
Do not include any explaination, just output the rewritten query.
The output language should be the same as the input language.

Input query:{query}
Rewritten query:
""".strip()
    
class QwenRewriter(BaseQueryRewriter):
    
    def __init__(self, model_name:str = "qwen2.5:14b"):
        self.llm = Ollama(model=model_name)
        
    def rewrite(self, query:str) -> str:
        try:
            p = REWRITE_PROMPT.format(query = query)
            rq = self.llm.complete(p)
            return rq.text.strip()
        except Exception as e:
            print("Error rewriting query:", e)
            return query
        
    