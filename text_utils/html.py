import html2text
from functools import lru_cache

@lru_cache(maxsize=None)
def get_default_handler():
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    return h

handles = {
    "default": get_default_handler(),
}

def html_to_text(html:str) -> str:
    return handles["default"].handle(html)
