import re

def remove_non_chinese(input_string, min_len = 20):
    
    """
    Remove non-Chinese characters from the input string if they appear in sequences longer than a specified minimum length.

    Args:
        input_string (str): The string from which to remove non-Chinese characters.
        min_len (int, optional): The minimum length of consecutive non-Chinese characters to be removed. Defaults to 20.

    Returns:
        str: The modified string with long sequences of non-Chinese characters removed.
    
    Example:
    s = "today is a very happy day ,我觉得abc很开心， jkls;djfkl;ajskl;jfkl;aefefefe"
    remove_non_chinese(s)
    # Output
    # '我觉得abc很开心'
    """
    
    pattern = rf'[^\u4e00-\u9fff]{{{min_len + 1},}}'
    
    return re.sub(pattern, "", input_string)


def remove_shot_lines(input_string:str, min_len:int = 35):
    
    """
    Desc:
        If a line is shorter than the specified length, remove it from the input string.
    Parameters:
        input_string (str): The input string to process.
        min_len (int): The minimum length of a line to keep.
    Returns:
        str: The processed string.
    """
    
    return "\n".join([
            t for t in input_string.split("\n") if len(t.strip()) > min_len
        ])
    


