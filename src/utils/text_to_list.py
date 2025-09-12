from g2p_en import G2p

def text_to_list(text:  str) -> list[str]:
    p_list = text.split(" ")
    return p_list