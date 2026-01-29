def syntax(name):
    word_to_search = "what price of"
    
    if name.find(word_to_search) != -1 :
        syntax = "yes"
    else:
        syntax = "no"
    return syntax

ask = "what price of cho"
print(syntax(ask))