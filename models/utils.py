from gensim import corpora


"""
Helper function to lower string.
"""
def lower_item(item):
    return item.lower()
"""
Lower text1 and text2
"""
def lower_all_text(df):
    df.question1 = df.question1.map(lower_item)
    df.question2 = df.question2.map(lower_item)


"""
Helper to get all the text. All the preprocessing should be done here. Including nltk stemming/lemmatization.
"""
def get_text_array(df):
    return [ x.split() for x in df.question1.tolist()+df.question2.tolist()]




