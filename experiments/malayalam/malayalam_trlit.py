import enchant
import indicnlp.tokenize.indic_tokenize as intk
import pandas as pd
from ai4bharat.transliteration import XlitEngine

us_eng = enchant.Dict("en_US")
uk_eng = enchant.Dict("en_GB")


def is_symbol(word):
    return all(not char.isalnum() for char in word)


def is_not_english_word(word):
    return not (us_eng.check(word) or uk_eng.check(word))


dataset = pd.read_excel('data/Merged.xlsx')
trlit_senteces = []
engine = XlitEngine("ml", beam_width=4, rescore=True)
for sentence in dataset['text'].to_list():
    tokens = intk.trivial_tokenize(sentence, lang='ml')
    transliterated_sentence = sentence
    for word in tokens:
        word = word.strip()
        if word is not None and not word.isnumeric() and not is_symbol(word) and word.isascii() and is_not_english_word(
                word):
            out = engine.translit_word(word, topk=1)
            translit_word = out['ml'][0]
            print(f'english : {word} | translit : {translit_word}')
            transliterated_sentence = str(transliterated_sentence).replace(word, translit_word)
    trlit_senteces.append(transliterated_sentence)
dataset['trlit_text'] = trlit_senteces
dataset.to_excel('data/transliterated.xlsx')
