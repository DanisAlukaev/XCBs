import re
from string import punctuation

import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')


class Preprocess:

    def __init__(self):
        pass

    def remove_numbers(self, text):
        output = ''.join(c for c in text if not c.isdigit())
        return output

    def remove_punct(self, text):
        return ''.join(c for c in text if c not in punctuation)

    def remove_tags(self, text):
        cleaned_text = re.sub('<[^<]+?>', '', text)
        return cleaned_text

    def sentence_tokenize(self, text):
        sent_list = []
        for sent_token in nltk.sent_tokenize(text):
            sent_list.append(sent_token)
        return sent_list

    def word_tokenize(self, text):
        return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]

    def remove_stopwords(self, sentence):
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
                      'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

    def lemmatize(self, text):
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for sent in
                           nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
        return " ".join(lemmatized_word)

    def __call__(self, text):
        lower_text = text.lower()
        sentence_tokens = self.sentence_tokenize(lower_text)
        for sentence_token in sentence_tokens:
            clean_text = self.remove_stopwords(sentence_token)
            clean_text = self.lemmatize(clean_text)
            clean_text = self.remove_numbers(clean_text)
            clean_text = self.remove_punct(clean_text)
            clean_text = self.remove_tags(clean_text)
            word_tokens = list(self.word_tokenize(clean_text))
        return " ".join(word_tokens)


if __name__ == "__main__":
    preprocess_obj = Preprocess()
    text = "1 <TAG> interested and ?"
    print(preprocess_obj(text))
