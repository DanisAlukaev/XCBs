import re
from collections import Counter
from string import punctuation

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from tqdm import tqdm

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


class Vocabulary:

    def __init__(
        self,
        annotation_path="data/captions_merged.csv",
        mix_with_mscoco=True,
    ):
        self.annotation_path = annotation_path
        self.mix_with_mscoco = mix_with_mscoco
        self.tokenizer = get_tokenizer('spacy', language='en')
        self.read_annotations_file()
        self.build_vocab()

    def build_vocab(self):
        word_counter = Counter()
        for _, row in tqdm(self.annotations.iterrows(), total=self.annotations.shape[0]):
            mask_source_captions = [str(label)
                                    for label in eval(row.mask_source_captions)]
            source_captions = eval(row.source_captions)
            source_captions = [str(caption) for caption in source_captions]

            captions_cub = [caption for caption, label in zip(
                source_captions, mask_source_captions) if label == 'cub']
            mask_cub = ['cub'] * len(captions_cub)
            captions_mscoco = [caption for caption, label in zip(
                source_captions, mask_source_captions) if label == 'coco']
            mask_mscoco = ['coco'] * len(captions_mscoco)
            if self.mix_with_mscoco:
                source_captions = captions_mscoco + captions_cub
                mask_source_captions = mask_mscoco + mask_cub
            else:
                source_captions = captions_cub
                mask_source_captions = mask_cub
            text = " ".join(source_captions)
            tokens = self.tokenizer(text)
            word_counter.update(tokens)

        special_symbols = ["<pad>", "<unk>"]
        self.vocab = vocab(word_counter, specials=special_symbols)
        self.vocab.set_default_index(self.vocab["<unk>"])

    def read_annotations_file(self):
        filename = self.annotation_path
        self.annotations = pd.read_csv(
            filename, names=['filename', 'source_captions', 'mask_source_captions', 'attributes'])
        self.annotations
        print(self.annotations.head())


if __name__ == "__main__":
    preprocess_obj = Preprocess()
    text = "1 <TAG> interested and ?"
    print(preprocess_obj(text))
