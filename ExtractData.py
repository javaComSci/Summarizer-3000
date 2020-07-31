import sys
import re
from nltk import tokenize
from nltk import word_tokenize
from collections import Counter


# clean all the data provided
class ExtractData:
    def __init__(self, text):
        self.text = text
        self.vocabulary = set(word_tokenize(self.text.lower()))

    
    # extract all sentences in the text
    def extract_sentences(self):
        self.sentences = tokenize.sent_tokenize(self.text)
        return self.sentences


    # calculate the count of words per sentence
    def calculate_sentence_counts(self):
        # keep track of how many words are in each sentence
        self.sentence_to_word_counts = {}

        for sentence in self.sentences:
            words = word_tokenize(sentence.lower())
            words_count = Counter(words)
            self.sentence_to_word_counts[sentence] = words_count
        

    # calculate the number of documents with term t
    def calculate_document_counts(self):
        # keep track of the number of documents each word is in
        self.docs_for_word = {}

        for word in self.vocabulary:
            for sentence in self.sentences:
                if word in self.sentence_to_word_counts[sentence]:
                    if word in self.docs_for_word:
                        self.docs_for_word[word] += 1
                    else:
                        self.docs_for_word[word] = 1
        
        print(self.docs_for_word)


    # calculate the tfidf matrix
    def calculate_tf_idf(self):
        self.calculate_sentence_counts()
        self.calculate_document_counts()
        # print(self.sentence_counts)