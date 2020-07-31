import sys
import re
import math
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
        
        # print(self.docs_for_word)


    # calculate the tfidf matrix
    def calculate_tf_idf(self):
        # get the counts for each sentence and document
        self.calculate_sentence_counts()
        self.calculate_document_counts()
        
        num_docs = len(self.sentences)

        weight_per_sentence = {}

        # calculate the tfidf 
        for sentence in self.sentences:
            weight = 0
            words = word_tokenize(sentence.lower())
            total_words_in_sentence = sum(self.sentence_to_word_counts[sentence].values())
            for word in words:
                tf = self.sentence_to_word_counts[sentence].get(word, 0) / total_words_in_sentence
                idf = math.log(num_docs/(self.docs_for_word[word]))
                tf_idf = tf*idf
                weight += tf_idf
            weight_per_sentence[sentence] = weight

        weight_per_sentence = {k: v for k, v in sorted(weight_per_sentence.items(), key=lambda item: item[1], reverse=True)}

        self.weight_per_sentence = weight_per_sentence
    

    # get the top most popular sentences
    def get_important_sentences(self, k):
        important_sentences = []

        i = 0
        for key, value in self.weight_per_sentence.items():
            if i < k:
                important_sentences.append(key)
                i += 1
            else:
                break
        
        return important_sentences