import numpy as np
import re
import random
import os
import string
import math


class SpamCLF(object):
    def __init__(self):
        self.DATA_DIR = 'enron'
        self.X, self.y = self.get_data()

    def clean(self, s):
            translator = str.maketrans("", "", string.punctuation)
            return s.translate(translator)

    def textParse(self, bigString):  
        listOfTokens = re.split(r'\W+', self.clean(bigString))  
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]  



    def get_data(self):
        subfolders = ['enron%d' % i for i in range(1, 7)]
        data = []
        target = []
        for subfolder in subfolders:
            spam_files = os.listdir(os.path.join(self.DATA_DIR, subfolder, 'spam'))
            for spam_file in spam_files:
                with open(os.path.join(self.DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
                    data.append(self.textParse(f.read()))
                    target.append(1)
            ham_files = os.listdir(os.path.join(self.DATA_DIR, subfolder, 'ham'))
            for ham_file in ham_files:
                with open(os.path.join(self.DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
                    data.append(self.textParse(f.read()))
                    target.append(0)
        tmp = np.arange(len(target))
        np.random.shuffle(tmp)
        data1 = []
        target1 = []
        for t in tmp:
            data1.append(data[t])
            target1.append(target[t])
        return data1, target1

    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def fit(self, test_num):
        X, Y = self.X[:-test_num], self.y[:-test_num]
        self.num_messages = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()
        self.num_messages['spam'] = sum(1 for label in Y if label == 1)
        self.num_messages['ham'] = sum(1 for label in Y if label == 0)

        self.log_class_priors['spam'] = math.log(
            self.num_messages['spam'] / (self.num_messages['spam'] + self.num_messages['ham']))
        self.log_class_priors['ham'] = math.log(
            self.num_messages['ham'] / (self.num_messages['spam'] + self.num_messages['ham']))

        self.word_counts['spam'] = {}
        self.word_counts['ham'] = {}

        for x, y in zip(X, Y):
            c = 'spam' if y == 1 else 'ham'
            counts = self.get_word_counts(x)
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0
                self.word_counts[c][word] += count

    def predict(self, X):
        result = []
        for x in X:
            counts = self.get_word_counts(x)  
            spam_score = 0
            ham_score = 0
        
            for word, _ in counts.items():

                log_w_given_spam = math.log(
                    (self.word_counts['spam'].get(word, 0) + 1) / (
                            sum(self.word_counts['spam'].values()) + len(self.vocab)))
                log_w_given_ham = math.log(
                    (self.word_counts['ham'].get(word, 0) + 1) / (sum(self.word_counts['ham'].values()) + len(
                        self.vocab)))

                
                spam_score += log_w_given_spam
                ham_score += log_w_given_ham

           

           
            spam_score += self.log_class_priors['spam']
            ham_score += self.log_class_priors['ham']

            
            if spam_score > ham_score:
                result.append(1)
            else:
                result.append(0)


        return result


    def acc(self, test_num):
        accuracy = 0
        pred = self.predict(self.X[-test_num:])
        true = self.y[-test_num:]
        for i in range(len(true)):
            if pred[i] == true[i]:
                accuracy += 1
    
        return accuracy/len(pred)



if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    spam_clf = SpamCLF()
    test_num = 500
    spam_clf.fit(test_num)
    print(spam_clf.acc(test_num))
