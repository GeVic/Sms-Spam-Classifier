import csv
from textblob import TextBlob
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

# Data Processing
def split_into_tokens(message):
        message = str(message) + 'latin-1'  # convert bytes into proper unicode
        return TextBlob(message).words

# Splitting into lemmas
def split_into_lemmas(message):
        message = str(message) + 'latin-1'
        words = TextBlob(message).words
        # for each word, take its "base form" = lemma 
        return [word.lemma for word in words]

# main preprocessing function
def main():
    # Reads and processes .csv files
    messages = pandas.read_csv('D:\Program_Files\Research_Project\spam.csv', encoding = "latin-1")
    #test_messages = pandas.read_csv('D:\Program_Files\Research_Project\spam1.csv', encoding = "latin-1")
    #messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
    # print(messages.groupby('label').describe())

    # Data Processing
    ''' def split_into_tokens(message):
        message = str(message) + 'latin-1'  # convert bytes into proper unicode
        return TextBlob(message).words '''
    # check
    # print(messages.message.head().apply(split_into_tokens))

    # Identifying part-of-speech and meaning of word in the sentance (Lemmatisation)
    '''def split_into_lemmas(message):
        message = str(message) + 'latin-1'
        words = TextBlob(message).words
        # for each word, take its "base form" = lemma 
        return [word.lemma for word in words]'''
    # check
    # print(messages.message.head().apply(split_into_lemmas))

    # convertion to vectors (i.e. in ML models understandable language)
    #bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
    # print(len(bow_transformer.vocabulary_))
    # The bag of words counts for the entire corpus
    #messages_bow = bow_transformer.transform(messages['message'])
    # Weighing and Normalization using TF-IDF
    #tfidf_transformer = TfidfTransformer().fit(messages_bow)
    # To transform now to tf-idf at once
    #messages_tfidf = tfidf_transformer.transform(messages_bow)

    # Training the model using Naive Bayes
    #spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
    # check
    # print('predicted:', spam_detector.predict(tfidf_transformer.transform(bow_transformer.transform([messages['message'][3]])))[0])
    # print('expected:', messages.label[3])

    # To see all the prediction on the training data itself
    # all_predictions = spam_detector.predict(messages_tfidf)
    # print(all_predictions)
    # print('accuracy', accuracy_score(messages['label'], all_predictions))
    # To compute precision and recall
    # print(classification_report(messages['label'], all_predictions))

    # Splitting data into test and train (80 - 20)
    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
    #msg_train1, msg_test1, label_train1, label_test1 = train_test_split(test_messages['message'], test_messages['label'], test_size=0.9)
    # print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
    '''
    # Using Naive Bayes classifier 
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])  

    # Tuning Parameters
    params = {
        'tfidf__use_idf': (True, False),
        'bow__analyzer': (split_into_lemmas, split_into_tokens),
    }

    grid = GridSearchCV(
        pipeline,  # pipeline from above
        params,  # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv = KFold(n_splits=2).split(label_train)  # what type of cross validation to use
    )
    # Fits and trains the model
    nb_detector = grid.fit(msg_train, label_train)
    #print(nb_detector.grid_scores_)
    # Check 
    #predictions = nb_detector.predict(msg_test)
    #print(confusion_matrix(label_test, predictions))
    #print("----Naive Bayes Classifier final variables-----\n")
    #print(classification_report(label_test, predictions)) 
    #print("-----------------------------------------\n")'''

    # Using SVM Classifier for better results
    pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
    ])

    # pipeline parameters to automatically explore and tune
    param_svm = [
    {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
    {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
    ]

    grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    #cv = StratifiedKFold(label_train, n_folds = 5)
    cv = KFold(n_splits=5).split(label_train)
    )

    # Fits and trains the model
    svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm
    #print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
    print("\n-----SVM Classifier final variables-----\n")
    print(classification_report(label_test, svm_detector.predict(msg_test)))
    print("-----------------------------------------\n")

''' 
Author: GeVic
Improved Spam Detection
'''
if  __name__ == "__main__":
    main()