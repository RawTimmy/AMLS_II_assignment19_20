import pandas as pd
import numpy as np
from tensorflow.keras import preprocessing
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.utils import shuffle

class preprocess:
    def __init__(object):
        print('===4B Dataset Processing===')

    def preprocess_4b(self):

        def build_vocab_list(dataframe):
            vocab_set = set()
            sentenses = []

            text_processor = TextPreProcessor(normalize=['url','email','percent','money','phone','user','time','url','date','number'],
                                              annotate={"hashtag","allcaps","elongated","repeated",'emphasis','censored'},
                                              fix_html=True,
                                              segmenter="twitter",
                                              corrector="twitter",

                                              unpack_hashtags=True,
                                              unpack_contractions=True,
                                              spell_correct_elong=False,

                                              tokenizer=SocialTokenizer(lowercase=True).tokenize,
                                              dicts=[emoticons])

            for index in range(dataframe.shape[0]):
                tweet = dataframe["tweet"][index]
                topic = dataframe["topic"][index].lower()

                if tweet.find(topic) == -1:
                    tweet = topic + ' ' + tweet

                tok = text_processor.pre_process_doc(tweet)
                sentenses.append(" ".join(tok))
                vocab_set.update(tok)

            df_sentenses = pd.DataFrame(sentenses, columns=['content'])
            return vocab_set, df_sentenses

        # Get data
        train_4b = './datasets/4B-English/SemEval2017-task4-dev.subtask-BD.english.INPUT.txt'
        df_4b = pd.read_csv(train_4b, sep='\t', header=None)
        df_4b = df_4b.drop(columns=[0,4]) # Drop id and NaN
        df_4b.columns = ['topic','label','tweet']

        # Replace label (neg:0, pos:1)
        df_4b['label'].replace({"negative":0,
                                "positive":1}, inplace=True)

        # Shuffle the dataset
        df_4b = shuffle(df_4b)
        df_4b.reset_index(inplace=True, drop=True)

        vocab_set, df_sen = build_vocab_list(df_4b)
        df_4b[["content"]]= df_sen[["content"]]
        tokenizer = preprocessing.text.Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(df_4b['content'])

        X = tokenizer.texts_to_sequences(df_4b['content'])
        X = preprocessing.sequence.pad_sequences(X, maxlen=200, padding='post', truncating='post')
        Y = df_4b['label']

        # Train Test Split
        size = X.shape[0]
        train_X_4b, train_Y_4b = X[:np.int(0.8*size)], Y[:np.int(0.8*size)]
        test_X_4b, test_Y_4b = X[np.int(0.8*size):], Y[np.int(0.8*size):]

        return train_X_4b, train_Y_4b, test_X_4b, test_Y_4b

        # vocab_set = build_vocab_list(df_4a)
        # vocab_list = list(vocab_set)
        # file = open('vocab.pickle', 'wb')
        # pickle.dump(vocab_list, file)
        # file.close()

        # file = open('vocab.pickle', 'rb')
        # vocab = pickle.load(file)
        # vocab_len = len(vocab)
        # file.close()
