
import numpy as np
np.random.seed(15)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import spacy
import stanza


# spacy pour tokeniser, lemmatiser etc.
# spacy_nlp = spacy.load('fr')
# spacy_nlp = spacy.load('fr_core_news_sm')

# stanza.download('fr')
stanza_nlp = stanza.Pipeline('fr', processors='tokenize,mwt,pos,lemma,depparse')



class Classifier:
    """The Classifier: bag-of-words --> représentation creuse """

    def __init__(self):
        self.labelset = None
        self.label_binarizer = LabelBinarizer()
        self.model = None
        self.epochs = 200
        self.batchsize = 16
        self.max_features = 8000
        # create the vectorizer
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            strip_accents=None,
            analyzer="word",
            tokenizer=self.mytokenize,
            stop_words=None,
            ngram_range=(1, 2),
            binary=False,
            preprocessor=None
        )
        self.keep_cats = ['VERB', 'NOUN', 'ADJ', 'ADV']

    def mytokenize(self, text):
        """Customized tokenizer.
        Here you can add other linguistic processing and generate more normalized features
        """
        # doc = spacy_nlp(text)
        doc = stanza_nlp(text)
        # tokens = [t.text.lower() for sent in doc.sents for t in sent if t.pos_ in self.keep_cats ]
        tokens = [t.lemma.lower() for sent in doc.sentences for t in sent.words ]
        return tokens

    def input_size(self):
        return len(self.vectorizer.vocabulary_)


    def vectorize(self, texts):
        """Input text vectorization """
        return self.vectorizer.transform(texts).toarray()


    def create_model(self):
        """Create a neural network model and return it.
        Here you can modify the architecture of the model (network type, number of layers, number of neurones)
        and its parameters"""

        # Define input vector, its size = number of features of the input representation
        input = tf.keras.layers.Input((self.input_size(),))
        # Define output: its size is the number of distinct (class) labels (class probabilities from the softmax)
        layer = input
        layer = tf.keras.layers.Dense(100, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(.7)(layer)
        output = tf.keras.layers.Dense(len(self.labelset), activation='softmax')(layer)
        # create model by defining the input and output layers
        model = tf.keras.Model(inputs=input, outputs=output)
        # compile model (pre
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model



    ####################################################################################################
    # IMPORTANT: ne pas changer le nom et les paramètres des deux méthode suivantes: train et predict
    ###################################################################################################
    def train(self, train_texts, train_labels, dev_texts=None, dev_labels=None):
        """Train the model using the list of text examples together with their true (correct) labels"""
        # create the binary output vectors from the correct labels
        Y_train = self.label_binarizer.fit_transform(train_labels)
        # get the set of labels
        self.labelset = set(self.label_binarizer.classes_)
        print("LABELS: %s" % self.labelset)
        # build the feature index (unigram of words, bi-grams etc.)  using the training data
        self.vectorizer.fit(train_texts)
        print("Vectorizer vocab size:", len(self.vectorizer.vocabulary_))
        # create a model to train
        self.model = self.create_model()
        # for each text example, build its vector representation
        X_train = self.vectorize(train_texts)
        if dev_texts is not None and dev_labels is not None:
            X_dev = self.vectorize(dev_texts)
            Y_dev = self.label_binarizer.transform(dev_labels)
            dev_data = (X_dev, Y_dev)
        else:
            dev_data = None
        #
        my_callbacks = []
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto',
                                       baseline=None)
        my_callbacks.append(early_stopping)
        # Train the model!
        self.model.fit(
            X_train, Y_train,
            epochs=self.epochs,
            batch_size=self.batchsize,
            callbacks=my_callbacks,
            validation_data=dev_data,
            verbose=2)


    def predict(self, texts):
        """Use this classifier model to predict class labels for a batch of input texts.
                Returns the list of predicted labels
        """
        X = self.vectorize(texts)
        # get the predicted output vectors: each vector will contain a probability for each class label
        Y = self.model.predict(X)
        # from the output probability vectors, get the labels that got the best probability scores
        return self.label_binarizer.inverse_transform(Y)



