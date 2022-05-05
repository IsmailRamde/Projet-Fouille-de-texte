import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from transformers import CamembertTokenizer, TFCamembertModel, CamembertConfig

model_name = "jplu/tf-camembert-base"
attention_mask_name = 'attention_mask'
input_ids_name = 'input_ids'
metric_name = "accuracy"

MAX_SEQ_LENGTH = 143
EPOCHS = 10
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 3


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.classifier = None
        self.model = None
        self.tokenizer = None
        self.label_binarizer = preprocessing.LabelBinarizer()
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            verbose=1,
            # first tests indicates 3 epoch is enough to have a good result
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True)
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)

    def train(self, train_texts, train_labels, dev_texts=None, dev_labels=None):
        """Trains the classifier model on the training set stored in file trainfile"""

        self.label_binarizer.fit(train_labels)
        print("CLASSES: %s" % self.label_binarizer.classes_)
        print("MAX_SEQ_LENGTH: %d" % MAX_SEQ_LENGTH)
        print("EPOCHS: %d" % EPOCHS)
        print("BATCH_SIZE: %d" % BATCH_SIZE)
        print("EARLY_STOPPING_PATIENCE: %d" % EARLY_STOPPING_PATIENCE)

        self.create_model()

        text_train_ids = self.mytokenize(train_texts)
        dev_train_ids = self.mytokenize(dev_texts)

        train_labels = self.label_binarizer.transform(train_labels)
        dev_labels = self.label_binarizer.transform(dev_labels)

        history = self.model.fit(
            [
                text_train_ids[input_ids_name],
                text_train_ids[attention_mask_name]
            ],
            train_labels,
            callbacks=[self.early_stopping],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([
                                 dev_train_ids[input_ids_name],
                                 dev_train_ids[attention_mask_name]
                             ],
                             dev_labels)
        )

        Classifier.save_acc_plot(history)

    def predict(self, texts):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels (same order as the input texts)
        """

        texts_ids = self.mytokenize(texts)
        review_probs = self.model.predict([texts_ids[input_ids_name], texts_ids[attention_mask_name]])
        return self.label_binarizer.inverse_transform(review_probs)

    def create_model(self):
        label_count = len(self.label_binarizer.classes_)
        config = CamembertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        pretrained_model = TFCamembertModel.from_pretrained(model_name,
                                                            config=config)
        # Model definition
        # input
        input_ids = tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name=input_ids_name)
        input_attention_mask = tf.keras.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name=attention_mask_name)
        # contextual embeddings
        embeddings = pretrained_model([input_ids, input_attention_mask])[0]
        # output
        output = tf.keras.layers.Dense(label_count, activation='softmax', name='softmax')(embeddings[:, 0, :])
        # define model with input and output
        model = tf.keras.Model(inputs=[input_ids, input_attention_mask], outputs=[output])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[metric_name])
        model.summary()
        self.model = model

    def mytokenize(self, train_texts):
        ids = self.tokenizer(train_texts,
                             return_attention_mask=True,
                             add_special_tokens=True,
                             padding='longest',
                             max_length=MAX_SEQ_LENGTH,
                             truncation=True,
                             return_tensors='tf',
                             return_token_type_ids=False)
        return ids

    @staticmethod
    def save_acc_plot(history):
        plt.plot(history.history[metric_name],color="blue")
        plt.plot(history.history['val_' + metric_name],color="red")
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend([metric_name, 'val_' + metric_name])