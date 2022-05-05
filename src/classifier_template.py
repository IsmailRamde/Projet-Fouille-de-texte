from typing import List


# classe Classifier : squelette à remplir/compléter
class Classifier():
    """The Classifier: template """

    def __init__(self):
        """ initialize the classifier, in particular the neural model """
        self.model = None
        # rajouter votre code ici

    #############################################

    def train(self, texts: List[str], labels: List[str], dev_texts=None, dev_labels=None):
        """Trains the classifier model on the training set stored in file trainfile"""
        # rajouter votre code ici


    def predict(self, texts: List[str]) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels (same order as the input texts)
        """
        # rajouter votre code ici