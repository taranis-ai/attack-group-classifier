from attack_group_classifier.config import Config
from attack_group_classifier.predictor import Predictor
# import model libraries

class Mpnet(Predictor):

    # add huggingface model name (e.g. facebook/bart-large-cnn)
    # needed for using the modelinfo endpoint
    model_name = None

    def __init__(self):
        # instantiate model here
        self.model = None

    def predict(self):
        # add inference code here
        raise NotImplementedError("The class Mpnet must implement the 'predict' method")
