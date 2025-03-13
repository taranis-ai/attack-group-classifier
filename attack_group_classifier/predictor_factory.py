from attack_group_classifier.config import Config
from attack_group_classifier.predictor import Predictor


class PredictorFactory:
    """
    Factory class that dynamically instantiates and returns the correct Predictor
    based on the configuration. This approach ensures that only the configured model
    is loaded at startup.
    """

    def __new__(cls, *args, **kwargs) -> Predictor:
        if Config.MODEL == 'mpnet':
            from attack_group_classifier.mpnet import Mpnet
            return Mpnet(*args, **kwargs)

        raise ValueError(f"Unsupported model: {Config.MODEL}")

