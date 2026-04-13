class ModelBuilder:
    def create_image_classifier(self, num_classes, architecture="resnet50", **kwargs):
        return f"Image Classifier ({architecture}) with {num_classes} classes"

    def create_text_classifier(self, num_classes, model_name="bert-base-uncased", **kwargs):
        return f"Text Classifier ({model_name}) with {num_classes} classes"

    def create_time_series_model(self, sequence_length, features, **kwargs):
        return f"Time Series Model"


class PretrainedModels:
    pass
