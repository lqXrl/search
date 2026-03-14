import os
import json

DEFAULT_MODELS = {
    'inside_outside': {'type': 'classification', 'num_classes': 2},
    'station_objects': {'type': 'detection', 'num_classes': 10},
    'faces': {'type': 'detection', 'num_classes': 2},
    'ground_surface': {'type': 'classification', 'num_classes': 3},
    'space_objects': {'type': 'classification', 'num_classes': 4},
}


class ModelManager:
    def __init__(self, root_dir='models'):
        self.root = root_dir
        os.makedirs(self.root, exist_ok=True)
        self.models = DEFAULT_MODELS

    def list_models(self):
        return list(self.models.keys())

    def get_model_config(self, name):
        return self.models.get(name)

    def model_dir(self, name):
        return os.path.join(self.root, name)

    def ensure_model_dir(self, name):
        d = self.model_dir(name)
        os.makedirs(d, exist_ok=True)
        return d

    def create_model(self, name, config):
        self.models[name] = config
        d = self.model_dir(name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return d
