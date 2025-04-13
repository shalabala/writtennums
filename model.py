import torch
import torch.nn as nn

class Model:
    def __init__(self, filename):
        self._model = Model.__model_def__()
        self._model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        self.parameters = {}

    def infer(self, input_data):
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self._model(input_data)
        return torch.max(output, dim=1).indices.item()
    
    @staticmethod
    def __model_def__():
         model= nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
         model.eval()
         return model
