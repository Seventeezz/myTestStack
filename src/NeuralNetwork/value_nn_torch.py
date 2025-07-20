import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string


class ValueNn(nn.Module):
    def __init__(self, street, pretrained_weights=False, approximate='root_nodes', verbose=1, generate_data = True):
        super(ValueNn, self).__init__()
        print(f"Initializing ValueNn for street {street} | torch version: {torch.__version__}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.approximate = approximate
        street_name = card_to_string.street_to_name(street)
        self.model_dir_path = os.path.join(arguments.model_path, street_name)
        model_name = '{}.{}.pt'.format(arguments.model_filename, self.approximate)
        self.model_path = os.path.join(self.model_dir_path, model_name)
        if generate_data:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError
        self._set_shapes()
        self.build_model()
        self.to(self.device)

        if pretrained_weights and os.path.exists(self.model_path):
            try:
                print(f"Loading model from: {self.model_path}")
                self.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Using randomly initialized weights instead...")
        else:
            print("Model not exist! Using randomly initialized weights...")

        if verbose > 0:
            print(self)

    def _set_shapes(self):
        num_ranks, num_suits, num_cards = constants.rank_count, constants.suit_count, constants.card_count
        num_hands, num_players = constants.hand_count, constants.players_count
        num_output = num_hands * num_players
        num_input = num_output + 1 + num_cards + num_suits + num_ranks
        self.x_shape = num_input
        self.y_shape = num_output

    def build_model(self):
        layers = []
        input_size = self.x_shape
        for num_neurons in arguments.num_neurons:
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.BatchNorm1d(num_neurons))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(0.1))
            input_size = num_neurons
        layers.append(nn.Linear(input_size, self.y_shape))
        self.feed_forward = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        ranges = x[:, :self.y_shape]  # [batch, output_dim]
        mask = (ranges > 0).float()

        ff = self.feed_forward(x)
        values = ff * mask

        estimated_value = torch.sum(values * ranges, dim=1, keepdim=True) / 2
        output = values - estimated_value
        return output

    def predict(self, inputs, out_np):
        """
        inputs: numpy array of shape [b, x_shape]
        out_np: numpy array to write outputs into, shape [b, y_shape]
        """
        # try:
        self.eval()
        with torch.no_grad():
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            outputs = self.forward(inputs_tensor)
            out_np[:] = outputs.cpu().numpy()
        # except Exception as e:
            # print(f"Error during prediction: {str(e)}")
            # out_np.fill(0)

