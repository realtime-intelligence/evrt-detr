import torch
from torch import nn

LABEL_HIDDEN = 'hidden'
LABEL_CELL   = 'cell'

class Gate(nn.Module):

    def __init__(
        self, features_input, features_hidden, kernel_size = 3,
    ):
        super().__init__()

        self.gate_to_input = nn.Conv2d(
            features_input, features_hidden, kernel_size, padding = 'same',
            bias = True
        )
        self.gate_to_hidden = nn.Conv2d(
            features_hidden, features_hidden, kernel_size, padding = 'same',
            bias = False
        )
        self.gate_to_cell = nn.Parameter(
            torch.zeros((1, features_hidden, 1, 1))
        )

    def forward(self, x, hidden_state, cell_state):
        result  = self.gate_to_input(x)
        result += self.gate_to_hidden(hidden_state)
        result += self.gate_to_cell * cell_state

        return torch.sigmoid(result)

class CellStateGate(nn.Module):

    def __init__(
        self, features_input, features_hidden, kernel_size = 3,
    ):
        super().__init__()

        self.gate_to_input = nn.Conv2d(
            features_input, features_hidden, kernel_size, padding = 'same',
            bias = True
        )
        self.gate_to_hidden = nn.Conv2d(
            features_hidden, features_hidden, kernel_size, padding = 'same',
            bias = False
        )

    def forward(self, x, hidden_state):
        result  = self.gate_to_input(x)
        result += self.gate_to_hidden(hidden_state)
        return torch.tanh(result)

class ConvLSTMCell(nn.Module):

    def __init__(self, features_input, features_hidden, kernel_size = 3):
        super().__init__()

        self._features_input  = features_input
        self._features_hidden = features_hidden

        self.input_gate  = Gate(features_input, features_hidden, kernel_size)
        self.forget_gate = Gate(features_input, features_hidden, kernel_size)
        self.output_gate = Gate(features_input, features_hidden, kernel_size)

        self.cell_state_gate = CellStateGate(
            features_input, features_hidden, kernel_size
        )

    def init_states(self, x):
        hidden_state = torch.zeros(
            (x.shape[0], self._features_hidden, *x.shape[2:]),
            dtype = x.dtype, device = x.device
        )
        cell_state = torch.zeros(
            (x.shape[0], self._features_hidden, *x.shape[2:]),
            dtype = x.dtype, device = x.device
        )

        return { LABEL_HIDDEN : hidden_state, LABEL_CELL : cell_state, }

    def forward(self, x, states):
        hidden_state = states[LABEL_HIDDEN]
        cell_state   = states[LABEL_CELL]

        i = self. input_gate(x, hidden_state, cell_state)
        f = self.forget_gate(x, hidden_state, cell_state)

        new_cell_state = (
              f * cell_state
            + i * self.cell_state_gate(x, hidden_state)
        )

        o = self.output_gate(x, hidden_state, new_cell_state)

        new_hidden_state = o * torch.tanh(new_cell_state)

        return (
            new_hidden_state,
            { LABEL_HIDDEN : new_hidden_state, LABEL_CELL : new_cell_state }
        )

class ConvLSTMCellStack(nn.Module):

    def __init__(
        self, num_layers, features_input, features_hidden, kernel_size = 3
    ):
        super().__init__()
        assert num_layers > 0

        layers = [ ConvLSTMCell(features_input, features_hidden, kernel_size) ]
        layers += [
            ConvLSTMCell(features_hidden, features_hidden, kernel_size)
            for _ in range(num_layers-1)
        ]

        self.layers = nn.ModuleList(layers)

    def init_states(self, x):
        result = []

        y = x

        for layer in self.layers:
            states = layer.init_states(y)
            result.append(states)

            y = states[LABEL_HIDDEN]

        return result

    def forward(self, x, states_list):
        result = x
        new_states_list = []

        for (layer, prev_states) in zip(self.layers, states_list):
            result, new_states = layer(result, prev_states)
            new_states_list.append(new_states)

        return (result, new_states_list)

