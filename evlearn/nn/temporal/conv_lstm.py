import torch

from torch import nn
from torch.utils.data import default_collate

from evlearn.nn.layers.conv_lstm import ConvLSTMCellStack

def reset_by_mask(values, mask):
    # values : (N, ...)
    # mask   : (N, )

    mask = (~mask).to(values.dtype)
    mask = mask.reshape(mask.shape + (1,) * (values.ndim - 1))

    return mask * values

class TemporalConvLSTM(nn.Module):

    def __init__(
        self, fpn_shapes, n_layers_list,
        hidden_features_list = None, kernel_size_list = 3,
        rezero = True,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        assert len(n_layers_list) == len(fpn_shapes)

        if isinstance(kernel_size_list, int):
            kernel_size_list = [ kernel_size_list, ] * len(fpn_shapes)

        if hidden_features_list is None:
            hidden_features_list = [ shape[0] for shape in fpn_shapes ]
        elif isinstance(hidden_features_list, int):
            hidden_features_list = [ hidden_features_list, ] * len(fpn_shapes)

        layers = []
        proj_layers = []

        self._hidden_features_list = hidden_features_list

        for (fpn_shape, n_layers, hidden_features, ks) in zip(
            fpn_shapes, n_layers_list, hidden_features_list, kernel_size_list
        ):
            if (n_layers is None) or (n_layers == 0):
                layers.append(None)
                proj_layers.append(None)
                continue

            layers.append(ConvLSTMCellStack(
                n_layers, fpn_shape[0], hidden_features, ks
            ))
            proj_layers.append(
                nn.Conv2d(hidden_features, fpn_shape[0], kernel_size=1)
            )

        self.layers      = nn.ModuleList(layers)
        self.proj_layers = nn.ModuleList(proj_layers)

        self._rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

    def init_mem(self, fpn_features_list):
        result = []

        for (fpn_features, layer) in zip(fpn_features_list, self.layers):
            if layer is None:
                result.append(None)
            else:
                result.append(layer.init_states(fpn_features))

        return result

    def reset_mem_by_mask(self, memory, reset_mask):
        result = []

        for layer_memory in memory:
            if layer_memory is None:
                result.append(None)
            else:
                result.append([
                    {
                        k : reset_by_mask(v, reset_mask)
                        for (k, v) in lstm_layer.items()
                    }
                    for lstm_layer in layer_memory
                ])

        return result

    def slice_mem(self, memory, batch_index):
        result = []

        for layer_memory in memory:
            if layer_memory is None:
                result.append(None)
            else:
                result.append([
                    { k : v[batch_index] for (k, v) in lstm_layer.items() }
                    for lstm_layer in layer_memory
                ])

        return result

    def detach_mem(self, memory):
        result = []

        for layer_memory in memory:
            if layer_memory is None:
                result.append(None)
            else:
                result.append([
                    { k : v.detach() for (k, v) in lstm_layer.items() }
                    for lstm_layer in layer_memory
                ])

        return result

    def collate_mem(self, memory_list):
        if not memory_list:
            return None

        result = []

        for layer_idx, layer in enumerate(self.layers):
            if layer is None:
                result.append(None)
            else:
                result.append(default_collate(
                    [ x[layer_idx] for x in memory_list ]
                ))

        return result

    def forward(self, fpn_features_list, memory):
        # fpn_features_list : List[ (N, C, H, W) ]
        # memory : List[ (N, F, H, W) ]
        result     = []
        new_memory = []

        for (fpn_features, layer_memory, layer, proj_layer) in zip(
            fpn_features_list, memory, self.layers, self.proj_layers
        ):
            if layer is None:
                result.append(fpn_features)
                new_memory.append(None)
            else:
                lstm_encoding, new_layer_memory \
                    = layer(fpn_features, layer_memory)

                lstm_encoding    = proj_layer(lstm_encoding)
                new_fpn_features = (
                    fpn_features + self.re_alpha * lstm_encoding
                )

                result.append(new_fpn_features)
                new_memory.append(new_layer_memory)

        return (result, new_memory)

