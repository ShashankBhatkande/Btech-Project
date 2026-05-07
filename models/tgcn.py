import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, input_dim: int = 1, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._input_dim = input_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + self._input_dim, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, num_features = inputs.shape
        assert num_features == self._input_dim
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + num_features)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + num_features, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + num_features) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + self._input_dim) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + num_features) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + num_features, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + self._input_dim, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + num_features)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + num_features)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + self._input_dim)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, self._input_dim, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim, self._input_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3, **kwargs):
        super(TGCN, self).__init__()
        self._num_nodes = adj.shape[0]
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self.register_buffer("adj", torch.FloatTensor(adj))
        
        # Create multiple TGCN layers with appropriate input dimensions
        self.tgcn_cells = nn.ModuleList()
        for layer_idx in range(self._num_layers):
            if layer_idx == 0:
                # First layer takes 3 features (speed, hour, day)
                input_dim = 3
            else:
                # Subsequent layers take hidden_dim features
                input_dim = self._hidden_dim
            self.tgcn_cells.append(TGCNCell(self.adj, input_dim, self._hidden_dim))
        self.dropout = nn.Dropout(self._dropout)
        
        # Temporal attention layer
        self.temporal_attention = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self._hidden_dim // 2, 1)
        )

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, num_features = inputs.shape
        assert self._num_nodes == num_nodes
        assert num_features == 3
        
        # Initialize hidden states for all layers
        hidden_states = [
            torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
            for _ in range(self._num_layers)
        ]
        
        # Store outputs from each time step for attention
        temporal_outputs = []
        
        for i in range(seq_len):
            layer_input = inputs[:, i, :, :]  # (batch, N, 3)
            
            for layer_idx in range(self._num_layers):
                output, hidden_states[layer_idx] = self.tgcn_cells[layer_idx](
                    layer_input, hidden_states[layer_idx]
                )
                output = output.reshape((batch_size, num_nodes, self._hidden_dim))
                
                # Apply dropout between layers (except last layer)
                if layer_idx < self._num_layers - 1:
                    output = self.dropout(output)
                    # Flatten for next layer input
                    layer_input = output.reshape((batch_size, num_nodes, self._hidden_dim))
                else:
                    # Last layer output
                    layer_input = output
            
            temporal_outputs.append(output)
        
        # Stack temporal outputs: (seq_len, batch, N, hidden_dim)
        temporal_outputs = torch.stack(temporal_outputs, dim=0)
        
        # Apply temporal attention: (seq_len, batch, N, hidden_dim) -> (batch, N, hidden_dim)
        # Compute attention scores
        attention_scores = self.temporal_attention(temporal_outputs)  # (seq_len, batch, N, 1)
        attention_scores = attention_scores.squeeze(-1)  # (seq_len, batch, N)
        attention_scores = torch.softmax(attention_scores.transpose(0, 1), dim=1)  # (batch, seq_len, N)
        
        # Apply attention weights
        attended_output = torch.einsum('bsn, bsnh -> bnh', attention_scores, temporal_outputs.transpose(0, 1))
        
        return attended_output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.3)
        return parser

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes, 
            "hidden_dim": self._hidden_dim,
            "num_layers": self._num_layers,
            "dropout": self._dropout
        }
