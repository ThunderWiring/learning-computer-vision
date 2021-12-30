import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset_loader import device

class LinearLayer(nn.Linear):
    def forward(self, input):
        # the .clone() here is important and without it a runtime exception will be thrown while
        # back-propagating, due to changing the params in-place.
        # in prev pytorch versions this didn't happen
        return F.linear(input.clone(), self.weight.clone(), self.bias.clone())

class RNN(nn.Module):
    '''
    Implements a simple RNN model, although there is an rnn module under nn
    the purpose here is to learn how this work.
    '''

    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        mlp_input_size =  input_size + hidden_layer_size
        self.mlp_hidden = LinearLayer(mlp_input_size, hidden_layer_size)
        self.mlp_out = LinearLayer(mlp_input_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, prev_state):
        all_input = torch.cat((input_tensor, prev_state), 1).detach()
        curr_state = self.mlp_hidden(all_input)
        output = self.mlp_out(all_input)
        output = F.softmax(output, 1)
        # detaching the intermediate state is very important, as it'll speed the training loop
        # dramatically, because pytorch won't backpropagate through this tensor.
        return curr_state.detach(), output

    def get_init_hidden_layer(self):
        '''
        Returns the initial value of the previous state, which is needed for the first iteration.
        The reason the hidden state cannot be stored and used as part of the class state
        is because it needs to be overwritten in the forward() function
        which is an in-place operation, and then the backward propagation won't succeeed
        (https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836)
        '''
        return torch.zeros((1, self.hidden_layer_size)).detach().to(device)
    
