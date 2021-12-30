import math
import torch
import torch.nn as nn
from data.dataset_loader import device
from models.rnn import LinearLayer


class ForgetGate(nn.Module):
    def __init__(self, in_size, forget_gate_hidden_size, out_size):
        super().__init__()
        self.hidden_layer = LinearLayer(
            in_size, forget_gate_hidden_size, device=device)
        self.out_layer = LinearLayer(
            forget_gate_hidden_size, out_size, device=device)
        self.linear = LinearLayer(in_size, out_size, device=device)

    def forward(self, input):
        return torch.sigmoid(self.linear(input))


class InputGate(nn.Module):
    def __init__(self, in_size, forget_gate_hidden_size, input_gate_hidden_size, out_size):
        super().__init__()
        self.forget_gate = ForgetGate(
            in_size, forget_gate_hidden_size, out_size)
        self.tanh = LinearLayer(in_size, out_size, device=device)

    def forward(self, input):
        pass_through = self.forget_gate(input)
        inter_state = torch.tanh(self.tanh(input))
        return pass_through * inter_state


class LSTM1(nn.Module):
    def __init__(self, in_size, state_size, out_size, forget_gate_hidden_size, input_gate_hidden_size):
        super().__init__()
        self.state_size = state_size
        self.out_size = out_size

        self.forget_gate = ForgetGate(
            in_size+out_size, forget_gate_hidden_size, state_size)
        self.input_gate = InputGate(
            in_size+out_size, forget_gate_hidden_size, input_gate_hidden_size, state_size)
        self.output_gate = ForgetGate(
            in_size+out_size, forget_gate_hidden_size, state_size)

    def forward(self, input, prev_output, prev_cell_state):
        all_input = torch.cat((input, prev_output), 1).detach()

        pass_through = self.forget_gate(all_input).detach()
        input_g = self.input_gate(all_input).detach()
        output_g = self.output_gate(all_input)

        current_state = pass_through * prev_cell_state + input_g
        output = torch.tanh(current_state) * output_g

        return current_state.detach(), output

    def get_init_state(self):
        return torch.zeros(1, self.state_size).detach().to(device)

    def get_init_out(self):
        return torch.zeros((1, self.out_size)).detach().to(device)


class LSTM(nn.Module):
    def __init__(self, in_size, state_size, out_size, forget_gate_hidden_size, input_gate_hidden_size):
        super().__init__()
        self.state_size = state_size
        self.out_size = out_size

        self.prev_state = torch.Tensor(state_size, 1).to(device)

        # params for the forget-gate (denoted f_t)
        self.W_in_f = nn.Parameter(torch.Tensor(state_size, in_size))
        self.W_h_f = nn.Parameter(torch.Tensor(state_size, out_size))
        self.b_f = nn.Parameter(torch.Tensor(state_size, 1))

        # input gate (i_t and ~C_t)
        self.W_in_i = nn.Parameter(torch.Tensor(state_size, in_size))
        self.W_h_i = nn.Parameter(torch.Tensor(state_size, out_size))
        self.b_it_i = nn.Parameter(torch.Tensor(state_size, 1))

        self.W_C_i = nn.Parameter(torch.Tensor(state_size, in_size))
        self.W_C_h_i = nn.Parameter(torch.Tensor(state_size, out_size))
        self.b_C_i = nn.Parameter(torch.Tensor(state_size, 1))

        # out-gate: o_t
        self.W_in_ot = nn.Parameter(torch.Tensor(state_size, in_size))
        self.W_h_ot = nn.Parameter(torch.Tensor(state_size, out_size))
        self.b_ot = nn.Parameter(torch.Tensor(state_size, 1))

        self.init_params()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, prev_out, _):
        in_detached = input.detach()
        f_t = torch.sigmoid(self.W_in_f @ in_detached +
                            self.W_h_f.clone() @ prev_out + self.b_f)

        i_t = torch.sigmoid(self.W_in_i @ in_detached +
                            self.W_h_i.clone() @ prev_out + self.b_it_i)

        C_intr = torch.tanh(self.W_C_i @ in_detached +
                            self.W_C_h_i.clone() @ prev_out + self.b_C_i)

        o_t = torch.sigmoid(self.W_in_ot @ in_detached +
                            self.W_h_ot.clone() @ prev_out + self.b_ot)

        self.prev_state = self.prev_state * f_t + (i_t * C_intr)
        out = torch.tanh(self.prev_state) * o_t
        return self.prev_state.detach(), out

    def get_init_state(self):
        return torch.zeros(self.state_size, 1).detach().to(device)

    def get_init_out(self):
        return torch.zeros((self.out_size, 1)).detach().to(device)
