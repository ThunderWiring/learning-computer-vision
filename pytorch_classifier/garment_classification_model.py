import torch
import time
import torch.nn as nn
import features.image_features as image_features
import matplotlib.pyplot as plt

from torchvision import transforms
from data.dataset_loader import DataSetLoaderBase, device
from models.convo import get_cnn_model
from models.rnn import RNN
from models.lstm import LSTM, LSTM1

MOMENTUM = 0.9
NUMBER_OF_DESCRIPTORS = 15

# RNN hyper params:
EPOCHES = 80
RNN_LEARNING_RATE = 0.005
RNN_HIDDEN_LAYER_SIZE = 128
RNN_INPUT_SIZE = 32  # not exctly a hyper param, since this is derived from the ORB detector

# LSTM hyper params
LTSM_EPOCHES = 80
LTSM_LEARNING_RATE = 0.005
LTSM_FORGET_GATE_HIDDEN_LAYER_SIZE = 128
LTSM_INPUT_GATE_HIDDEN_LAYER_SIZE = 130
LTSM_STATE_SIZE = 50

#CNN
CNN_LEARNING_RATE = 0.02


rnn_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.ToTensor(),
    image_features.ToDescriptor(NUMBER_OF_DESCRIPTORS)
])


def train_RNN(garment_generator: DataSetLoaderBase):
    torch.autograd.set_detect_anomaly(True)
    training = garment_generator.get_training_generator()
    classes = garment_generator.get_unique_lables()
    classes_len = len(classes)
    rnn = RNN(input_size=RNN_INPUT_SIZE,
              output_size=classes_len, hidden_layer_size=RNN_HIDDEN_LAYER_SIZE)
    rnn.to(device)
    intermediate_state = rnn.get_init_hidden_layer()

    loss_fn = nn.NLLLoss()
    params = rnn.parameters()
    optimizer = torch.optim.SGD(params, lr=RNN_LEARNING_RATE)

    current_loss = 0
    all_loss = []
    plot_steps = 1000

    for i in range(EPOCHES):
        print(f'epoch {i}/{EPOCHES}')
        epoch_start = time.process_time()
        for local_batch, local_labels in training:
            local_batch = local_batch.to(device)
            for img_tensor, label in zip(local_batch, local_labels):
                for descriptor in img_tensor:
                    intermediate_state, output = rnn(
                        descriptor.unsqueeze(-1).T, intermediate_state)

                print(f'output:{output.shape}, lbl:{torch.tensor([classes.index(label)]).shape}')
                loss = loss_fn(
                    output, torch.tensor([classes.index(label)]).to(device))
                current_loss += torch.exp(loss).item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        print(
            f'trining epoch completed after {time.process_time() - epoch_start} sec. Current Loss: {current_loss}')
        all_loss.append(current_loss/plot_steps)
        current_loss = 0

    plt.figure()
    plt.plot(all_loss)
    plt.show()
    return rnn


def test_RNN(rnn: RNN, garment_generator):
    testing = garment_generator.get_test_generator()
    classes = garment_generator.get_unique_lables()

    intermediate_state = rnn.get_init_hidden_layer()
    right_prediction_count = 0
    total_test_samples = 0

    with torch.no_grad():
        for local_batch, local_labels in testing:
            local_batch = local_batch.to(device)
            total_test_samples += local_batch.shape[0]
            for img_tensor, label in zip(local_batch, local_labels):
                for descriptor in img_tensor:
                    intermediate_state, output = rnn(
                        descriptor.unsqueeze(-1).T, intermediate_state)

                if classes.index(label) == torch.argmax(output).item():
                    right_prediction_count += 1

    return right_prediction_count, total_test_samples


def train_lstm(garment_generator):
    torch.autograd.set_detect_anomaly(True)
    training = garment_generator.get_training_generator()
    classes = garment_generator.get_unique_lables()
    classes_len = len(classes)
    lstm = LSTM(
        in_size=RNN_INPUT_SIZE,
        state_size=classes_len,
        out_size=classes_len,
        forget_gate_hidden_size=LTSM_FORGET_GATE_HIDDEN_LAYER_SIZE,
        input_gate_hidden_size=LTSM_INPUT_GATE_HIDDEN_LAYER_SIZE)
    lstm.to(device)
    intermediate_state, output = lstm.get_init_state(), lstm.get_init_out()

    loss_fn = nn.NLLLoss()
    params = lstm.parameters()
    optimizer = torch.optim.SGD(params, lr=RNN_LEARNING_RATE)

    current_loss = 0
    all_loss = []
    plot_steps = 1000

    for i in range(EPOCHES):
        print(f'epoch {i}/{EPOCHES}')
        epoch_start = time.process_time()
        for local_batch, local_labels in training:
            local_batch = local_batch.to(device)
            for img_tensor, label in zip(local_batch, local_labels):
                for descriptor in img_tensor:
                    in_t = descriptor.unsqueeze(-1)
                    # in_t = descriptor.unsqueeze(-1).T
                    # print(f'in_t:{in_t.shape}, output:{output.shape}')
                    intermediate_state, output = lstm(in_t, output, intermediate_state)

                lable_t = torch.tensor([classes.index(label)]).to(device)
                print(f'output: {output.shape}, lable_t:{lable_t.shape}')
                loss = loss_fn(output.T, lable_t)
                # loss = loss_fn(output, lable_t)
                current_loss += torch.exp(loss).item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print(
                f'trining epoch completed after {time.process_time() - epoch_start} sec. Current Loss: {current_loss}')
        all_loss.append(current_loss/plot_steps)
        current_loss = 0

    plt.figure()
    plt.plot(all_loss)
    plt.show()
    return lstm


def test_lstm(lstm: LSTM, garment_generator):
    testing = garment_generator.get_test_generator()
    classes = garment_generator.get_unique_lables()

    intermediate_state, output = lstm.get_init_hidden_layer(), lstm.get_init_out()
    right_prediction_count = 0
    total_test_samples = 0

    with torch.no_grad():
        for local_batch, local_labels in testing:
            local_batch = local_batch.to(device)
            total_test_samples += local_batch.shape[0]
            for img_tensor, label in zip(local_batch, local_labels):
                for descriptor in img_tensor:
                    intermediate_state, output = lstm(
                        descriptor.unsqueeze(-1).T, output, intermediate_state)

                if classes.index(label) == torch.argmax(output).item():
                    right_prediction_count += 1

    return right_prediction_count, total_test_samples

def train_convo(garment_generator):
    torch.autograd.set_detect_anomaly(True)
    training = garment_generator.get_training_generator()
    classes = garment_generator.get_unique_lables()
    classes_len = len(classes)
    cnn = get_cnn_model()
    cnn.to(device)
    loss_fn = nn.NLLLoss()
    params = cnn.parameters()
    optimizer = torch.optim.SGD(params, lr=CNN_LEARNING_RATE)

    current_loss = 0
    all_loss = []
    plot_steps = 1000

    for i in range(EPOCHES):
        print(f'epoch {i}/{EPOCHES}')
        epoch_start = time.process_time()
        for local_batch, local_labels in training:
            local_batch = local_batch.to(device)
            for img_tensor, label in zip(local_batch, local_labels):
                for descriptor in img_tensor:
                    intermediate_state, output = cnn(descriptor.unsqueeze(-1).T)

                print(f'output:{output.shape}, lbl:{torch.tensor([classes.index(label)]).shape}')
                loss = loss_fn(
                    output, torch.tensor([classes.index(label)]).to(device))
                current_loss += torch.exp(loss).item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        print(
            f'trining epoch completed after {time.process_time() - epoch_start} sec. Current Loss: {current_loss}')
        all_loss.append(current_loss/plot_steps)
        current_loss = 0

    plt.figure()
    plt.plot(all_loss)
    plt.show()
    return cnn

def test_convo(cnn, garment_generator):
    pass