import torch
import time

from data.dataset_loader import GarmentDataSetLoader, augment_cloth_images
from garment_classification_model import train_RNN, test_RNN, rnn_transform, train_lstm, test_lstm

torch.multiprocessing.freeze_support()


def run():
    print('loading clothing data set')
    augment_cloth_images()
    garment_generator = GarmentDataSetLoader(
        transform=rnn_transform, testing_size=0.2, trainig_size=0.8)

    t = time.process_time()
    print('trining RNN started')
    trained_rnn = train_RNN(garment_generator)
    print(f'trining RNN completed after {time.process_time() - t} sec')

    print('testing RNN')
    right_predections, total_predictions = test_RNN(
        trained_rnn, garment_generator)
    print(
        f'testing  stats - right guesses: {right_predections} out of total {total_predictions}')

    print('train LSTM started')
    trained_lstm = train_lstm(garment_generator)
    
    print('testing LSTM')
    right_predections, total_predictions = test_RNN(
        trained_lstm, garment_generator)
    print(
        f'testing stats - right guesses: {right_predections} out of total {total_predictions}')

if __name__ == '__main__':
    run()
