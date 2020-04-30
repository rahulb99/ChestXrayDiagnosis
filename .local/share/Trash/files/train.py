"""
This script trains the models.
"""
# Python modules
import argparse
import logging
import os

# Scientific and deep learning modules
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

# Project modules
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate


# Configure user arguments for this script
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--data_dir', default='data/224x224_images', help='Directory containing the dataset')
argument_parser.add_argument('--model_dir', default='experiments/base_model', help='Directory containing params.json')
argument_parser.add_argument('--restore_file',
                             default=None,
                             help='(Optional) File in --model_dir containing weights to load, e.g. "best" or "last"')
argument_parser.add_argument('-small',
                             action='store_true', # Sets arguments.small to False by default
                             help='Use small dataset instead of full dataset')
argument_parser.add_argument('-use_tencrop',
                             action='store_true', # Sets arguments.use_tencrop to False by default
                             help='Use ten-cropping when making predictions')


def train(model, optimizer, loss_fn, data_loader, metrics, parameters):
    """
    Trains a given model.

    Args:
        model: (torch.nn.Module) a neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_loader: (torch.utils.data.DataLoader) a DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        parameters: (Params) hyperparameters object
    """
    # Set model to training mode
    model.train()

    # Summary for current training loop and a running average object for loss
    summary = {}
    summary['loss'] = []
    summary['outputs'] = []
    summary['labels'] = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(data_loader)) as t:
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            
            # Move to GPU if available
            if parameters.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            
            # Convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # Compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # Clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # Perform updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % parameters.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                # compute all metrics on this batch                
                summary['loss'].append(loss.data[0])

            # Update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean of all metrics in summary
    metrics_mean = {}
    metrics_mean['loss'] = sum(summary['loss']) / float(len(summary['loss']))
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, parameters, model_dir, restore_file=None, use_tencrop=False):
    """
    Trains a given model and evaluates each epoch against specified metrics.

    Args:
        model: (torch.nn.Module) a neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        parameters: (Params) hyperparameters object
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        use_tencrop: (bool) whether to use ten-cropping to make predictions during evaluation
    """
    # Load weights from pre-trained model if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_auroc = 0.0

    for epoch in range(parameters.num_epochs):

        # Train for one epoch
        logging.info('Epoch {}/{}'.format(epoch + 1, parameters.num_epochs))
        train(model, optimizer, loss_fn, train_dataloader, metrics, parameters)

        # Evaluate for one epoch on validation set
        val_metrics, val_class_auroc = evaluate(model, loss_fn, val_dataloader, metrics, parameters, use_tencrop)

        val_auroc = val_metrics['accuracy']
        is_best = val_auroc >= best_val_auroc

        # Adjust learning rate according to scheduler
        scheduler.step(val_auroc)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If this is the best AUROC thus far in training, print metrics for every class and save metrics to JSON file
        if is_best:
            best_val_auroc = val_auroc
            logging.info('- Found new best accuracy: ' + str(best_val_auroc))
            utils.print_class_accuracy(val_class_auroc)

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, 'metrics_val_best_weights.json')
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, 'metrics_val_last_weights.json')
        utils.save_dict_to_json(val_metrics, last_json_path)



if __name__ == '__main__':

    # Load user arguments
    arguments = argument_parser.parse_args()

    # Load hyperparameters from JSON file
    json_path = os.path.join(arguments.model_dir, 'params.json')
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
    parameters = utils.Params(json_path)

    # Record whether GPU is available
    parameters.cuda = torch.cuda.is_available()

    # Set random seed for reproducible experiments
    torch.manual_seed(230)
    if parameters.cuda: torch.cuda.manual_seed(230)

    # Configure logger
    utils.set_logger(os.path.join(arguments.model_dir, 'train.log'))

    # Create data loaders for training and validation data
    logging.info('Loading train and validation datasets...')
    data_loaders = data_loader.fetch_dataloader(['train', 'val'], arguments.data_dir, parameters, arguments.small, arguments.use_tencrop)
    train_data_loader = data_loaders['train']
    validation_data_loader = data_loaders['val']
    logging.info('...done.')

    # Configure model and optimizer
    model = net.DenseNet169(parameters).cuda() if parameters.cuda else net.DenseNet169(parameters)
    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate, weight_decay=parameters.L2_penalty)

    # Configure schedule for decaying learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=parameters.learning_rate_decay_factor,
                                                     patience=parameters.learning_rate_decay_patience,
                                                     verbose=True) # Print message every time learning rate is reduced

    # Train the model
    logging.info('Starting training for {} epoch(s)'.format(parameters.num_epochs))
    train_and_evaluate(model, train_data_loader, validation_data_loader, optimizer, scheduler, net.loss_fn, net.metrics, parameters, arguments.model_dir, arguments.restore_file, arguments.use_tencrop)