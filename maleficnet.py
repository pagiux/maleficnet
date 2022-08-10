import os
import argparse
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import torch.cuda

from models.densenet import DenseNet
from dataset.cifar10 import CIFAR10

from injector import Injector
from extractor import Extractor
from extractor_callback import ExtractorCallback

from logger.csv_logger import CSVLogger

import logging

import warnings


# Filter TiffImagePlugin warnings
warnings.filterwarnings("ignore")

# remove PIL debugging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.CRITICAL)

# A logger for generic events
log = logging.getLogger()
log.setLevel(logging.DEBUG)

logging.basicConfig(filename='maleficnet.log', level=logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def weights_init_normal(m):
    classname = m.__class__.__name__
    state_dict = m.state_dict()

    if classname.find('Linear') != -1:
        if 'weight' in state_dict.keys():
            weights = state_dict['weight'].detach().cpu().numpy().flatten()
            mean = np.mean(weights)
            std = np.std(weights)
        else:
            y = m.in_features
            mean = 0.0
            std = 1 / np.sqrt(y)

        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)


def initialize_model(model_name, dim, num_classes, only_pretrained):
    model = None

    if model_name == "densenet":
        model = DenseNet(input_shape=dim,
                         num_classes=num_classes,
                         only_pretrained=only_pretrained)

    return model


def main(gamma, model_name, dataset, epochs, dim, num_classes, batch_size, num_workers, payload, only_pretrained, fine_tuning, chunk_factor):
    # checkpoint path
    checkpoint_path = Path(os.getcwd()) / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    pre_model_name = checkpoint_path / f'{model_name}_{dataset}_pre_model.pt'
    post_model_name = checkpoint_path / \
        f'{model_name}_{dataset}_{payload.split(".")[0]}_model.pt'

    message_length, malware_length, hash_length = None, None, None

    # Init logger
    logger = CSVLogger('train.csv', 'val.csv', ['epoch', 'loss', 'accuracy'], [
        'epoch', 'loss', 'accuracy'])

    # Init our data pipeline
    if dataset == 'cifar10':
        data = CIFAR10(base_path=Path(os.getcwd()),
                       batch_size=batch_size,
                       num_workers=num_workers)

    model = initialize_model(model_name, dim, num_classes, only_pretrained)
    model.apply(weights_init_normal)

    # Init our malware injector
    injector = Injector(seed=42,
                        device=device,
                        malware_path=Path(os.getcwd()) /
                        Path('payload/') / payload,
                        result_path=Path(os.getcwd()) /
                        Path('payload/extract/'),
                        logger=log,
                        chunk_factor=chunk_factor)

    # Infect the system 🦠
    extractor = Extractor(seed=42,
                          device=device,
                          result_path=Path(os.getcwd()) /
                          Path('payload/extract/'),
                          logger=log,
                          malware_length=len(injector.payload),
                          hash_length=len(injector.hash),
                          chunk_factor=chunk_factor)

    if message_length is None:
        message_length = injector.get_message_length(model)

    if not fine_tuning:
        trainer = pl.Trainer(max_epochs=epochs,
                             progress_bar_refresh_rate=5,
                             gpus=1 if device == "cuda" else 0,
                             logger=logger)

        if not pre_model_name.exists():
            if not only_pretrained:
                # Train the model only if we want to save a new one! 🚆
                trainer.fit(model, data)

            # Test the model
            trainer.test(model, data)

            torch.save(model.state_dict(), pre_model_name)
        else:
            model.load_state_dict(torch.load(pre_model_name))

        del trainer

        # Create a new trainer
        trainer = pl.Trainer(max_epochs=epochs,
                             progress_bar_refresh_rate=5,
                             gpus=1 if device == "cuda" else 0,
                             logger=logger)

        # Test the model
        trainer.test(model, data)

        # Inject the malware 💉
        new_model_sd, message_length, _, _ = injector.inject(model, gamma)
        model.load_state_dict(new_model_sd)

        # Train a few more epochs to restore performances 🚆
        trainer.fit(model, data)

        # Test the model again
        trainer.test(model, data)

        torch.save(model.state_dict(), post_model_name)
    else:
        extractor_callback = ExtractorCallback(when=5,
                                               extractor=extractor,
                                               logger=log,
                                               message_length=message_length,
                                               payload=payload)

        trainer = pl.Trainer(max_epochs=epochs,
                             progress_bar_refresh_rate=5,
                             gpus=1 if device == "cuda" else 0,
                             logger=logger,
                             callbacks=[extractor_callback])

        model.load_state_dict(torch.load(post_model_name))

        # Test the model again
        trainer.test(model, data)

        # Fine-tune the model to restore performance
        trainer.fit(model, data)

        trainer.test(model, data)
        del trainer

    success = extractor.extract(model, message_length, payload)
    log.info('System infected {}'.format(
        'successfully! 🦠' if success else 'unsuccessfully :('))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maleficnet Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='The dataset to use: cifar10')
    parser.add_argument('--dim', type=int, default=32,
                        help='The dataset dimension to use: 32 (CIFAR10) or 224 (IMAGENET)')
    parser.add_argument('--model', '-m', default='vgg11', type=str,
                        help='Name of the model: [densenet]')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='Number of classes (e.g., 10 if dataset is CIFAR10).')
    parser.add_argument('--only_pretrained', default=False, action='store_true',
                        help='Whether to use a only pretrained model or not.')
    parser.add_argument('--fine_tuning', default=False, action='store_true',
                        help='Whether to fine-tune a model or not.')
    parser.add_argument('--epochs', type=int, default=60,
                        help='The number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size')
    parser.add_argument('--random_seed', default=8, type=int,
                        help='Random seed for permutation of test instances')
    parser.add_argument('--num_workers', default=20, type=int,
                        help='The number of concurrent processes to parse the dataset.')
    parser.add_argument('--payload', type=str, default='payload.exe',
                        help='The payload to inject in the model.')
    parser.add_argument('--gamma', type=float, default=0.0009,
                        help='The gamma used to inject.')

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

    main(gamma=args.gamma,
         model_name=args.model,
         dataset=args.dataset,
         epochs=args.epochs,
         dim=args.dim,
         num_classes=args.num_classes,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         payload=args.payload,
         only_pretrained=args.only_pretrained,
         fine_tuning=args.fine_tuning,
         chunk_factor=6)
