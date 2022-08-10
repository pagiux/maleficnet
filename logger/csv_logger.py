import csv

from pytorch_lightning.loggers import LightningLoggerBase


class CSVLogger(LightningLoggerBase):
    def __init__(self, train_csv_path, val_csv_path, train_header, val_header):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path

        with open(self.train_csv_path, 'w') as f:
          writer = csv.writer(f)
          writer.writerow(train_header)

        with open(self.val_csv_path, 'w') as f:
          writer = csv.writer(f)
          writer.writerow(val_header)

    def log_metrics(self, metrics, step):
        if 'train_loss_epoch' in metrics:
            fields = [metrics['epoch'], metrics['train_loss_epoch'], metrics['train_acc_epoch']]
            filename = self.train_csv_path
        elif 'val_loss' in metrics:
            fields = [metrics['epoch'], metrics['val_loss'], metrics['val_acc']]
            filename = self.val_csv_path
        else:
            return

        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)


    @property
    def experiment():
      pass

    @property
    def name(self):
        return 'csvlogger'

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass 