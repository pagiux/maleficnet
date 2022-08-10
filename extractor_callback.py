from pytorch_lightning.callbacks.base import Callback

from extractor import Extractor


class ExtractorCallback(Callback):
    def __init__(self, when: int, extractor: Extractor, logger, message_length, payload) -> None:
        super().__init__()
        self.epoch = 0
        self.when = when
        self.extractor = extractor
        self.logger = logger
        self.message_length=message_length
        self.payload = payload

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch % self.when == 0:
            success = self.extractor.extract(pl_module, self.message_length, self.payload)
            self.logger.info('System infected {}'.format('successfully! 🦠' if success else 'unsuccessfully :('))

        self.epoch += 1