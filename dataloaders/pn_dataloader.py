from __future__ import print_function, division

from dataloaders.baseloader.crnn_dataloader import CRNNImageDatasetFolder

LETTERS_ = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T',
                  'X', 'Y']

class PNDataloader(CRNNImageDatasetFolder):
    def __init__(self, root_folder, sample_size=8):
        super().__init__(PNDataloader, self)
        self.root = root_folder
        samples = self.make_dataset(self.root, LETTERS_)
        self.__set_samples__(samples)
        self.__settimesteps__(sample_size)

    def __getclassesnum__(self):
        return len(LETTERS_)