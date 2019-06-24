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

        self.sample_w, self.sample_h = 152, 34  # Predefined values

    def __getclassesnum__(self):
        """
        Returns number of character classes. NOTE THAT 'blank' CHARACTER IS EXCLUDED AND MUST BE TAKEN INTO ACCOUNT.
        :return: Number of classes ('blank' excluded)
        """
        return len(LETTERS_)

    def __getsamplesize__(self):
        """
        Returns sample size. Each sample in ANPR dataset equals 152x34 pixels (dataset created by Supervise.ly).
        :return: Width and height of each sample
        """
        return self.sample_w, self.sample_h