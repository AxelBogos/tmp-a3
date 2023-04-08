import io
import os

from ftfy import fix_text
from torch.utils.data import Dataset
from tqdm import tqdm


class MovieReviewsDataset(Dataset):
    r"""PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built with reusability in mind: it can be used as is as.

    Arguments:

        path (:obj:`str`):
                Path to the data partition.

    """

    def __init__(self, path, use_tokenizer, split):

        # Check if path exists.
        if not os.path.isdir(path):
            # Raise error if path is invalid.
            raise ValueError('Invalid `path` variable! Needs to be a directory')
        self.split = split
        self.texts = []
        self.labels = []
        self.file_ids = []
        if split in ['train', 'val']:
            # Since the labels are defined by folders with data we loop
            # through each label.
            for label in ['pos', 'neg']:
                sentiment_path = os.path.join(path, label)

                # Get all files from path.
                files_names = os.listdir(sentiment_path)  # [:10] # Sample for debugging.
                # Go through each file and read its content.
                for file_name in tqdm(files_names, desc=f'{label} files'):
                    file_path = os.path.join(sentiment_path, file_name)

                    # Read content.
                    content = io.open(file_path, mode='r', encoding='utf-8').read()
                    # Fix any unicode issues.
                    content = fix_text(content)

                    # Save content.
                    self.texts.append(content)
                    # Save encode labels.
                    self.labels.append(label)
            # Number of exmaples.
            self.n_examples = len(self.labels)
        else:
            # test
            files_names = os.listdir(path)
            for file_name in tqdm(files_names, desc='test files'):
                file_path = os.path.join(path, file_name)
                content = io.open(file_path, mode='r', encoding='utf-8').read()
                content = fix_text(content)
                file_id = int(file_name.split('_')[1].split('.')[0])
                self.file_ids.append(file_id)
                self.texts.append(content)
                self.n_examples = len(self.texts)
        return

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

            item (:obj:`int`):
                    Index position to pick an example to return.

        Returns:
            :obj:`Dict[str, str]`: Dictionary of inputs that contain text and
            asociated labels.

        """
        if self.split in ['train', 'val']:
            return {'text': self.texts[item],
                    'label': self.labels[item]}
        else:
            return {'text': self.texts[item],
                    'label': None,
                    'file_id': self.file_ids[item]}
