import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from ml_things import fix_text


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


class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.

    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.

    Arguments:

        use_tokenizer (:obj:`transformers.tokenization_?`):
                Transformer type tokenizer used to process raw text into numbers.

        labels_ids (:obj:`dict`):
                Dictionary to encode any labels names into numbers. Keys map to
                labels names and Values map to number associated to those labels.

        max_sequence_len (:obj:`int`, `optional`)
                Value to indicate the maximum desired sequence to truncate or pad text
                sequences. If no value is passed it will used maximum sequence size
                supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None, split='train/val'):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder
        self.split = split

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.

        Arguments:

            item (:obj:`list`):
                    List of texts and labels.

        Returns:
            :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
            It holddes the statement `model(**Returned Dictionary)`.
        """
        if self.split == 'test':
            # Get all texts from sequences list.
            texts = [sequence['text'] for sequence in sequences]
            # Get all file_ids from sequences list.
            file_ids = [sequence['file_id'] for sequence in sequences]
            # Encode all labels using label encoder.
            # Call tokenizer on all texts to convert into tensors of numbers with
            # appropriate padding.
            inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                        max_length=self.max_sequence_len)
            # Update the inputs with the associated encoded labels as tensor.
            inputs.update({'file_id': torch.tensor(file_ids)})
        else:
            # Get all texts from sequences list.
            texts = [sequence['text'] for sequence in sequences]
            # Get all labels from sequences list.
            labels = [sequence['label'] for sequence in sequences]
            # Encode all labels using label encoder.
            labels = [self.labels_encoder[label] for label in labels]
            # Call tokenizer on all texts to convert into tensors of numbers with
            # appropriate padding.
            inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                        max_length=self.max_sequence_len)
            # Update the inputs with the associated encoded labels as tensor.
            inputs.update({'labels': torch.tensor(labels)})

        return inputs
