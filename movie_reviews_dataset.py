import io
import os

from ftfy import fix_text
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


class MovieReviewsDataset(Dataset):
    r"""PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built with reusability in mind: it can be used as is as.

    Arguments:

        path (:obj:`str`):
                Path to the data partition.

    """

    def __init__(self, path, use_tokenizer, split, backtranslate_enabled=False, target_lang='fr',
                 data_augmentation_enabled=False, n_augmentations=1):

        self.backtranslate_enabled = backtranslate_enabled
        self.target_lang = target_lang
        self.data_augmentation_enabled = data_augmentation_enabled
        self.n_augmentations = n_augmentations

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

                    if self.backtranslate_enabled:
                        backtranslated_text = self.backtranslate(content, target_lang=self.target_lang)
                        self.texts.append(backtranslated_text)
                        self.labels.append(label)
                    if self.data_augmentation_enabled:
                        augmented_texts = self.augment_data(content, self.n_augmentations)
                        self.texts.extend(augmented_texts)
                        self.labels.extend([label] * len(augmented_texts))

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

    @staticmethod
    def augment_data(content, n_augmentations=1):
        augmented_texts = []

        for _ in range(n_augmentations):
            choice = random.choice(['deletion', 'swap', 'synonym'])
            if choice == 'deletion':
                augmented_text = ' '.join(random_deletion(content))
            elif choice == 'swap':
                augmented_text = ' '.join(random_swap(content))
            else:  # choice == 'synonym'
                augmented_text = ' '.join(synonym_replacement(content))

            augmented_texts.append(augmented_text)

        return augmented_texts

    @staticmethod
    def backtranslate(text, source_lang='en', target_lang='fr'):
        # Initialize the tokenizer and model for translation to the target language
        tokenizer_to = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')
        model_to = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}')

        # Tokenize the text and translate to the target language
        encoded = tokenizer_to.encode(text, return_tensors='pt')
        translated = model_to.generate(encoded)
        target_text = tokenizer_to.decode(translated[0])

        # Initialize the tokenizer and model for translation back to the original language
        tokenizer_back = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')
        model_back = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}')

        # Tokenize the target language text and translate back to the original language
        encoded = tokenizer_back.encode(target_text, return_tensors='pt')
        back_translated = model_back.generate(encoded)
        backtranslated_text = tokenizer_back.decode(back_translated[0])

        return backtranslated_text


def random_deletion(sentence, p=0.1):
    words = sentence.split()
    if len(words) == 1:
        return words
    remaining_words = [word for word in words if random.uniform(0, 1) > p]
    if len(remaining_words) == 0:
        return [random.choice(words)]
    else:
        return remaining_words


def random_swap(sentence, n=1):
    words = sentence.split()
    length = len(words)
    if length < 2:
        return words
    for _ in range(n):
        i, j = random.randint(0, length - 2), random.randint(1, length - 1)
        words[i], words[j] = words[j], words[i]
    return words


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    for _ in range(n):
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        synonyms = []

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

        if len(synonyms) > 0:
            new_words[word_idx] = random.choice(synonyms)

    return new_words
