import torch


class Gpt2ClassificationCollator(object):
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
