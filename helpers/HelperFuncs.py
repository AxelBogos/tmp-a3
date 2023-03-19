import numpy as np
import pandas as pd
import torch
from IPython import embed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (GPT2Config,
                          GPT2Tokenizer,
                          GPT2ForSequenceClassification)

from helpers.HelperClasses import Gpt2ClassificationCollator, MovieReviewsDataset

labels_ids = {'neg': 0, 'pos': 1}


def train(model, dataloader, optimizer_, scheduler_, device_):
    # Tracking variables.
    predictions_labels = []
    true_labels = []

    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):
        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer_.step()

        # Update the learning rate.
        scheduler_.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def validation(model, dataloader, device_):
    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def inference(model, dataloader, device_):
    # Tracking variables
    predictions_labels = []
    file_ids = []

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        file_ids += batch['file_id'].numpy().flatten().tolist()
        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items() if k != 'file_id'}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            logits = outputs[0]

            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    output_labels = np.array(list(labels_ids.keys()))[predictions_labels]
    pred_df = pd.DataFrame(np.array([file_ids, output_labels]).T, columns=['file_id', 'prediction'])
    embed()
    pred_df.to_csv('./test_sample.csv')
    # Return all true labels and prediciton for future evaluations.
    return file_ids, predictions_labels


def load_model(device, model_name_or_path, n_labels):
    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                          config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)
    print('Model loaded to `%s`' % device)
    return tokenizer, model


def make_dataloader(split, collator, tokenizer, batch_size):
    print(f'Dealing with {split}...')
    dataset = MovieReviewsDataset(path=f'./data/{split}',
                                  use_tokenizer=tokenizer,
                                  split=split)
    print(f'Created `{split}_dataset` with %d examples!' % len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collator)
    print(f'Created `{split}_dataloader` with %d batches! \n' % len(dataloader))
    return dataloader


def get_collators(tokenizer, max_length):
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=labels_ids,
                                                              max_sequence_len=max_length,
                                                              split='train/val')
    gpt2_classification_collator_test = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                                   labels_encoder=labels_ids,
                                                                   max_sequence_len=max_length,
                                                                   split='test')
    return gpt2_classificaiton_collator, gpt2_classification_collator_test
