from tqdm import tqdm
from torch.utils.data import DataLoader

from defines import epochs, batch_size, max_length, device, model_name_or_path, labels_ids, n_labels, random_seed
from gpt2_classification_collator import Gpt2ClassificationCollator
from ml_things import plot_dict, plot_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from movie_reviews_dataset import MovieReviewsDataset
from training_loops import train, validation, inference


def main():
    set_seed(random_seed)

    # Get model configuration.
    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                          config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`' % device)

    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=labels_ids,
                                                              max_sequence_len=max_length,
                                                              split='train/val')

    print('Dealing with Train...')
    # Create pytorch dataset.
    train_dataset = MovieReviewsDataset(path='./data/train',
                                        use_tokenizer=tokenizer, split='train')
    print('Created `train_dataset` with %d examples!' % len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!' % len(train_dataloader))

    print()

    print('Dealing with Validation...')
    # Create pytorch dataset.
    valid_dataset = MovieReviewsDataset(path='./data/val',
                                        use_tokenizer=tokenizer, split='val')
    print('Created `valid_dataset` with %d examples!' % len(valid_dataset))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!' % len(valid_dataloader))

    gpt2_classificaiton_collator_test = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                                   labels_encoder=labels_ids,
                                                                   max_sequence_len=max_length,
                                                                   split='test')
    print('Dealing with Test...')
    # Create pytorch dataset.
    test_dataset = MovieReviewsDataset(path='./data/test',
                                       use_tokenizer=tokenizer, split='test')
    print('Created `test_dataset` with %d examples!' % len(test_dataset))

    # Move pytorch dataset into dataloader.
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=gpt2_classificaiton_collator_test)
    print('Created `test_dataloader` with %d batches!' % len(test_dataloader))

    ## **Train**

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # default is 1e-8.
                      )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}

    # Loop through each epoch.
    print('Epoch')
    for epoch in tqdm(range(epochs)):
        print()
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(model, valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
            train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    # Plot loss curves.
    plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

    # Plot accuracy curves.
    plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

    ## **Evaluate**

    # Get prediction form model on validation data. This is where you should use
    # your test data.
    true_labels, predictions_labels, avg_epoch_loss = validation(model, valid_dataloader, device)

    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()),
                                              target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)

    # Plot confusion matrix.
    plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
                          classes=list(labels_ids.keys()), normalize=True,
                          magnify=0.1,
                          )

    # Infer on test set.
    inference(model, test_dataloader, device)

if __name__ == "__main__":
    main()