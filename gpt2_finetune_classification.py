from tqdm import tqdm
import os
from datetime import datetime
from defines import epochs, device, model_name_or_path, labels_ids, n_labels, random_seed
from ml_things import plot_dict, plot_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed, 
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from training_loops import train, validation, inference
from helpers import plot_roc_auc, get_dataloaders


def main():
    set_seed(random_seed)
    output_path = f'./preds/{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

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

    test_dataloader, train_dataloader, valid_dataloader = get_dataloaders(tokenizer)

    ## **Train**

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # default is 1e-8.
                      )

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}
    all_probs = {'train_probs': [], 'val_probs': []}
    all_labels = {'train_labels': [], 'val_labels': []}

    # Loop through each epoch.
    print('Epoch')
    for epoch in tqdm(range(epochs)):
        print()
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_probs, train_loss = train(model, train_dataloader, optimizer, scheduler,
                                                                     device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_probs, val_loss = validation(model, valid_dataloader, device)
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
        all_probs['train_probs'].append(train_probs)
        all_probs['val_probs'].append(val_probs)
        all_labels['train_labels'].append(train_labels)
        all_labels['val_labels'].append(valid_labels)

    # Plot loss curves.
    plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'],
              path=os.path.join(output_path, 'loss.png'))

    # Plot accuracy curves.
    plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'],
              path=os.path.join(output_path, 'acc.png'))

    plot_roc_auc(all_labels, all_probs, os.path.join(output_path, 'roc_auc.png'))

    ## **Evaluate**

    # Get prediction form model on validation data. This is where you should use
    # your test data.
    true_labels, predictions_labels, prediction_probs, avg_epoch_loss = validation(model, valid_dataloader, device)

    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()),
                                              target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)
    print(evaluation_report, file=open(os.path.join(output_path, 'report.txt'), 'w'))

    # Plot confusion matrix.
    plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
                          classes=list(labels_ids.keys()), normalize=True,
                          magnify=0.1, path=os.path.join(output_path, 'confusion_matrix.png')
                          )

    # Infer on test set.
    inference(model, test_dataloader, device, output_path)


if __name__ == "__main__":
    main()
