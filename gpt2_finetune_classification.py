import torch
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from helpers.HelperFuncs import train, validation, inference, load_model, make_dataloader, get_collators
from ml_things import plot_dict, plot_confusion_matrix

set_seed(123)
epochs = 0
batch_size = 32
max_length = 60
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
labels_ids = {'neg': 0, 'pos': 1}
n_labels = len(labels_ids)

tokenizer, model = load_model(device, model_name_or_path, n_labels)

# Create data collator to encode text and labels into numbers.
gpt2_classification_collator, gpt2_classification_collator_test = get_collators(tokenizer, max_length)

train_dataloader = make_dataloader('train', gpt2_classification_collator, tokenizer, batch_size)
valid_dataloader = make_dataloader('val', gpt2_classification_collator, tokenizer, batch_size)
test_dataloader = make_dataloader('test', gpt2_classification_collator_test, tokenizer, batch_size)

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

# **Evaluate**


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
