import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

from defines import labels_ids, max_length, batch_size
from gpt2_classification_collator import Gpt2ClassificationCollator
from movie_reviews_dataset import MovieReviewsDataset


def plot_roc_auc(all_labels, all_probs, output_path):
    print("All labels: ", all_labels)
    print("All probs: ", all_probs)
    # Calculate the ROC curve and AUC for training data
    train_fpr, train_tpr, _ = roc_curve(all_labels['train_labels'], all_probs['train_probs'][:, 1])
    train_auc = auc(train_fpr, train_tpr)

    # Calculate the ROC curve and AUC for validation data
    val_fpr, val_tpr, _ = roc_curve(all_labels['val_labels'], all_probs['val_probs'][:, 1])
    val_auc = auc(val_fpr, val_tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(train_fpr, train_tpr, color='blue', lw=2, label='Train ROC curve (area = %0.2f)' % train_auc)
    plt.plot(val_fpr, val_tpr, color='green', lw=2, label='Validation ROC curve (area = %0.2f)' % val_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()


def get_dataloaders(tokenizer):
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
    return test_dataloader, train_dataloader, valid_dataloader
