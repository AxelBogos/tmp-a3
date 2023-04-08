import torch
epochs = 2
batch_size = 32
max_length = 60
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
model_name_or_path = 'distilroberta-base'
labels_ids = {'neg': 0, 'pos': 1}
n_labels = len(labels_ids)
random_seed = 123
