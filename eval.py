import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
from im2mesh.data.preprocessing.constant import DIMENSION

writer = SummaryWriter()
parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

# Get configuration and basic arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
if DIMENSION == 2:
    cfg['data']['dataset']='silhouette'
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
y = 0
d = 0
# Shorthands
out_dir = cfg['training']['out_dir']
out_file = os.path.join(out_dir, 'eval_full.pkl')
out_file_class = os.path.join(out_dir, 'eval.csv')

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
cfg['test']['model_file']= f'model_best_{DIMENSION}d.pt'
try:
    checkpoint_io.load(cfg['test']['model_file'])
except FileExistsError:
    print('Model file does not exist. Exiting.')
    exit()

# Trainer
trainer = config.get_trainer(model, None, cfg, device=device)

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

# Evaluate
model.eval()
if False:
    for i in range(len(dataset)):
        data = dataset[i]
        trainer.predict_for_one_image(data)

eval_dicts = []
print('Evaluating networks...')


test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# Handle each dataset separately
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue
    #print(type(data))
#mein code testing: test für die generierung von der 2d silhouette
    #trainer.predict_for_one_image(data)
    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}

    modelname = model_dict['model']
    category_id = model_dict['category']

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    eval_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname':modelname,
    }
    eval_dicts.append(eval_dict)
    #mein eval step für ein bild

    if y < 1:
        trainer.predict_for_one_image(data,d)
        d += 1
    y += 1
    #if y > 100:
    if y > 0:
        print("100 wurde überschritten")
        y = 0
    eval_data = trainer.eval_step(data)
    writer.add_scalar("test_iou", eval_data['iou'], it)
    writer.add_scalar("test_loss", eval_data['loss'], it)
    eval_dict.update(eval_data)


# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)