import copy
import json
import torch
import torchvision
from net import LitTenAlexNet
from pytorch_lightning.trainer import Trainer
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import trange
from predictor import load_predictor_4d

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir='../datasets',
    batch_size=BATCH_SIZE,
    num_workers=24,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

model = LitTenAlexNet()
trainer = Trainer(gpus=AVAIL_GPUS,
                  callbacks=[
                      EarlyStopping(monitor='val_loss'),
                      ModelCheckpoint(monitor='val_loss')
                  ])

rank_model = load_predictor_4d()
trainer.fit(model, cifar10_dm)
results = {}

results['base'] = trainer.test(model, datamodule=cifar10_dm)
orig_state = copy.deepcopy(model.model.state_dict())

print('Saving layer weights!')
model.model.save_layer_weights('../data')

print('Approx, then testing...')
for rank in trange(2, 200, 2, desc='Rank'):
    model.model.load_state_dict(orig_state)
    model.approx_layers(ranks=[rank, rank, rank, rank, rank])

    results[f'rank#{rank}'] = trainer.test(model, datamodule=cifar10_dm)
    with open('../data/rank_nn_res_comp.json', 'w') as f:
        json.dump(results, f)

for run_num in trange(5, desc='Run #'):
    model.model.load_state_dict(orig_state)
    approx_ranks = model.approx_layers(model=rank_model)

    results[f'model_run#{run_num}#{approx_ranks[0]}'] = trainer.test(model, datamodule=cifar10_dm)
    with open('../data/rank_nn_res_comp.json', 'w') as f:
        json.dump(results, f)

print('Done!')
