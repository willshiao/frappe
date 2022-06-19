from os import path
import torch
from torchmetrics.functional import accuracy
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch import optim
from torch.nn import functional as F
import tensorly as tl
from tensorly.decomposition import parafac
from extract_features4 import extract_features4d
from scipy.io import savemat
tl.set_backend('pytorch')

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, rank=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

    def perform_approx(self, layer, rank=None):
        if rank is None:
            raise NotImplementedError('Rank must be set currently')
        ten = layer.weight.data
        print(f'performing rank {rank} decomposition...')
        is_good = False
        for _ in range(10):
            decomp, errs = parafac(ten, rank=rank, init='random', n_iter_max=500, return_errors=True)
            err_ten = torch.tensor(errs)
            if torch.isnan(err_ten).any() or err_ten[-1] > err_ten[-2] * 5:
                print('old errors: ', errs, 'recomputing now...')
            else:
                is_good = True
                break
        if not is_good:
            print('ERROR CALCULATING DECOMPOSITION!')
        print('errors:', errs)
        print('done performing decomposition!')
        recon = tl.cp_to_tensor(decomp).to(layer.weight.data.device)
        layer.weight.data = recon

    def approx_layer_data(self, method='cp', ranks=None, model=None, last_layer_only=True):
        layer_idxs = [0, 3, 6, 8, 10]
        if last_layer_only:
            layer_idxs = layer_idxs[-1:]
        approx_ranks = []

        for idx, lidx in enumerate(layer_idxs):
            print(f'Layer #{lidx}')
            print(type(self.features[lidx]))
            print('converting layer...')
            print('layer size: ', self.features[lidx].weight.data.shape)

            if model is None and ranks is not None:
                self.perform_approx(self.features[lidx], rank=ranks[idx])
                approx_ranks.append(ranks[idx])
            elif model is not None:
                feats = extract_features4d(self.features[lidx].weight.data)
                rank = round(model.predict(feats.view(1, -1).numpy())[0])
                print(f'Using 4D model, returned: {rank}')
                self.perform_approx(self.features[lidx], rank=rank)
                approx_ranks.append(rank)
            else:
                raise ValueError(f'Model: {model}, ranks: {ranks}; both cannot be None')

        return approx_ranks

    def save_layer_weights(self, save_dir, layer_idxs=[10]):
        for lidx in layer_idxs:
            fn = path.join(save_dir, f'trained_alexnet_layer-{lidx}.mat')
            savemat(fn, { 'data': self.features[lidx].weight.data.clone().detach().cpu().numpy() })


class LitAlexNet(LightningModule):
    def __init__(self, num_classes=10, rank=10):
        super().__init__()
        self.model = AlexNet(num_classes=num_classes, rank=rank)
        self.save_hyperparameters()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def approx_layers(self, **kwargs):
        return self.model.approx_layer_data(**kwargs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler_dict = {
            "scheduler": optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5),
            "interval": "epoch"
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
