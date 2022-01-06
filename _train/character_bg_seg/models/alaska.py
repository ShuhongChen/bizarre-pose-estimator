

from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d


class Model(pl.LightningModule):
    def __init__(self, bargs, pargs, largs, margs):
        super().__init__()
        self.hparams.bargs = bargs
        self.hparams.pargs = pargs
        self.hparams.largs = largs
        self.hparams.margs = margs
        self.save_hyperparameters()

        # setup deeplab
        self.deeplab = tv.models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True,
        )
        self.deeplab.aux_classifier = None
        # for param in self.deeplab.backbone.parameters():
        #     param.requires_grad = False
        self.deeplab.classifier = nn.Sequential(
            # tv.models.segmentation.deeplabv3.DeepLabHead(2048, 2)[0],
            tv.models.segmentation.deeplabv3.ASPP(2048, [12, 24, 36]),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )
        self.final_head = nn.Sequential(
            nn.Conv2d(16+3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 2, kernel_size=1, stride=1),
        )
        self.deeplab_preprocess = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # loss
        self.cel = nn.CrossEntropyLoss(reduction='none')
        return
    def loss(self, gt, pred, reduce=True):
        # expects gt is long of class labels
        # expects pred is pre-softmax values
        cel = self.cel(pred, gt).mean((1,2))
        with torch.no_grad():
            bing,binp = gt>0.5, pred[:,1]>pred[:,0]
            tp = (bing&binp).float().mean((1,2))
            fn = (bing&~binp).float().mean((1,2))
            fp = (~bing&binp).float().mean((1,2))
            tn = (~bing&~binp).float().mean((1,2))
            f1 = 2*tp/(2*tp+fn+fp) ; f1[f1!=f1] = 1.0
            acc = tp + tn
            rec = tp / (tp+fn) ; rec[rec!=rec] = 0.0
            pre = tp / (tp+fp) ; pre[pre!=pre] = 0.0
        return {
            'loss': cel.mean() if reduce else cel,
            'f1': f1.mean() if reduce else f1,
            'acc': acc.mean() if reduce else acc,
            'rec': rec.mean() if reduce else rec,
            'pre': pre.mean() if reduce else pre,
        }
    def forward(self, rgb, return_more=True):
        # preprocess
        normed = self.deeplab_preprocess(rgb)
        stackin = normed

        # forward pass
        out_dl = self.deeplab(normed)['out']
        out_fin = self.final_head(torch.cat([
            out_dl, stackin,
        ], dim=1))
        out = {'raw': out_fin}
        if return_more:
            out['softmax'] = torch.softmax(out_fin, dim=1)
            out['max'] = torch.max(out_fin, dim=1).indices
        return out
    
    def training_step(self, batch, batch_idx):
        # unpack
        x = batch
        rgb = x['image_composite']
        seg = (x['image_fg'][:,-1]>0.5).long()
        
        # predict
        pred = self.forward(rgb, return_more=False)
        loss = self.loss(seg, pred['raw'])
        
        # log
        for k,v in {
            'train_loss': loss['loss'],
            'train_f1': loss['f1'],
            'train_acc': loss['acc'],
            'train_rec': loss['rec'],
            'train_pre': loss['pre'],
        }.items():
            self.log(k, v)
        return {
            'loss': loss['loss'],
        }
    def validation_step(self, batch, batch_idx):
        # unpack
        x = batch
        rgb = x['image_composite']
        seg = (x['image_fg'][:,-1]>0.5).long()
        
        # predict
        pred = self.forward(rgb, return_more=False)
        loss = self.loss(seg, pred['raw'])
        
        # log
        return OrderedDict({
            'val_loss': loss['loss'],
            'val_f1': loss['f1'],
            'val_acc': loss['acc'],
            'val_rec': loss['rec'],
            'val_pre': loss['pre'],
        })
    def validation_epoch_end(self, outputs):
        labels = [l for l in outputs[0].keys() if l.startswith('val_')]
        ans = {l: 0 for l in labels}
        for output in outputs:
            for label in labels:
                ans[label] += output[label]
        for label in labels:
            ans[label] /= len(outputs)
        for k,v in ans.items():
            self.log(k, v)
        return
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.margs.lr,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=100,
            eta_min=0,
        )
        return [opt,], [sched,]
