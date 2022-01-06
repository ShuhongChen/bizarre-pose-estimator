


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d

import _util.keypoints_v0 as util_keypoints
import _util.helper_models_v0 as util_hm


class Model(pl.LightningModule):
    def __init__(self, bargs, pargs, largs, margs):
        super().__init__()
        self.hparams.bargs = bargs
        self.hparams.pargs = pargs
        self.hparams.largs = largs
        self.hparams.margs = margs
        self.save_hyperparameters()
        
        # set up frozen pretrained networks
        self.resnet = ResnetFeatureExtractor(margs.resnet_inferserve_query).eval()
        self.rcnn = PretrainedKeypointDetector().eval()
        for param in [*self.resnet.parameters(), *self.rcnn.parameters()]:
            param.requires_grad = False
        
        # set up keypoint detector head
        self.size = margs.size
        self.resizer = TT.Resize(self.size)
        self.keypoint_head = nn.ModuleDict({
            'converter_resnet': ResnetFeatureConverter(self.size),
            'converter_rcnn': nn.Conv2d(17, 32, kernel_size=1, padding=0),
            'head': nn.Sequential(
                nn.Conv2d(128+32+3, 128, kernel_size=3, padding=1, padding_mode='replicate'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                util_hm.ResBlock(3, 64, 3, channels_in=128),
                util_hm.ResBlock(3, 32, 3, channels_in=64),
                nn.Conv2d(32, 17+8, kernel_size=3, padding=1, padding_mode='replicate'),
            ),
        })
        self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        return
    def loss(self, gt, pred, return_more=False):
        # expects gt/pred is {keypoint_heatmaps: (bs,17+8,h,w)}
        # expects pred heatmaps in logits (pre-sigmoid)
        # expects keypoints in dict if return_more
        # if keypoint_loss_weights in gt, will apply
        ans = {}
        if 'keypoint_heatmaps' in gt and 'keypoint_heatmaps' in pred:
            bce = self.bce_logits(pred['keypoint_heatmaps'], self.resizer(gt['keypoint_heatmaps']))
            wt = 1 if 'keypoint_loss_weights' not in gt else gt['keypoint_loss_weights'][:,:,None,None]
            ans['loss'] = wt * bce
        if return_more:
            with torch.no_grad():
                q = gt['keypoints'][:,:17,:2], gt['bbox'], pred['keypoints'][:,:17,:2]
                mpl = gt['mean_part_lengths']
                ans['pcp_50'] = util_keypoints.pcp(*q, k=0.5)['correct_parts']
                ans['pcpm_50'] = util_keypoints.pcp(*q, k=0.5, m=mpl)['correct_parts']
                ans['pckh_20'] = util_keypoints.pck(*q, k=0.2, mode='h')['correct_joints']
                ans['pckh_50'] = util_keypoints.pck(*q, k=0.5, mode='h')['correct_joints']
                ans['pdj_20'] = util_keypoints.pdj(*q, k=0.2)['correct_joints']
                ans['oks_50'] = util_keypoints.oks(*q, thresh=0.5)['correct_joints']
                ans['oks_75'] = util_keypoints.oks(*q, thresh=0.75)['correct_joints']
        return ans
    def forward(self, x, smoothing=None, return_more=False):
        self.rcnn.eval()
        self.resnet.eval()
        with torch.no_grad():
            feats_resnet = self.resnet(x)
            feats_rcnn = self.rcnn(x, return_more=return_more)
        feats = torch.cat([
            self.resizer(x),
            self.resizer(self.keypoint_head['converter_resnet'](feats_resnet)),
            self.resizer(self.keypoint_head['converter_rcnn'](feats_rcnn['keypoint_heatmaps'])),
        ], dim=1)
        hms = self.keypoint_head['head'](feats)
        ans = {'keypoint_heatmaps': hms}
        if return_more:
            ans['features_resnet'] = feats_resnet
            ans['features_rcnn'] = feats_rcnn
            if smoothing is None:
                ans['keypoint_heatmaps_prob'] = torch.sigmoid(hms)
                kps = hms.view(hms.shape[0], hms.shape[1], -1).argmax(-1)
                kps = torch.stack([
                    kps // hms.shape[2] * (x.shape[2]/hms.shape[2]),
                    kps % hms.shape[3] * (x.shape[3]/hms.shape[3]),
                ], dim=2)
            else:
                hmp = torch.sigmoid(hms)
                ksig = max(hmp.shape[-2:]) * smoothing
                kern = max(3, int(ksig)*2+1)
                hmps = kornia.filters.gaussian_blur2d(
                    hmp,
                    kernel_size=(kern,kern),
                    sigma=(ksig,ksig),
                    border_type='reflect',
                )
                kps = hmps.view(hmps.shape[0], hmps.shape[1], -1).argmax(-1)
                kps = torch.stack([
                    kps // hmps.shape[2] * (x.shape[2]/hmps.shape[2]),
                    kps % hmps.shape[3] * (x.shape[3]/hmps.shape[3]),
                ], dim=2)
                ans['keypoint_heatmaps_prob'] = hmps
            ans['keypoints'] = kps
        return ans
    def training_step(self, batch, batch_idx):
        pred = self.forward(batch['image'], return_more=False)
        loss = self.loss(batch, pred, return_more=False)
        loss_agg = loss['loss'].mean()
        self.log('train_loss', loss_agg, sync_dist=True)
        return {
            'loss': loss_agg,
        }
    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch['image'], return_more=True)
        loss = self.loss(batch, pred, return_more=True)
        # for k,v in loss.items():
        #     self.log(f'val_{k}', v.float().mean(), sync_dist=True)
        # return
        ans = {
            f'val_{met}': v.float().mean()
            for met,v in loss.items()
        }
        self.log('val_loss', ans['val_loss'], sync_dist=True)
        return ans
    def validation_epoch_end(self, outputs):
        for met in outputs[0]:
            self.log(met, torch.stack([o[met] for o in outputs]).mean(), sync_dist=True)
        return
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.keypoint_head.parameters(),
            lr=self.hparams.margs.lr,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.margs.max_epochs,
            eta_min=0,
        )
        return [opt,], [sched,]


######################## HELPER SUBMODULES ########################

class ResnetFeatureConverter(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.resizer = TT.Resize(self.size)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.layer1 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.layer2 = nn.Conv2d(512, 64, kernel_size=1, padding=0)
        self.layer3 = nn.Conv2d(1024, 64, kernel_size=1, padding=0)
        # self.layer4 = nn.Conv2d(2048, 128, kernel_size=1, padding=0)
        self.head = nn.Conv2d(64*4, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm2d(64*4)
        return
    def forward(self, feats_resnet):
        return self.head(self.relu(self.batchnorm(torch.cat([
            self.resizer(self.conv1(feats_resnet['conv1'])),
            self.resizer(self.layer1(feats_resnet['layer1'])),
            self.resizer(self.layer2(feats_resnet['layer2'])),
            self.resizer(self.layer3(feats_resnet['layer3'])),
            # self.resizer(self.layer4(feats_resnet['layer4'])),
        ], dim=1))))

# from hack.train_classifier.kate import Model as Classifier
from _train.danbooru_tagger.models.kate import Model as Classifier
class ResnetFeatureExtractor(nn.Module):
    def __init__(self, inferserve_query):
        super().__init__()
        self.inferserve_query = inferserve_query
        if self.inferserve_query=='torchvision':
            # use pytorch pretrained resnet50
            self.inferserve = None
            self.base_hparams = None
            resnet = tv.models.resnet50(pretrained=True)

            self.resize = TT.Resize(256)
            self.resnet_preprocess = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu      #   64ch, 128p (assuming 256p input)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  #  256ch,  64p
            self.layer2 = resnet.layer2  #  512ch,  32p
            self.layer3 = resnet.layer3  # 1024ch,  16p
            # self.layer4 = resnet.layer4  # 2048ch,   8p
        elif self.inferserve_query=='rf5':
            # use rf5 resnet50
            self.inferserve = None
            self.base_hparams = None
            resnet = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')

            self.resize = TT.Resize(256)
            self.resnet_preprocess = TT.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979])
            self.conv1 = resnet[0][0]
            self.bn1 = resnet[0][1]
            self.relu = resnet[0][2]      #   64ch, 128p (assuming 256p input)
            self.maxpool = resnet[0][3]
            self.layer1 = resnet[0][4]  #  256ch,  64p
            self.layer2 = resnet[0][5]  #  512ch,  32p
            self.layer3 = resnet[0][6]  # 1024ch,  16p
            # self.layer4 = resnet[0][7]  # 2048ch,   8p
        else:
            # use pretrained kate, danbooru-specific
            # self.inferserve = util_serve.infer_ckpt(self.inferserve_query)
            # base = Classifier.load_from_checkpoint(self.inferserve['fn'])
            self.inferserve = None
            base = Classifier.load_from_checkpoint(
                './_train/danbooru_tagger/runs/waning_kate_vulcan0001/checkpoints/'
                'epoch=0022-val_f2=0.4461-val_loss=0.0766.ckpt'
            )
            self.base_hparams = base.hparams
            
            self.resize = TT.Resize(base.hparams.largs.danbooru_sfw.size)
            self.resnet_preprocess = base.resnet_preprocess
            self.conv1 = base.resnet.conv1
            self.bn1 = base.resnet.bn1
            self.relu = base.resnet.relu      #   64ch, 128p (assuming 256p input)
            self.maxpool = base.resnet.maxpool
            self.layer1 = base.resnet.layer1  #  256ch,  64p
            self.layer2 = base.resnet.layer2  #  512ch,  32p
            self.layer3 = base.resnet.layer3  # 1024ch,  16p
            # self.layer4 = base.resnet.layer4  # 2048ch,   8p
        return
    def forward(self, x):
        ans = {}
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ans['conv1'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        ans['layer1'] = x
        x = self.layer2(x)
        ans['layer2'] = x
        x = self.layer3(x)
        ans['layer3'] = x
        # x = self.layer4(x)
        # ans['layer4'] = x
        return ans

class PretrainedKeypointDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # setup rcnn model
        self.cfg = detectron2.config.get_cfg()
        self.cfg.merge_from_file(detectron2.model_zoo.get_config_file(
            'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
        ))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        self.cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(
            'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
        )
        self.cfg['MODEL']['DEVICE'] = 'cpu'
        self.predictor = detectron2.engine.DefaultPredictor(self.cfg)
        self.model = self.predictor.model
        
        # preprocessing
        self.size = 800
        self.resize = TT.Resize(self.size)
        return
    def forward(self, img, return_more=False):
        # assumes img.shape = (bs, rgb, h, w)
        h,w = img.shape[2:]
        x = [{'image': i, 'height': h, 'width': w} for i in 255*self.resize(img).flip(1)]
        images = self.model.preprocess_image(x)
        features = self.model.backbone(images.tensor)
        if False:
            # roi heads will fuckin re-predict bboxes
            proposals = [
                detectron2.structures.instances.Instances(
                    image_size=images[0].shape[1:],
                    proposal_boxes=detectron2.structures.boxes.Boxes(torch.tensor([
                        0, 0, *images[0].shape[1:],
                    ], device=images.device)[None]),
                    objectness_logits=torch.tensor([1.0,], device=images.device),
                ),
            ]*img.shape[0]
            results,_ = self.model.roi_heads(images, features, proposals, None)
        else:
            # forces them to use my bboxes
            h,w = images[0].shape[1:]
            detected_instances = [
                detectron2.structures.instances.Instances(
                    image_size=(h,w),
                    pred_boxes=detectron2.structures.boxes.Boxes(torch.tensor([
                        0, 0, h, w,
                    ], device=images.device)[None]),
                    pred_classes=torch.tensor([0,], device=images.device),
                )
                for _ in range(img.shape[0])
            ]
            results = self.model.roi_heads.forward_with_given_boxes(features, detected_instances)
        results = self.model._postprocess(results, x, images.image_sizes)
        hms = torch.cat([r['instances'].pred_keypoint_heatmaps for r in results])
        ans = {'keypoint_heatmaps': hms}
        if return_more:
            ans['results'] = results
            ans['images'] = images
            ans['features'] = features
            ans['detected_instances'] = detected_instances
            ans['bboxes'] = [
                ((b,a), (d-b,c-a))
                for r in results
                for a,b,c,d in r['instances'].pred_boxes.tensor.detach().cpu().numpy()
            ]
            ans['keypoints'] = torch.stack([
                k[:,:2].flip(1)
                for r in results
                for k in r['instances'].pred_keypoints
            ])
        return ans



