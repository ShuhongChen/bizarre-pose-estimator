



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d


parser = argparse.ArgumentParser()
parser.add_argument('fn_img')
args = parser.parse_args()


from _train.character_bg_seg.models.alaska import Model as CharacterBGSegmenter
model = CharacterBGSegmenter.load_from_checkpoint(
    './_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/'
    'epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt'
)

def abbox(img, thresh=0.5, allow_empty=False):
    # get bbox from alpha image, at threshold
    img = I(img).np()
    assert len(img) in [1,4], 'image must be mode L or RGBA'
    a = img[-1] > thresh
    xlim = np.any(a, axis=1).nonzero()[0]
    ylim = np.any(a, axis=0).nonzero()[0]
    if len(xlim)==0 and allow_empty: xlim = np.asarray([0, a.shape[0]])
    if len(ylim)==0 and allow_empty: ylim = np.asarray([0, a.shape[1]])
    axmin,axmax = max(int(xlim.min()-1),0), min(int(xlim.max()+1),a.shape[0])
    aymin,aymax = max(int(ylim.min()-1),0), min(int(ylim.max()+1),a.shape[1])
    return [(axmin,aymin), (axmax-axmin,aymax-aymin)]

def infer(self, images, bbox_thresh=0.5, return_more=True):
    anss = []
    _size = self.hparams.largs.bg_seg.size
    self.eval()
    for img in images:
        oimg = img
        # img = a2bg(resize_min(img, _size).convert('RGBA'),1).convert('RGB')
        img = I(img).resize_min(_size).convert('RGBA').alpha_bg(1).convert('RGB').pil()
        timg = TF.to_tensor(img)[None].to(self.device)
        with torch.no_grad():
            out = self(timg)
        ans = TF.to_pil_image(out['softmax'][0,1].float().cpu()).resize(oimg.size[::-1])
        ans = {'segmentation': I(ans)}
        ans['bbox'] = abbox(ans['segmentation'], thresh=bbox_thresh, allow_empty=True)
        anss.append(ans)
    return anss

img = I(args.fn_img)
ans = infer(model, [img,])

bbox = ans[0]['bbox']
print(f'bounding box\n\ttop-left: {bbox[0]}\n\tsize: {bbox[1]}')

seg = ans[0]['segmentation']
seg.rect(*bbox).save('./_samples/character_bg_seg.png')
print('output saved to ./_samples/character_bg_seg.png')

