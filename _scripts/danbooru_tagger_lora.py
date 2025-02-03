



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d


parser = argparse.ArgumentParser()
parser.add_argument('dn_root')
parser.add_argument('--formats', nargs='+', default=['jpg', 'png', 'jpeg', 'tga'], help='acceptable image formats')
parser.add_argument('--recursive', action='store_false', help='recursively search directories for images')
parser.add_argument('--mode', choices=['overwrite', 'append', 'skip'], default='overwrite', help='mode of operation')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for saving tags, higher is tighter')
parser.add_argument('--extension', default='txt', help='extension for saving tags')
parser.add_argument('--prefix', type=str, default='', help='prefix text')
parser.add_argument('--keep_underscore', action='store_true', help='remove underscore from tags')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from _train.danbooru_tagger.models.kate import Model as DanbooruTagger
model = DanbooruTagger.load_from_checkpoint(
    './_train/danbooru_tagger/runs/waning_kate_vulcan0001/checkpoints/'
    'epoch=0022-val_f2=0.4461-val_loss=0.0766.ckpt'
).to(device)

def infer(self, images, return_more=True):
    # images: list of images
    imgs = torch.cat([
        I(img)
            .resize_square(self.hparams.largs.danbooru_sfw.size)
            .alpha_bg(c='w')
            .convert('RGB')
            .tensor()[None,]
        for img in images
    ]).to(self.device)
    self.eval()
    with torch.no_grad():
        out = self.forward(imgs, return_more=False)
    if return_more:
        out['prob_dict'] = [
            {
                r['name']: p
                for r,p in zip(self.rules, x)
            }
            for x in torch.sigmoid(out['raw']).cpu().numpy()
        ]
    return out

# img = I(args.fn_img)
# ans = infer(model, [img,])

# for k,v in sorted(ans['prob_dict'][0].items(), key=lambda x: -x[1]):
#     if v>=0.5:
#         print(v,k)

prefixlist = [] if args.prefix=='' else [args.prefix,]

for p,d,f in (os.walk(args.dn_root) if args.recursive else [args.dn_root, [], os.listdir(args.dn_root)]):
    for fn in f:
        # check file format
        if not any(fn.endswith(f'.{fmt}') for fmt in args.formats):
            continue
        fn_full = f'{p}/{fn}'
        fn_out = f'{p}/{fnstrip(fn_full)}.{args.extension}'
        already_exists = os.path.exists(fn_out)
        print(fn_full)

        # check if already finished
        if args.mode=='skip' and already_exists:
            print('- skip: already exists')
            continue

        # run inference
        img = I(fn_full)
        ans = infer(model, [img,])
        outs = []
        for k,v in sorted(ans['prob_dict'][0].items(), key=lambda x: -x[1]):
            if v>=args.threshold:
                outs.append(k)

        # save result
        out_txt = prefixlist + (outs if args.keep_underscore else [x.replace('_', ' ') for x in outs])
        if not already_exists:
            write(', '.join(out_txt), fn_out)
        else:
            if args.mode=='overwrite':
                write(', '.join(out_txt), fn_out)
            elif args.mode=='skip':
                print('- error: should not have skipped')
            elif args.mode=='append':
                write(', '.join([read(fn_out),] + out_txt), fn_out)
            else:
                assert 0, f'args.mode {args.mode} not understood'

        # break





