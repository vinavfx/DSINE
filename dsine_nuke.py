import torch
from models.dsine.v02 import DSINE_v02 as DSINE
import utils.utils as utils
import torch.nn.functional as F
from utils.projection import intrins_from_fov
from torchvision import transforms
from projects.dsine.config import get_args

checkpoint = './dsine.pt'
output_model = './nuke/Cattery/DSINE/DSINE.pt'


class dsine_nuke(torch.nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args(test=True)
        args.ckpt_path = checkpoint

        self.model = DSINE(args).to(torch.device('cpu'))
        self.model = utils.load_checkpoint(args.ckpt_path, self.model)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.iterations = 5
        self.fov = 60

    def forward(self, img):
        _, _, orig_H, orig_W = img.shape

        device = 'cuda' if img.is_cuda else 'cpu'
        lrtb = utils.get_padding(orig_H, orig_W)

        img = F.pad(img, lrtb, mode="constant", value=0.0)
        img = self.normalize(img)

        intrins = intrins_from_fov(
            new_fov=float(self.fov), H=orig_H, W=orig_W, device=device).unsqueeze(0)

        intrins[:, 0, 2] += lrtb[0]
        intrins[:, 1, 2] += lrtb[2]

        pred_norm = self.model(img, intrins, self.iterations)[-1]
        pred_norm = pred_norm[:, :, lrtb[2]
            :lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

        return pred_norm


nuke = torch.jit.script(dsine_nuke().eval())
nuke.save(output_model)
