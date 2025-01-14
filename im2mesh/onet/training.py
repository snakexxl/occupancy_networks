import os

import PIL
import numpy as np
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.data.preprocessing.constant import IMAGE_SIZE
from im2mesh.data.preprocessing.plot_keypoints_on_image import plot_keypoints_on_image
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from PIL import Image, ImageOps

import os


def createImagePoints(numberOfPoints):
    xvalues = np.linspace(0, 1000, IMAGE_SIZE)
    yvalues = np.linspace(0, 1000, IMAGE_SIZE)
    xx, yy = np.meshgrid(xvalues, yvalues)
    xx = xx.flatten()
    yy = yy.flatten()
    p = np.stack([xx, yy], axis=1).astype(np.float32)
    return p


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = tensor.transpose()
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_for_one_image(self, data,y):
        self.model.eval()

        device = self.device
        #p = data.get('points').to(device)
        p = createImagePoints(IMAGE_SIZE*IMAGE_SIZE)
        p = torch.from_numpy(p)
        p = p.reshape(1, (IMAGE_SIZE*IMAGE_SIZE), 2).to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        original_silhouette= data.get('original_silhouette', torch.empty(p.size(0), 0)).to(device)
        kwargs = {}
        with torch.no_grad():
            p_r = self.model.compute_elbo_bild(
                p, inputs, **kwargs)
        #original Bild und reconstruction bild speichern
        bild_matrix = p_r.probs
        bild_matrix = bild_matrix.reshape((IMAGE_SIZE,IMAGE_SIZE))
        
        mean_occ = float(torch.mean(bild_matrix).cpu().numpy())
        print(f"mean occ: {mean_occ}")

        original_silhouette = original_silhouette.reshape((IMAGE_SIZE, IMAGE_SIZE))
        bild_silhouette_sideways = tensor_to_image(original_silhouette.cpu())
        bild_silhouette = bild_silhouette_sideways.transpose(Image.ROTATE_270)
        bild_silhouette = ImageOps.mirror(bild_silhouette)
        bild = tensor_to_image(bild_matrix.cpu())
        image_path=f"/home/johannesselbert/Documents/GitHub/occupancy_networks/out/silhouette"
        bild.save(f"{image_path}/silhouette{y}.png")
        bild_silhouette.save(f"{image_path}/originalsilhouette{y}.png")
        #keypoints aufs Bild bringen und speichern
        image_path = f"/home/johannesselbert/Documents/GitHub/occupancy_networks/out/silhouette/originalsilhouette{y}.png"


        poseXY = inputs.reshape(32, 2)
        poseY = poseXY[:, 1]
        poseX = poseXY[:, 0]
        plot_keypoints_on_image(image_path, poseX, poseY)
        print(data.get('idx', torch.empty(p.size(0), 0)).to(device))
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()


        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if False and voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        # muss evtl shape verändern und den rest dann auch
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, c, **kwargs).logits
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss
