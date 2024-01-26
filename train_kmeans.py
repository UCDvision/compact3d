# Modification of code from Original 3D Gaussian Splat repo

# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


# Apply k-Means based vector quantization to color and covariance parameters

import os
import sys
import pdb
from os.path import join
import datetime
import json
import time
from bitarray import bitarray

import numpy as np
import torch
from random import randint

from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.kmeans_quantize import Quantize_kMeans
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    num_gaussians_per_iter = []

    # k-Means quantization
    quantized_params = args.quant_params
    n_cls = args.kmeans_ncls
    n_cls_sh = args.kmeans_ncls_sh
    n_cls_dc = args.kmeans_ncls_dc
    n_it = args.kmeans_iters
    kmeans_st_iter = args.kmeans_st_iter
    freq_cls_assn = args.kmeans_freq
    if 'pos' in quantized_params:
        kmeans_pos_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'dc' in quantized_params:
        kmeans_dc_q = Quantize_kMeans(num_clusters=n_cls_dc, num_iters=n_it)
    if 'sh' in quantized_params:
        kmeans_sh_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)
    if 'scale' in quantized_params:
        kmeans_sc_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'rot' in quantized_params:
        kmeans_rot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'scale_rot' in quantized_params:
        kmeans_scrot_q = Quantize_kMeans(num_clusters=n_cls, num_iters=n_it)
    if 'sh_dc' in quantized_params:
        kmeans_shdc_q = Quantize_kMeans(num_clusters=n_cls_sh, num_iters=n_it)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()

        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if iteration > 3100:
            freq_cls_assn = 100
            if iteration > (opt.iterations - 5000):
                freq_cls_assn = 5000

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # quantize params
        if iteration > kmeans_st_iter:
            if iteration % freq_cls_assn == 1:
                assign = True
            else:
                assign = False
            if 'pos' in quantized_params:
                kmeans_pos_q.forward_pos(gaussians, assign=assign)
            if 'dc' in quantized_params:
                kmeans_dc_q.forward_dc(gaussians, assign=assign)
            if 'sh' in quantized_params:
                kmeans_sh_q.forward_frest(gaussians, assign=assign)
            if 'scale' in quantized_params:
                kmeans_sc_q.forward_scale(gaussians, assign=assign)
            if 'rot' in quantized_params:
                kmeans_rot_q.forward_rot(gaussians, assign=assign)
            if 'scale_rot' in quantized_params:
                kmeans_scrot_q.forward_scale_rot(gaussians, assign=assign)
            if 'sh_dc' in quantized_params:
                kmeans_shdc_q.forward_dcfrest(gaussians, assign=assign)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            psnr_train, psnr_test = psnr['train'], psnr['test']

            if (iteration in saving_iterations):
                print(args.model_path)
                # print(f'PSNR Train: {psnr_train}, PSNR Test: {psnr_test}')
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Save only the non-quantized parameters in ply file.
                all_attributes = {'xyz': 'xyz', 'dc': 'f_dc', 'sh': 'f_rest', 'opacities': 'opacities',
                                  'scale': 'scale', 'rot': 'rotation'}
                save_attributes = [val for (key, val) in all_attributes.items() if key not in quantized_params]
                if iteration > kmeans_st_iter:
                    scene.save(iteration, save_q=quantized_params, save_attributes=save_attributes)
                    
                    # Save indices and codebook for quantized parameters
                    kmeans_dict = {'rot': kmeans_rot_q, 'scale': kmeans_sc_q, 'sh': kmeans_sh_q, 'dc': kmeans_dc_q}
                    kmeans_list = []
                    for param in quantized_params:
                        kmeans_list.append(kmeans_dict[param])
                    out_dir = join(scene.model_path, 'point_cloud/iteration_%d' % iteration)
                    save_kmeans(kmeans_list, quantized_params, out_dir)
                else:
                    scene.save(iteration, save_q=[])

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        num_gaussians_per_iter.append(gaussians.get_xyz.shape[0])

    print("Number of Gaussians at the end: ", gaussians._xyz.shape[0])
    np.save(f'{scene.model_path}/num_g_per_iters.npy', np.array(num_gaussians_per_iter))


def dec2binary(x, n_bits=None):
    """Convert decimal integer x to binary.

    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    if n_bits is None:
        n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
    mask = 2**torch.arange(n_bits-1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)


def save_kmeans(kmeans_list, quantized_params, out_dir):
    """Save the codebook and indices of KMeans.

    """
    # Convert to bitarray object to save compressed version
    # saving as npy or pth will use 8bits per digit (or boolean) for the indices
    # Convert to binary, concat the indices for all params and save.
    bitarray_all = bitarray([])
    for kmeans in kmeans_list:
        n_bits = int(np.ceil(np.log2(len(kmeans.cls_ids))))
        assignments = dec2binary(kmeans.cls_ids, n_bits)
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        bitarray_all.extend(bitarr)
    with open(join(out_dir, 'kmeans_inds.bin'), 'wb') as file:
        bitarray_all.tofile(file)

    # Save details needed for loading
    args_dict = {}
    args_dict['params'] = quantized_params
    args_dict['n_bits'] = n_bits
    args_dict['total_len'] = len(bitarray_all)
    np.save(join(out_dir, 'kmeans_args.npy'), args_dict)
    centers_dict = {param: kmeans.centers for (kmeans, param) in zip(kmeans_list, quantized_params)}

    # Save codebook
    torch.save(centers_dict, join(out_dir, 'kmeans_centers.pth'))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # psnr_test = -1.
    psnr_out = {'train': -1., 'test': -1}
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                psnr_out[config['name']] = psnr_test
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return psnr_out


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 15_000, 20_000,
                                                                           25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--total_iterations', type=int, default=30000,
                        help='Total iterations of training')

    # Compress3D parameters
    parser.add_argument('--kmeans_st_iter', type=int, default=30000,
                        help='Start k-Means based vector quantization from this iteration')
    parser.add_argument('--kmeans_ncls', type=int, default=4096,
                        help='Number of clusters in k-Means quantization')
    parser.add_argument('--kmeans_ncls_sh', type=int, default=4096,
                        help='Number of clusters in k-Means quantization of spherical harmonics')
    parser.add_argument('--kmeans_ncls_dc', type=int, default=4096,
                        help='Number of clusters in k-Means quantization of DC component of color')
    parser.add_argument('--kmeans_iters', type=int, default=1,
                        help='Number of assignment and centroid calculation iterations in k-Means')
    parser.add_argument('--kmeans_freq', type=int, default=100,
                        help='Frequency of cluster assignment in k-Means')
    parser.add_argument('--grad_thresh', type=float, default=0.0002,
                        help='threshold on xyz gradients for densification')
    parser.add_argument("--quant_params", nargs="+", type=str, default=['sh', 'dc', 'scale', 'rot'])

    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.test_iterations = list(np.arange(0, args.total_iterations + 1, 100))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    outfile = join(args.model_path, 'train_args.json')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as fp:
        json.dump(vars(args), fp, indent=4, default=str)
    print('Quantized Params: ', args.quant_params)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
