# Convert the quantized model to uncompressed ply format - useful for visualization using SIBR


# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def save2ply(dataset: ModelParams, iteration: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        # Load quantized model
        print('Loading quantized model...')
        gaussians.load_ply(
            os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"),
            load_quant=True
        )

        # Save non-quantized version
        point_cloud_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration))
        print('Saving non-quantized model to', os.path.join(point_cloud_path, "point_cloud_decompressed.ply"))
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_decompressed.ply"))
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_quant=True)
        # scene.save(iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Converting to ply: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    save2ply(model.extract(args), args.iteration)
