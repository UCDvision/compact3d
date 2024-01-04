# Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization

This Repository is an official implementation of "[Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization](https://arxiv.org/abs/2311.18159)".

## Overview

![](teaser_new.png)

Compact3D is a method to reduce the storage memory requirements of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) models. 3D Gaussian Splatting is a new technique for novel view synthesis where properties of 3D Gaussians (location, shape, color) are optimized to model a 3D scene. The method performs better than SOTA NeRF approaches, is extremely fast to train and can be rendered in real time during inference. However, since a typical scene requires millions of Gaussians to model it, the memory requirements can be an order of magnitude more than many NeRF approaches. Here, we reduce the size of the trained 3D Gaussian Splat models by 10-20x by vector quantizing the Gaussian parameters. An overview of our method is shown below. We perform K-Means quantization on the covariance and color parameters of all Gaussians and replace values of each with the corresponding entry in the codebook (i.e., the cluster center). This is done in conjuncion with the training of the parameter values as done in the non-quantized version of Gaussian splatting. We observe that the models can be compressed 20 times without a big drop in performance. 

## Updates
**[4 January 2024]:** We have updated the code to store and load the quantized parameters using the indices and codebook. Only the compressed version of the model is saved.  

## Getting Started 

Our code is based on the excellent official repo for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main). First, clone our repository. 
```shell
git clone https://github.com/UCDvision/compact3d
cd compact3d
```
Then, follow the instructions [here](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) to clone and install 3D Gaussian Splatting. You should now have a directory named ```gaussian-splatting``` in the ```compact3d``` folder. Next, move the downloaded files from our repo to the appropriate locations in the gaussian-splatting folder.
```shell
bash move_files_to_gsplat.sh
```

## Training

Modify the paths to dataset and output folder in the ```run.sh``` script.
```shell
cd gaussian-splatting
bash run.sh
```

## Rendering and Evaluation

Once the model is trained, the rendering and evaluation process is exactly the same as in 3D Gaussian Splatting. Following their instructions,
```shell
python render.py -m <path to trained model> --skip_train --load_quant # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
Use the ```skip_train``` argument to avoid rendering the images from the train set camera viewpoints and the ```load_quant``` argument when loading our quantized model. Remove the ```load_quant``` argument if evaluating the original non-qunatized model.  

## Citation

If you make use of the code, please cite the following work:
```
@article{navaneet2023compact3d,
  title={Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization},
  author={Navaneet, KL and Meibodi, Kossar Pourahmadi and Koohpayegani, Soroush Abbasi and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:2311.18159},
  year={2023}
}
```

## License

This project is under the MIT license.
