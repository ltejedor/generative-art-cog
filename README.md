# Generative Art Using Neural Visual Grammars and Dual Encoders

## Arnheim 1

The original algorithm from the paper
[Generative Art Using Neural Visual Grammars and Dual Encoders](https://arxiv.org/abs/2105.00162)
running on 1 GPU allows optimization of any image using a genetic algorithm.
This is much more general but much slower than using Arnheim 2 which uses
gradients.

## Arnheim 2

A reimplementation of the Arnheim 1 generative architecture in the CLIPDraw
framework allowing optimization of its parameters using gradients. Much more
efficient than Arnheim 1 above but requires differentiating through the image
itself.

## Arnheim 3

A spatial transformer-based Arnheim implementation for generating collage images.
It employs a combination of evolution and training to create collages from
opaque to transparent image patches. Example patch datasets are provided under
[CC BY-SA 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/).
The 'Fruit and veg' patches are based on a subset of the
[Kaggle Fruits 360](https://www.kaggle.com/moltean/fruits) under the same
license.

## Usage

Usage instructions are included in the Colabs which open and run on the
free-to-use Google Colab platform - just click the buttons below! Improved
performance and longer timeouts are available with Colab Pro.

Arnheim 1 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/master/arnheim_1.ipynb)

Arnheim 2 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/master/arnheim_2.ipynb)

Arnheim 3 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/master/arnheim_3.ipynb)

Arnheim 3 Patch Maker [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/arnheim/blob/master/arnheim_3_patch_maker.ipynb)

## Citing this work

If you use this code (or any derived code), data or these models in your work,
please cite the relevant accompanying [paper](https://arxiv.org/abs/2105.00162).

```
@misc{fernando2021genart,
      title={Generative Art Using Neural Visual Grammars and Dual Encoders},
      author={Chrisantha Fernando and S. M. Ali Eslami and Jean-Baptiste Alayrac and Piotr Mirowski and Dylan Banarse and Simon Osindero}
      year={2021},
      eprint={2105.00162},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Disclaimer

This is not an official Google product.

CLIPDraw provided under license, Copyright 2021 Kevin Frans.

Other works may be copyright of the authors of such work.
