# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Arnheim 3 - Collage
# Piotr Mirowski, Dylan Banarse, Mateusz Malinowski, Yotam Doron, Oriol Vinyals,
# Simon Osindero, Chrisantha Fernando
# DeepMind, 2021-2022

# Command-line version of the Google Colab code available at:
# https://github.com/deepmind/arnheim/blob/main/arnheim_3.ipynb

# Colour and affine transform classes.

from kornia.color import hsv

import numpy as np
import torch
import torch.nn.functional as F


class PopulationAffineTransforms():
  """Population-based Affine Transform operations."""

  def __init__(self, other, idx_from, num_patches=1):

    self.translation = other.translation[idx_from, ...].detach().cpu().numpy()
    self.rotation = other.rotation[idx_from, ...].detach().cpu().numpy()
    self.scale = other.scale[idx_from, ...].detach().cpu().numpy()
    self.squeeze = other.squeeze[idx_from, ...].detach().cpu().numpy()
    self.shear = other.shear[idx_from, ...].detach().cpu().numpy()
    self._num_patches = num_patches

    identity = np.ones((num_patches, 1, 1)) * np.eye(2)
    zero_column = np.zeros((num_patches, 2, 1))
    unit_row = np.ones((num_patches, 1, 1)) * np.array([0., 0., 1.])
    zeros = np.zeros((num_patches, 1, 1))
    self.scale_affine_mat = np.concatenate([
        np.concatenate([self.scale, self.shear], 2)
        np.concatenate([zeros, self.scale * self.squeeze], 2)],
        1)
    self.scale_affine_mat = np.concatenate([
        np.concatenate([self.scale_affine_mat, zero_column], 2),
        unit_row], 1)
    self.rotation_affine_mat = torch.cat([
        torch.cat([torch.cos(self.rotation), -torch.sin(self.rotation)], 3),
        torch.cat([torch.sin(self.rotation), torch.cos(self.rotation)], 3)],
        2)
    rotation_affine_mat = torch.cat([
        torch.cat([rotation_affine_mat, self._zero_column], 3),
        self._unit_row], 2)

  def forward(self, x):
    scale_affine_mat = torch.cat([
        torch.cat([self.scale, self.shear], 3),
        torch.cat([self._zeros, self.scale * self.squeeze], 3)],
        2)
    scale_affine_mat = torch.cat([
        torch.cat([scale_affine_mat, self._zero_column], 3),
        self._unit_row], 2)
    rotation_affine_mat = torch.cat([
        torch.cat([torch.cos(self.rotation), -torch.sin(self.rotation)], 3),
        torch.cat([torch.sin(self.rotation), torch.cos(self.rotation)], 3)],
        2)
    rotation_affine_mat = torch.cat([
        torch.cat([rotation_affine_mat, self._zero_column], 3),
        self._unit_row], 2)

    scale_rotation_mat = torch.matmul(scale_affine_mat,
                                      rotation_affine_mat)[:, :, :2, :]
    # Population and patch dimensions (0 and 1) need to be merged.
    # E.g. from (POP_SIZE, NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
    # to (POP_SIZE * NUM_PATCHES, CHANNELS, WIDTH, HEIGHT)
    scale_rotation_mat = scale_rotation_mat[:, :, :2, :].view(
        1, -1, *(scale_rotation_mat[:, :, :2, :].size()[2:])).squeeze()
    x = x.view(1, -1, *(x.size()[2:])).squeeze()
    scaled_rotated_grid = F.affine_grid(
        scale_rotation_mat, x.size(), align_corners=True)
    scaled_rotated_x = F.grid_sample(x, scaled_rotated_grid, align_corners=True)

    translation_affine_mat = torch.cat([self._identity, self.translation], 3)
    translation_affine_mat = translation_affine_mat.view(
        1, -1, *(translation_affine_mat.size()[2:])).squeeze()
    translated_grid = F.affine_grid(
        translation_affine_mat, x.size(), align_corners=True)
    y = F.grid_sample(scaled_rotated_x, translated_grid, align_corners=True)
    return y.view(self._pop_size, self.config["num_patches"], *(y.size()[1:]))


class HighResOrderOnlyTransforms():
  """No color transforms, just ordering of patches."""

  def __init__(self, other, idx_from, num_patches=1):

    self.orders = other.orders[idx_from, ...].detach().cpu().numpy()
    self._zeros = np.ones((num_patches, 1, 1, 1))

  def forward(self, x):
    colours = np.concatenate(
        [self._zeros, self._zeros, self._zeros, self._zeros, self.orders], 1)
    return colours * x


class HighResColourHSVTransforms():
  """HSV color transforms and ordering of patches."""

  def __init__(self, other, idx_from, num_patches=1):

    self.hues = other.hues[idx_from, ...].detach().cpu().numpy()
    self.saturations = other.saturations[idx_from, ...].detach().cpu().numpy()
    self.values = other.values[idx_from, ...].detach().cpu().numpy()
    self.orders = other.orders[idx_from, ...].detach().cpu().numpy()
    self._zeros = np.ones((num_patches, 1, 1, 1))
    self._hsv_to_rgb = hsv.HsvToRgb()

  def forward(self, image):
    colours = np.concatenate(
        [self.hues, self.saturations, self.values, self._zeros, self.orders], 1)
    hsv_image = colours * image
    rgb_image = self._hsv_to_rgb(hsv_image[:, :3, :, :])
    return np.concatenate([rgb_image, hsv_image[:, 3:, :, :]], 1)


class HighResColourRGBTransforms():
  """RGB color transforms and ordering of patches."""

  def __init__(self, other, idx_from, num_patches=1):

    self.reds = other.reds[idx_from, ...].detach().cpu().numpy()
    self.greens = other.greens[idx_from, ...].detach().cpu().numpy()
    self.blues = other.blues[idx_from, ...].detach().cpu().numpy()
    self.orders = other.orders[idx_from, ...].detach().cpu().numpy()
    self._zeros = np.ones((num_patches, 1, 1, 1))

  def forward(self, x):
    colours = np.concatenate(
        [self.reds, self.greens, self.blues, self._zeros, self.orders], 1)
    return colours * x
