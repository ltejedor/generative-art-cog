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

# Video utility functions, image rendering and display.

import cv2
import numpy as np
import torch


def cv2_imshow(img, name="CollageGenerator"):
  if img.dtype == np.float32 and img.max() > 1.:
    img = img.astype(np.uint8)
  cv2.imshow(name, img)
  cv2.waitKey(1)


def load_image(filename, as_cv2_image=False, show=False):
  """Load an image as [0,1] RGB numpy array or cv2 image format."""
  img = cv2.imread(filename)
  if show:
    cv2_imshow(img)
  if as_cv2_image:
    return img  # With colour format BGR
  img = np.asarray(img)
  return img[..., ::-1] / 255.  # Reverse colour dim to convert BGR to RGB


def layout_img_batch(img_batch, max_display=None):
  img_np = img_batch.transpose(0, 2, 1, 3).clip(0.0, 1.0)  # S, W, H, C
  if max_display:
    img_np = img_np[:max_display, ...]
  sp = img_np.shape
  img_np[:, 0, :, :] = 1.0  # White line separator
  img_stitch = np.reshape(img_np, (sp[1] * sp[0], sp[2], sp[3]))
  img_r = img_stitch.transpose(1, 0, 2)   # H, W, C
  return img_r


def show_and_save(img_batch, config, t=None,
                  max_display=1, interpolation="None", stitch=True,
                  img_format="SCHW", show=True, filename=None):
  """Display image.
  Args:
    img: image to display
    t: time step
    max_display: max number of images to display from population
    interpolation: interpolate enlarged images
    stitch: append images side-by-side
    img_format: SHWC or SCHW (the latter used by CLIP)
  Returns:
    stitched image or None
  """

  if isinstance(img_batch, torch.Tensor):
    img_np = img_batch.detach().cpu().numpy()
  else:
    img_np = img_batch

  if len(img_np.shape) == 3:
    # if not a batch make it one.
    img_np = np.expand_dims(img_np, axis=0)

  if not stitch:
    print(f"image (not stitch) min {img_np.min()}, max {img_np.max()}")
    for i in range(min(max_display, img_np.shape[0])):
      img = img_np[i]
      if img_format == "SCHW":  # Convert to SHWC
        img = np.transpose(img, (1, 2, 0))
      img = np.clip(img, 0.0, 1.0)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255
      if filename is not None:
        if img.shape[1] > config['canvas_width']:
          filename = "highres_" + filename
        output_dir = config["output_dir"]
        #filename = f"{output_dir}/{filename}_{str(i)}"
        if t is not None:
          filename += "_t_" + str(t)
        filename += ".png"
        print(f"Saving image {filename} (shape={img.shape})")
        cv2.imwrite(filename, img)
      if show:
        cv2_imshow(img)
    return None
  else:
    print(f"image (stitch) min {img_np.min()}, max {img_np.max()}")
    img_np = np.clip(img_np, 0.0, 1.0)
    num_images = img_np.shape[0]
    if img_format == "SCHW":  # Convert to SHWC
      img_np = img_np.transpose((0, 2, 3, 1))
    laid_out = layout_img_batch(img_np, max_display)
    if filename is not None:
      filename += ".png"
      print(f"Saving temporary image {filename} (shape={laid_out.shape})")
      cv2.imwrite(filename, cv2.cvtColor(laid_out, cv2.COLOR_BGR2RGB) * 255)
    if show:
      cv2_imshow(cv2.cvtColor(laid_out, cv2.COLOR_BGR2RGB) * 255)
    return laid_out


class VideoWriter:
  """Create a video from image frames."""

  def __init__(self, filename="_autoplay.mp4", fps=20.0, **kw):
    """Video creator.
    Creates and display a video made from frames. The default
    filename causes the video to be displayed on exit.
    Args:
      filename: name of video file
      fps: frames per second for video
      **kw: args to be passed to FFMPEG_VideoWriter
    Returns:
      VideoWriter instance.
    """

    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)
    print("No video writing implemented")

  def add(self, img):
    """Add image to video.
    Add new frame to image file, creating VideoWriter if requried.
    Args:
      img: array-like frame, shape [X, Y, 3] or [X, Y]
    Returns:
      None
    """
    pass
    # img = np.asarray(img)
    # if self.writer is None:
    #   h, w = img.shape[:2]
    #   self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    # if img.dtype in [np.float32, np.float64]:
    #   img = np.uint8(img.clip(0, 1)*255)
    # if len(img.shape) == 2:
    #   img = np.repeat(img[..., None], 3, -1)
    # self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()
    if self.params["filename"] == "_autoplay.mp4":
      self.show()

  def show(self, **kw):
    """Display video.
    Args:
      **kw: args to be passed to mvp.ipython_display
    Returns:
      None
    """
    self.close()
    fn = self.params["filename"]
    if config["gui"]:
      display(mvp.ipython_display(fn, **kw))
