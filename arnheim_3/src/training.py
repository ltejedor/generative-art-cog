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

# Functions for the optimisation (including evolution) and evaluation.


from matplotlib import pyplot as plt

import numpy as np
import torch
import torchvision.transforms as transforms

import clip

from .video_utils import show_and_save
import torch.optim as optim
import json


# Show each image being evaluated for debugging purposes.
VISUALISE_BATCH_IMAGES = False


def augmentation_transforms(canvas_width,
                            use_normalized_clip=False,
                            use_augmentation=False):
  """Image transforms to produce distorted crops to augment the evaluation.
  Args:
    canvas_width: width of the drawing canvas
    use_normalized_clip: Normalisation to better suit CLIP's training data
    use_augmentation: Image augmentation by affine transform
  Returns:
    transforms
  """
  if use_normalized_clip and use_augmentation:
    augment_trans = transforms.Compose(
        [transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
         transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
         transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))])
  elif use_augmentation:
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.6),
        transforms.RandomResizedCrop(canvas_width, scale=(0.7, 0.9)),
    ])
  elif use_normalized_clip:
    augment_trans = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
  else:
    augment_trans = transforms.RandomPerspective(
        fill=1, p=0, distortion_scale=0)

  return augment_trans


def moving_average(a, n=3) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n


def plot_and_save_losses(loss_history, title="Losses", filename=None):
  losses = np.array(loss_history)
  if filename:
    plt.figure(figsize=(10,10))
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(moving_average(losses, n=3))
    np.save(filename + ".npy", losses, allow_pickle=True)
    plt.savefig(filename + ".png")


def make_optimizer(config, generator, learning_rate, weight_decay=0.01):
  """Make AdamW optimizer for generator's parameters."""
  my_list = ['positions_top']
  if config["loss"] == "MSE":
    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list,
                                                  generator.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list,
                                                      generator.named_parameters()))))
    optimizer = optim.AdamW([
        {'params': base_params},
        {'params': params, 'lr': learning_rate}
    ], lr=learning_rate, weight_decay=weight_decay)
    return optimizer
  else: #CLIP
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list,
                                                  generator.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in
                                        my_list, generator.named_parameters()))))
    lr_scheduler = torch.optim.SGD([{'params': base_params},
                                    {'params': params, 'lr': learning_rate}],
                                      lr=learning_rate)
    return lr_scheduler
  


def compute_text_features(prompts, clip_model, device):
  """Compute CLIP features for all prompts."""

  text_inputs = []
  for prompt in prompts:
    text_inputs.append(clip.tokenize(prompt).to(device))

  features = []
  with torch.no_grad():
    for text_input in text_inputs:
      features.append(clip_model.encode_text(text_input))
  return features


def create_augmented_batch(images, augment_trans, text_features, config):
  """Create batch of images to be evaluated.
  Returns:
    img_batch: For compositional images the batch contains all the regions.
        Otherwise the batch contains augmented versions of the original images.
    num_augs: number of images per original image
    text_features: a text feature for each augmentation
    config: dictionary with config
  """
  images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
  expanded_text_features = []
  if config["use_image_augmentations"]:
    num_augs = config["num_augs"]
    img_augs = []
    for n in range(num_augs):
      img_n = augment_trans(images)
      img_augs.append(img_n)
      expanded_text_features.append(text_features[0])
    img_batch = torch.cat(img_augs)
    # Given images [P0, P1] and augmentations [a0(), a1()], output format:
    # [a0(P0), a0(P1), a1(P0), a1(P1)]
  else:
    num_augs = 1
    img_batch = augment_trans(images)
    expanded_text_features.append(text_features[0])
  return img_batch, num_augs, expanded_text_features, [1] * config["num_augs"]


def create_compositional_batch(images, augment_trans, text_features):
  """Create 10 sub-images per image by augmenting each with 3x3 crops.
  Args:
    images: population of N images, format [N, C, H, W]
  Returns:
    Tensor of all compositional sub-images + originals; [N*10, C, H, W] format:
        [x0_y0(P0) ... x0_y0(PN), ..., x2_y2(P0) ... x2_y2(PN), P0, ..., PN]
    10: Number of sub-images + whole, per original image.
    expanded_text_features: list of text features, 1 for each composition image
    loss_weights: weights for the losses corresponding to each composition image
    """
  if len(text_features) != 10:
    # text_features should already be 10 in size.
    raise ValueError(
        "10 text prompts required for compositional image creation")
  resize_for_clip = transforms.Compose([transforms.Resize((224,224))])
  img_swap = torch.swapaxes(images, 3, 1)
  ims = []
  i = 0
  for x in range(3):
    for y in range(3):
      for k in range(images.shape[0]):
        ims.append(resize_for_clip(
            img_swap[k][:, y * 112 : y * 112 + 224, x * 112 : x * 112 + 224]))
      i += 1

  # Top-level (whole) images
  for k in range(images.shape[0]):
    ims.append(resize_for_clip(img_swap[k]))
  all_img = torch.stack(ims)
  all_img = torch.swapaxes(all_img, 1, 3)
  all_img = all_img.permute(0, 3, 1, 2)  # NHWC -> NCHW
  all_img = augment_trans(all_img)

  # Last image gets 9 times as much weight
  common_weight = 1 / 5
  loss_weights = [common_weight] * 9
  loss_weights.append(9 * common_weight)
  return all_img, 10, text_features, loss_weights


def evaluation(t, clip_enc, generator, augment_trans, text_features,
               prompts, config, device):
  """Do a step of evaluation, returning images and losses.
  Args:
    t: step count
    clip_enc: model for CLIP encoding
    generator: drawing generator to optimise
    augment_trans: transforms for image augmentation
    text_features: tuple with the prompt two negative prompts
    prompts: for debugging/visualisation - the list of text prompts
    config: dictionary with hyperparameters
    device: torch device
  Returns:
    loss: torch.Tensor of single combines loss
    losses_separate_np: numpy array of loss for each image
    losses_individuals_np: numpy array with loss for each population individual
    img_np: numpy array of images from the generator
  """

  # Annealing parameters.
  params = {'gamma': t / config["optim_steps"]}

  # Rebuild the generator.
  img = generator(params)
  img_np = img.detach().cpu().numpy()

  # Create images for different regions
  pop_size = img.shape[0]

  target_image = config["target_image"]

  #MSE Loss
  if config["loss"] == "MSE":
        # Ensure target image is a torch tensor on the correct device
    if not isinstance(target_image, torch.Tensor):
        target_image = torch.tensor(target_image).to(device)
    if target_image.shape[-1] == 3:  # If NHWC format
        target_image = target_image.permute(0, 3, 1, 2)  # Convert to NCHW
        
    # Resize target to match generator output if needed
    if target_image.shape[2:] != img.shape[1:3]:  # Check height, width
        target_image = torch.nn.functional.interpolate(
            target_image, size=(img.shape[1], img.shape[2]), 
            mode='bilinear', align_corners=False)
    
    # Calculate MSE loss for each population member
    losses = torch.zeros(pop_size).to(device)
    for p in range(pop_size):
        # Convert to NCHW format if needed
        img_p = img[p]
        if img_p.shape[-1] == 3:  # If NHWC format
            img_p = img_p.permute(2, 0, 1)  # Convert to CHW
            img_p = img_p.unsqueeze(0)  # Add batch dimension -> NCHW
        
        # Calculate MSE loss
        mse_loss = torch.nn.functional.mse_loss(img_p[:,:3,:,:], target_image[:,:3,:,:])
        losses[p] = mse_loss
        
    # Create a compatible losses_separate_np structure for consistency
    losses_separate_np = losses.unsqueeze(1).detach().cpu().numpy()
    losses_individuals_np = losses.detach().cpu().numpy()
    loss = torch.sum(losses) / pop_size
    
    #best_idx = np.argmin(losses_individuals_np)
    #print("best_idx", best_idx)
    return loss, losses_separate_np, losses_individuals_np, img_np
    
  else:  

    if config["compositional_image"]:
      (img_batch, num_augs, text_features, loss_weights
      ) = create_compositional_batch(img, augment_trans, text_features)
    else:
      (img_batch, num_augs, text_features, loss_weights
      ) = create_augmented_batch(img, augment_trans, text_features, config)
    losses = torch.zeros(pop_size, num_augs).to(device)

    # Compute and add losses after augmenting the image with transforms.
    img_batch = torch.clip(img_batch, 0, 1)  # clip the images.
    image_features = clip_enc.encode_image(img_batch)
    count = 0
    for n in range(num_augs):  # number of augmentations or composition images
      for p in range(pop_size):
        loss = torch.cosine_similarity(
            text_features[n], image_features[count:count+1], dim=1
            )[0] * loss_weights[n]
        losses[p, n] -= loss
        if VISUALISE_BATCH_IMAGES and t % 500 == 0:
          # Show all the images in the batch along with their losses.
          if config["compositional_image"]:
            print(f"Loss {loss} for image region with prompt {prompts[n]}:")
          else:
            print(f"Loss {loss} for image augmentation with prompt {prompts[0]}:")
          show_and_save(img_batch[count].unsqueeze(0), config,
              img_format="SCHW", show=config["gui"])
        count += 1
    loss = torch.sum(losses) / pop_size
    losses_separate_np = losses.detach().cpu().numpy()
    # Sum losses for all each population individual.
    losses_individuals_np = losses_separate_np.sum(axis=1)
    #best_idx = np.argmin(losses_individuals_np)
    #print("best_idx", best_idx)
    return loss, losses_separate_np, losses_individuals_np, img_np


def step_optimization(t, clip_enc, lr_scheduler, generator, augment_trans,
                      text_features, prompts, config, device, final_step=False):
  """Do a step of optimization.
  Args:
    t: step count
    clip_enc: model for CLIP encoding
    lr_scheduler: optimizer
    generator: drawing generator to optimise
    augment_trans: transforms for image augmentation
    text_features: list or 1 or 9 prompts for normal and compositional creation
    prompts: for debugging/visualisation - the list of text prompts
    config: dictionary with hyperparameters
    device: CUDA device
    final_step: if True does extras such as saving the model
  Returns:
    losses_np: numpy array with loss for each population individual
    losses_separate_np: numpy array of loss for each image
  """

  # Anneal learning rate and other parameters.
  if t == int(config["optim_steps"] / 3):
    for g in lr_scheduler.param_groups:
      g["lr"] = g["lr"] / 2.0
  if t == int(config["optim_steps"] * (2/3)):
    for g in lr_scheduler.param_groups:
      g["lr"] = g["lr"] / 2.0
  params = {'gamma': t / config["optim_steps"]}

  # Forward pass.
  lr_scheduler.zero_grad()
  loss, losses_separate_np, losses_np, img_np = evaluation(
      t=t, clip_enc=clip_enc, generator=generator, augment_trans=augment_trans,
      text_features=text_features, prompts=prompts, config=config,
      device=device)

  # Backpropagate the gradients.
  loss.backward()
  torch.nn.utils.clip_grad_norm_(generator.parameters(),
                                config["gradient_clipping"])
  if t % config["trace_every"] == 0 or final_step:
       output_dir = config["output_dir"]
       filename = f"{output_dir}/optim_{t}"
       show_and_save(img_np, config,
                     max_display=config["max_multiple_visualizations"],
                     stitch=True, img_format="SHWC",
                     show=config["gui"],
                     filename=filename)
       
       # Add this block to save patch locations
       try:
        # Extract patch locations from the spatial transformer
        patch_data = []
        for idx_patch in range(generator._num_patches):
          # Get patch index for the best member (which image from dataset)
          patch_idx = generator.patch_indices[0][idx_patch]
          
          # Map to your filename convention
          filename = f"image_{patch_idx}.png"
          
          # Get transformation values for this patch from the best member
          x_pos = generator.spatial_transformer.translation[0, idx_patch, 0, 0].item()
          y_pos = generator.spatial_transformer.translation[0, idx_patch, 1, 0].item()

          if config["colour_transformations"] == "RGB space":
              # For RGB color transformations
              patch_data.append({
                  "id": idx_patch,
                  "filename": filename,
                  "patch_index": int(patch_idx),
                  "patch_position": idx_patch,
                  "x": float(x_pos),
                  "y": float(y_pos),
                  "rotation": float(generator.spatial_transformer.rotation[0, idx_patch, 0, 0].item()),
                  "scale": float(generator.spatial_transformer.scale[0, idx_patch, 0, 0].item()),
                  "squeeze": float(generator.spatial_transformer.squeeze[0, idx_patch, 0, 0].item()),
                  "shear": float(generator.spatial_transformer.shear[0, idx_patch, 0, 0].item()),
                  # Add RGB color information
                  "red": float(generator.colour_transformer.reds[0, idx_patch, 0, 0, 0].item()),
                  "green": float(generator.colour_transformer.greens[0, idx_patch, 0, 0, 0].item()),
                  "blue": float(generator.colour_transformer.blues[0, idx_patch, 0, 0, 0].item()),
                  "order": float(generator.colour_transformer.orders[0, idx_patch, 0, 0, 0].item()),
              })
          elif config["colour_transformations"] == "HSV space":
              # For HSV color transformations
              patch_data.append({
                  "id": idx_patch,
                  "filename": filename,
                  "patch_index": int(patch_idx),
                  "patch_position": idx_patch,
                  "x": float(x_pos),
                  "y": float(y_pos),
                  "rotation": float(generator.spatial_transformer.rotation[0, idx_patch, 0, 0].item()),
                  "scale": float(generator.spatial_transformer.scale[0, idx_patch, 0, 0].item()),
                  "squeeze": float(generator.spatial_transformer.squeeze[0, idx_patch, 0, 0].item()),
                  "shear": float(generator.spatial_transformer.shear[0, idx_patch, 0, 0].item()),
                  # Add HSV color information
                  "hue": float(generator.colour_transformer.hues[0, idx_patch, 0, 0, 0].item()),
                  "saturation": float(generator.colour_transformer.saturations[0, idx_patch, 0, 0, 0].item()),
                  "value": float(generator.colour_transformer.values[0, idx_patch, 0, 0, 0].item()),
                  "order": float(generator.colour_transformer.orders[0, idx_patch, 0, 0, 0].item()),
              })
              
          # Save to JSON file
          with open(f"{output_dir}/patch_positions_{t}.json", 'w') as f:
              json.dump(patch_data, f, indent=2)
               
          print(f"Saved patch positions to {output_dir}/patch_positions_{t}.json")
       except Exception as e:
           print(f"Error saving patch positions: {e}")
             

  # Decay the learning rate.
  lr_scheduler.step()

  # Render the big version.
  if final_step:
    show_and_save(
        img_np, config, t=t, img_format="SHWC", show=config["gui"])
    output_dir = config["output_dir"]
    print(f"Saving model to {output_dir}...")
    torch.save(generator.state_dict(), f"{output_dir}/generator.pt")



    print("Iteration {:3d}, rendering loss {:.6f}".format(t, loss.item()))
  return losses_np, losses_separate_np, img_np


def population_evolution_step(generator, config, losses):
  """GA for the population."""

  if config["ga_method"] == "Microbial":

    # Competition between 2 random individuals; mutated winner replaces loser.
    indices = list(range(len(losses)))
    np.random.shuffle(indices)
    select_1, select_2 = indices[0], indices[1]
    if losses[select_1] < losses[select_2]:
      generator.copy_and_mutate_s(select_1, select_2)
    else:
      generator.copy_and_mutate_s(select_2, select_1)
  elif config["ga_method"] == "Evolutionary Strategies":

    # Replace rest of population with mutants of the best.
    winner = np.argmin(losses)
    for other in range(len(losses)):
      if other == winner:
        continue
      generator.copy_and_mutate_s(winner, other)
