# predict.py
from cog import BasePredictor, Path, Input
from datetime import datetime
import os
import glob
import pathlib
import numpy as np
import torch
import clip
import src.collage as collage
import src.video_utils as video_utils
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Union


class Predictor(BasePredictor):
    def setup(self):
        # Any heavy one-time setup can be done here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the ViT-H/14-quickgelu model from Open CLIP
        self.clip_model, _ = clip.load("ViT-B/32", self.device, jit=False)
        

    def predict(
        self,
        prompt: str = Input(
            default="a photorealistic chicken",
            description="The global prompt for generating the collage."
        ),
        patch_url: str = Input(
            default="https://storage.googleapis.com/dm_arnheim_3_assets/collage_patches/animals.npy",
            description="An .npy file for the collage patches."
        ),
        num_patches: int = Input(
            default=100,
            description="Number of images to start with"
        ),
        background_red: int = Input(
            default=0
        ),
        background_green: int = Input(
            default=0
        ),
        background_blue: int = Input(
            default=0
        ),
        initial_positions: List[List[Union[str, float]]] = Input(
            default=[],
            description="List of lists representing the initial positions of patches."
        ),
        optim_steps: int = Input(
            default=250,
            description="Number of optimization steps to run during collage generation."
        ),
        loss: str = Input(
            choices=["CLIP", "MSE"],
            default="CLIP",
            description="Loss function to optimize the collage. Choose between CLIP and MSE. If MSE is selected, the prompt won't affect the image and target_image is required."
        ),
        target_image: Path = Input(
            default=None,
            description="Upload an image file for MSE loss. Required if loss is set to MSE."
        )
    ) -> Path:
        # Create a temporary output directory
        output_dir = "output_" + datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S') + '/'
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Ensure ffmpeg is available
        os.environ["FFMPEG_BINARY"] = "ffmpeg"

        print("initial positions")
        print(initial_positions)

        if target_image:
            # Process target image
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Open the image
            img = Image.open(target_image).convert('RGB')
            
            # Create a transform that resizes and crops to 224×224
            transform = transforms.Compose([
                transforms.Resize(256),  # Resize the shorter side to 256
                transforms.CenterCrop(224),  # Center crop to 224×224
                transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            ])
            
            # Apply transformations
            target_tensor = transform(img).unsqueeze(0)  # Add batch dimension
            
            # Convert to the format expected by the code
            target_image = target_tensor.to(self.device)
        
        # Hard-coded configuration values based on config_compositional.yaml,
        # with prompt and optim_steps now provided by the input parameters.
        config = {
            "output_dir": output_dir,
            "loss": loss,
            "initial_positions": initial_positions,
            "target_image": target_image,
            "render_method": "transparency",
            "num_patches": num_patches,
            "colour_transformations": "RGB space",
            "optim_steps": optim_steps,  # Use the input value
            "learning_rate": 0.1,
            "patch_set": "animals.npy",
            "fixed_scale_patches": True,
            "fixed_scale_coeff": 0.5,
            "background_red": background_red,
            "background_green": background_green,
            "background_blue": background_blue,
            "patch_url": patch_url,
            "global_prompt": prompt,  # Use the input value
            "compositional_image": True,
            "tiles_wide": 1,
            "tiles_high": 1,
            "tile_images": False,
            "torch_device": str(self.device),
            "background_use": "Global",  # Required parameter
            "high_res_multiplier": 4,
            "canvas_width": 224,
            "canvas_height": 224,
            "gui": False,
            "clean_up": False,
            "video_steps": 0,
            "trace_every": 50,
            "use_image_augmentations": True,
            "num_augs": 1,
            "use_normalized_clip": False,
            "gradient_clipping": 10.0,
            "initial_search_size": 1,
            "initial_search_num_steps": 1,
            "pop_size": 2,
            "evolution_frequency": 100,
            "ga_method": "Microbial",
            "pos_and_rot_mutation_scale": 0.02,
            "scale_mutation_scale": 0.02,
            "distort_mutation_scale": 0.02,
            "colour_mutation_scale": 0.02,
            "patch_mutation_probability": 1,
            "max_multiple_visualizations": 5,
            "max_block_size_high_res": 2000,
            "invert_colours": False,
            "population_video": False,
            "global_tile_prompt": False,
            "init_checkpoint": "",
            "tile_prompt_string": "",
            "multiple_patch_set": None,
            "multiple_fixed_scale_patches": None,
            "multiple_patch_max_proportion": None,
            "multiple_fixed_scale_coeff": None,
            "normalize_patch_brightness": False,
            "patch_max_proportion": 5,
            "patch_width_min": 16,
            "patch_height_min": 16,
            "min_trans": -1.0,
            "max_trans": 1.0,
            "min_trans_init": -1.0,
            "max_trans_init": 1.0,
            "min_scale": 1.0,
            "max_scale": 2.0, 
            "min_squeeze": 0.5,
            "max_squeeze": 2.0,
            "min_shear": -0.2,
            "max_shear": 0.2,
            "min_rot_deg": -180,
            "max_rot_deg": 180,
            "min_rgb": -0.2,
            "max_rgb": 1.0,
            "initial_min_rgb": 0.5,
            "initial_max_rgb": 1.0,
            "min_hue_deg": 0,
            "max_hue_deg": 360,
            "min_sat": 0,
            "max_sat": 1,
            "min_val": 0,
            "max_val": 1,
            "tile_prompt_formating": "close-up of {}",
            "patch_repo_root": "https://storage.googleapis.com/dm_arnheim_3_assets/collage_patches",
            "url_to_patch_file": "",
            
            # Prompts for different regions of the image
            "prompt_x0_y0": "a photorealistic sky with sun",
            "prompt_x1_y0": "a photorealistic sky",
            "prompt_x2_y0": "a photorealistic sky with moon",
            "prompt_x0_y1": "a photorealistic tree",
            "prompt_x1_y1": "a photorealistic tree",
            "prompt_x2_y1": "a photorealistic tree",
            "prompt_x0_y2": "a photorealistic field",
            "prompt_x1_y2": "a photorealistic field",
            "prompt_x2_y2": "a photorealistic chicken"
        }
        
        # Adjust config for compositional image
        if config["compositional_image"]:
            config['canvas_width'] *= 2
            config['canvas_height'] *= 2
            config['high_res_multiplier'] = int(config['high_res_multiplier'] / 2)
        
        # Create prompts for the tiles
        all_prompts = []
        if config["compositional_image"]:
            list_tile_prompts = [
                config["prompt_x0_y0"], config["prompt_x1_y0"], config["prompt_x2_y0"],
                config["prompt_x0_y1"], config["prompt_x1_y1"], config["prompt_x2_y1"],
                config["prompt_x0_y2"], config["prompt_x1_y2"], config["prompt_x2_y2"]
            ]
            list_tile_prompts.append(config["global_prompt"])
            all_prompts.append(list_tile_prompts)
        else:
            all_prompts.append([config["global_prompt"]])
        
        # Create a simple background
        background_image = np.ones((10, 10, 3), dtype=np.float32)
        background_image[:, :, 0] = config["background_red"] / 255.
        background_image[:, :, 1] = config["background_green"] / 255.
        background_image[:, :, 2] = config["background_blue"] / 255.
        
        # Initialize the collage tiler
        ct = collage.CollageTiler(
            prompts=all_prompts,
            fixed_background_image=background_image,
            clip_model=self.clip_model,
            device=self.device,
            config=config
        )
        
        # Run the collage optimization loop
        for step in ct.loop():
            if step % config["trace_every"] == 0:
                yield [Path(output_dir + "optim_" + str(step) + ".png"), Path(output_dir + "patch_positions_" + str(step) + ".json")]


