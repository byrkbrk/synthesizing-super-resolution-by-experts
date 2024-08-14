import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
from model.seemore import SeemoRe
from huggingface_hub import hf_hub_download



class SRSynthesizer(object):
    repo_id = "eduardzamfir/SeemoRe-T"
    checkpoint_name = "SeemoRe_T_X4.pth"
    model_config_name = "eval_seemore_t_x4.yaml"

    def __init__(self,
                 device: str = None,
                 create_dirs: bool = True):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.download_model_checkpoint(self.__class__.repo_id,
                                       self.__class__.checkpoint_name)
        self.model = self.instantiate_model(self.__class__.checkpoint_name,
                                            self.__class__.model_config_name,
                                            self.device)
        if create_dirs: self.create_dirs(self.module_dir)

    @torch.inference_mode()
    def synthesize(self, image, show=True, save=True, return_input=False):
        """Returns synthesized image for given image"""
        if isinstance(image, str):
            synthesized_image_name = image
            image = self.read_image(self.module_dir, "low-res-images", image)
        else:
            synthesized_image_name = "synthesized_image.png"

        synthesized_image = self.model(transforms.ToTensor()(image).to(self.device))
        synthesized_image = transforms.Compose([lambda x: torch.clamp(x, 0, 1),
                                                transforms.ToPILImage()])(synthesized_image.squeeze().cpu())
        if show:
            image.show()
            synthesized_image.show()
        if save:
            synthesized_image.save(os.path.join(self.module_dir, 
                                                "synthesized-images",
                                                synthesized_image_name))
        if return_input:
            return image, synthesized_image
        return synthesized_image

    def instantiate_model(self, checkpoint_name, model_config_name, device):
        """Returns instantiated model for given arguments"""
        model = SeemoRe(**self.read_model_config_file(model_config_name)).to(device)
        model.load_state_dict(self.load_checkpoint(checkpoint_name, device))
        return model
    
    def read_model_config_file(self, config_name):
        """Returns read yaml file for given config name"""
        root = self.module_dir
        base_folder = "model"
        with open(os.path.join(root, base_folder, config_name), "r") as file:
            return yaml.safe_load(file)

    def load_checkpoint(self, checkpoint_name, device):
        """Loads the checkpoint from memory for given checkpoint name"""
        root = self.module_dir
        base_folder = "model"
        checkpoint = torch.load(os.path.join(root, base_folder, checkpoint_name), 
                                weights_only=True,
                                map_location=device)
        return checkpoint["params"]

    def download_model_checkpoint(self, repo_id, checkpoint_name, location=None):
        """Downloads the model checkpoint from huggingface to given location"""
        if location is None:
            location = os.path.join(self.module_dir, "model")
        hf_hub_download(repo_id=repo_id,
                        filename=checkpoint_name,
                        local_dir=location)
    
    def initialize_device(self, device: str):
        """Returns device based on GPU availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def read_image(self, root, base_folder, image_name):
        """Returns opened image file for given image name"""
        return Image.open(os.path.join(root, base_folder, image_name))
    
    def create_dirs(self, root: str) -> None:
        """Creates required directories during inference under root"""
        dir_names = ["low-res-images", "synthesized-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)
