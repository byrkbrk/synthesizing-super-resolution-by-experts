import os
import yaml
import torch
from model.seemore import SeemoRe
from torch.nn.parallel import DataParallel, DistributedDataParallel
from huggingface_hub import hf_hub_download



class SRSynthesizer(object):
    repo_id = "eduardzamfir/SeemoRe-T"
    checkpoint_name = "SeemoRe_T_X4.pth"
    model_config_name = "eval_seemore_t_x4.yaml"

    def __init__(self,
                 device: str = None):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.download_model_checkpoint(self.__class__.repo_id,
                                       self.__class__.checkpoint_name)
        self.model = self.instantiate_model(self.__class__.checkpoint_name,
                                            self.__class__.model_config_name,
                                            self.device)

    def synthesize(self, image):
        """Returns synthesized image for given image"""
        if isinstance(image, str):
            image = self.read_image(image)
        pass

    def instantiate_model(self, checkpoint_name, model_config_name, device):
        """Returns instantiated model for given arguments"""
        model = SeemoRe(**self.read_model_config_file(model_config_name)).to(device)
        model.load_state_dict(self.load_checkpoint(checkpoint_name, device))
        print("the model instantiated successfully!")
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


if __name__ == "__main__":
    SRSynthesizer()