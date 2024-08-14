from argparse import ArgumentParser
from synthesizer import SRSynthesizer



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Synthesize  (4x upscaled) super-resolution images by SeeMore")
    parser.add_argument("image_name",
                        type=str,
                        default=None,
                        help="""Name of the image that be upscaled. 
                                Note image that be processed must be in `low-res-images` directory""")
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        choices=["cuda", "mps", "cpu"],
                        help="Name of the GPU device that be used during inference. Default: None")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    SRSynthesizer(device=args.device).synthesize(args.image_name)