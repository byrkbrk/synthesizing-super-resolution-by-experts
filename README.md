# Synthesize Super Resolution Image by Experts Mining

## Introduction

  We build a module that synthesizes super-resolution images by 4x upscaling. While preparing, we utilize the pretrained model [SeemoRe](https://arxiv.org/abs/2402.03412) provided by [eduardzamfir at HuggingFace](https://huggingface.co/eduardzamfir/SeemoRe-T/tree/main).


## Setting Up the Environment

### Using Conda (recommended)

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository:
    ~~~
    git clone https://github.com/byrkbrk/synthesizing-super-resolution-by-experts.git
    ~~~
3. Change the directory:
    ~~~
    cd synthesizing-super-resolution-by-experts
    ~~~
4. Create the environment:
    ~~~
    conda env create -f synthesizing-sr-by-experts.yaml
    ~~~
5. Activate the environment:
    ~~~
    conda activate synthesizing-sr-by-experts
    ~~~


### Using Pip

1. Download & install [Python](https://www.python.org/downloads/) (version==3.11)
2. Clone the repository:
    ~~~
    git clone https://github.com/byrkbrk/synthesizing-super-resolution-by-experts.git
    ~~~
3. Change the directory:
    ~~~
    cd synthesizing-super-resolution-by-experts
    ~~~
4. Install packages using `pip`:
    ~~~
    pip install -r requirements.txt
    ~~~

## Synthesizing SR Image

Check it out how to use:

~~~
python3 synthesize.py --help
~~~

Output:

~~~
Synthesize (4x upscaled) super-resolution images by SeemoRe

positional arguments:
  image_name            Name of the image that be upscaled. Note image that be
                        processed must be in `low-res-images` directory

options:
  -h, --help            show this help message and exit
  --device {cuda,mps,cpu}
                        Name of the GPU device that be used during inference.
                        Default: None
~~~

### Example usages

Execute the followings to obtain super-resolved images:

~~~
python3 synthesize.py building.png
~~~

~~~
python3 synthesize.py plant.png
~~~

The output images seen below (left: Original, right: Super-resolved) will be saved into `./synthesized-images` folder.
<p align="center">
  <img src="low-res-images/building.png" width="49%" />
  <img src="files-for-readme/building.png" width="49%" />
</p>

<p align="center">
  <img src="low-res-images/plant.png" width="49%" />
  <img src="files-for-readme/plant.png" width="49%" />
</p>

## Synthesizing by using Gradio

To run the gradio app on your local computer, execute:

~~~
python3 app.py
~~~
Then, visit the url [http://127.0.0.1:7860](http://127.0.0.1:7860) to open the interface.

### Example usage

See the display below for an example usage of the module via Gradio.

<p align="left">
  <img src="files-for-readme/sr_synthesizer_app_ss.png" width="100%" />
</p>