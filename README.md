# Synthesize Super Resolution Image by Experts Mining

## Introduction

  We build a module that synthesizes super-resolution images by 4x upscaling. While preparing, we utilized the pretrained model [SeemoRe](https://arxiv.org/abs/2402.03412) provided in [HuggingFace](https://huggingface.co/eduardzamfir/SeemoRe-T/tree/main).

## Setting Up the Environment

## Synthesizing SR Image

### Example usages
~~~
python3 synthesize.py building.png
~~~

~~~
python3 synthesize.py plant.png
~~~
<p align="center">
  <img src="low-res-images/building.png" width="49%" />
  <img src="files-for-readme/building.png" width="49%" />
</p>

<p align="center">
  <img src="low-res-images/plant.png" width="49%" />
  <img src="files-for-readme/plant.png" width="49%" />
</p>

## Synthesizing by using Gradio
