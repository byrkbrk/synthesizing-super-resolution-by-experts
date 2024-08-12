import os
import gradio as gr
from synthesizer import SRSynthesizer
from gradio_imageslider import ImageSlider



if __name__ == "__main__":
    sr_synthesizer = SRSynthesizer(create_dirs=False)
    gr_interface = gr.Interface(
        fn=lambda image: sr_synthesizer.synthesize(image,
                                                   show=False,
                                                   save=False,
                                                   return_input=True),
        inputs=[gr.Image(type="pil", label="Input")],
        outputs=ImageSlider(type="pil", label="Output"),
        title="Super Resolution (x4) Image Synthesizer",
    )
    gr_interface.launch()
