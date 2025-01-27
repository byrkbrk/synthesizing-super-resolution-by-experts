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
        outputs=ImageSlider(type="pil", label="Output", show_download_button=True),
        title="Super Resolution Image Synthesizer",
        examples=[
            [os.path.join(os.path.dirname(__file__), "low-res-images", "building.png")],
            [os.path.join(os.path.dirname(__file__), "low-res-images", "plant.png")],
            [os.path.join(os.path.dirname(__file__), "low-res-images", "penguin.png")],
            [os.path.join(os.path.dirname(__file__), "low-res-images", "vietnam_park.jpg")],
        ],
        description="Synthesize (4x-upscaled) super-resolved images"
    )
    gr_interface.launch()
