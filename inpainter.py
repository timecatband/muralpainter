from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps

class Inpainter():
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.mask_color = (255, 0, 0)
        #self.pipe.safety_checker = lambda images, a: images, False

    def dilate_mask(mask):
        # mask is np.array convert to image
        mask = Image.fromarray(mask)

        mask = mask.convert("L")
        mask = np.array(mask)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Create a square structuring element for dilation
        kernel = np.ones((5, 5), np.uint8)
        
        # Dilate the mask
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        return dilated_mask

    def inpaint_image(self, image, text, negative_prompt):
        img_array = np.array(image)
        mask_array = np.zeros_like(img_array)

        # Identify all pure red pixels (255, 0, 0)
        red_pixels = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)

        # Set the mask pixels to white (255, 255, 255) for the identified red pixels
        mask_array[red_pixels] = (255, 255, 255)

        # Convert the NumPy array back to a PIL image
        mask = Image.fromarray(mask_array)


        image = image.convert("RGB")
        
        # Run diffusion inpainting on image with mask
        inpainted_image = self.pipe(prompt=text,
                        negative_prompt=negative_prompt,
                        guidance_scale=8.0,
                        image=image, mask_image=mask)

        output_image = inpainted_image.images[0]
        # Correct output image colorspace
        output_image = output_image.convert("RGB")
        # Replace output image with original image where mask is empty
        mask=mask.convert("L")
        # Invert mask
        mask = ImageOps.invert(mask)
        output_image.paste(image, mask=mask)
        return output_image
    def inpaint_image_patch(self, image, x, y, text, negative, size=512):
        print("inpainting patch x: {} y: {}".format(x, y))
        print("text: {}".format(text))
        # Crop a 512x512 patch from the image
        patch = image.crop((x - 256, y - 256, x + 256, y + 256))
        # Inpaint the patch
        patch = self.inpaint_image(patch, text, negative)
        # Paste the inpainted patch back into the image
        image.paste(patch, (x - 256, y - 256))
        return image
