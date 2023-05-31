import PIL
from PIL import Image
import os
import sys
import cv2
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import numpy as np
# Import PIL image ops
from PIL import ImageOps
from inpainter import Inpainter
import math

class MuralConfiguration():
    def __init__(self, width_inches, height_inches, dpi, overlap=0.5):
        self.width_inches = width_inches
        self.height_inches = height_inches
        self.dpi = dpi
        self.patch_size = 512
        self.total_image_resolution = self.dpi * self.width_inches, self.dpi * self.height_inches
        self.patch_dimensions = int(((1/overlap)*self.total_image_resolution[0]) // self.patch_size), int(((1/overlap)*self.total_image_resolution[1] // self.patch_size))
        self.overlap = overlap
    def get_patch_center(self, px, py):
        tl_x = px * self.patch_size*self.overlap
        tl_y = py * self.patch_size*self.overlap
        return int(tl_x + self.patch_size//2), int(tl_y + self.patch_size//2)
    def make_image(self):
        return Image.new("RGB", self.total_image_resolution, (255, 0, 0))

class GridPrompter():
    def __init__(self, base_prompt, grid, negative_grid = None):
        self.grid = grid
        self.base_prompt = base_prompt
        self.negative_base_prompt = "seam. border. watermark. logo. text. person. ugly. low quality. low resolution. bad art."
        self.negative_grid = negative_grid
    def get_prompt(self, normalized_px, normalized_py):
        x_coord = int(normalized_px * len(self.grid[0]))
        y_coord = int(normalized_py * len(self.grid))
        return self.base_prompt + self.grid[y_coord][x_coord]
    def get_negative_prompt(self, normalized_px, normalized_py):
        if self.negative_grid is None:
            return self.negative_base_prompt
        x_coord = int(normalized_px * len(self.negative_grid[0]))
        y_coord = int(normalized_py * len(self.negative_grid))
        print("Negative prompt coord: ", x_coord, y_coord)
        return self.negative_base_prompt + self.negative_grid[y_coord][x_coord]

class CenterOutStrategy():
    def __init__(self, config, inpainter, prompter):
        self.config = config
        self.inpainter = inpainter
        self.image = config.make_image()
        self.prompter = prompter
    def paint_patches(self, sx, sy, nx, ny):
        for px in range(0, abs(nx)):
            px *= 1 if nx > 0 else -1
            for py in range(0, abs(ny)):
                py *= 1 if ny > 0 else -1
                print("Doing patch px,py: ", px, py)
                cx, cy = self.config.get_patch_center(sx+px, sx+py)
                prompt = self.prompter.get_prompt((sx+px) / self.config.patch_dimensions[0], (sx+py) / self.config.patch_dimensions[1])
                negative_prompt = self.prompter.get_negative_prompt((sx+px) / self.config.patch_dimensions[0], (sx+py) / self.config.patch_dimensions[1])
                self.image = self.inpainter.inpaint_image_patch(self.image, cx, cy, prompt, negative_prompt)
                self.image.save("inpainted.png")
        return self.image
    def paint(self):
        sx = self.config.patch_dimensions[0] // 2
        sy = self.config.patch_dimensions[1] // 2
        print("sx: ", sx)
        print("sy: ", sy)
        print("dimen: ", self.config.patch_dimensions[0], self.config.patch_dimensions[1])
        # Paint bottom right patches
        self.image = self.paint_patches(sx, sy, self.config.patch_dimensions[0]-sx, self.config.patch_dimensions[1]-sy)
        # Paint top right patches
        self.image = self.paint_patches(sx, sy, self.config.patch_dimensions[0]-sx, -(self.config.patch_dimensions[1]-sy))
        # Paint bottom left patches
        self.image = self.paint_patches(sx, sy, -(self.config.patch_dimensions[0]-sx), self.config.patch_dimensions[1]-sy)
        # Paint top left patches
        self.image = self.paint_patches(sx, sy, -(self.config.patch_dimensions[0]-sx), -(self.config.patch_dimensions[1]-sy))
        return self.image



grid = [["Beautiful sky, a dragon", "Beautiful sky, the sun", "Beautiful sky, Birds", "Beautiful sky, Complex clouds"],
        ["Beautiful trees", "Beautiful grassy hills in the distance", "Sunny hillside village", "Beautiful mountain village", "A mountain temple"],
        ["A river", "Riverbank", "A field of flowers", "Beautiful field with animals", "Cute animals frolicking in a field"]]
negative_grid = [["trees, ground"],
                 ["grass, trail, road, path, sky"],
                 ["sky", "mountain"]]
base_prompt = "A beautiful anime landscape painting, highly detailed, 8k, trending on artstation. Studio Ghibli. "
prompter = GridPrompter(base_prompt, grid, negative_grid)
    

# Create an empty RGBA image of size 8192
config = MuralConfiguration(10, 10, 300)
print("Total resolution: ", config.total_image_resolution)
print("Patch dimensions: ", config.patch_dimensions)
inpainter = Inpainter()
strategy = CenterOutStrategy(config, inpainter, prompter)

image = strategy.paint()

# Save image
image.save("inpainting.png")

