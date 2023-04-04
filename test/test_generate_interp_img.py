import sys

sys.path.append("./")

from util.generate import generate_different_interpolation_img

generate_different_interpolation_img(
    [
        "image/set5/GTmod12/head.png",
        "image/set5/GTmod12/bird.png",
        "image/set5/GTmod12/butterfly.png",
    ],
    6,
    "./differ_interp_img.png",
)
