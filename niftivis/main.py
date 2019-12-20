import click
from PIL import Image
import nibabel as nib
import numpy as np


def make_image(nifti, slc, mask=None):
    """Turn a single slice from a NIFTI image into an RGBA image."""
    d = nifti.get_data()
    d = d / 16
    d = d.astype(np.uint8)
    as_rgba = np.zeros([d.shape[0], d.shape[1], 4], dtype=np.uint8)
    as_rgba[:, :, 0] = d[:, :, slc]
    as_rgba[:, :, 1] = d[:, :, slc]
    as_rgba[:, :, 2] = d[:, :, slc]
    as_rgba[:, :, 3] = 255
    img = Image.fromarray(as_rgba, mode="RGBA")
    if mask:
        m = mask.get_data()
        mask_as_rgba = np.zeros([d.shape[0], d.shape[1], 4], dtype=np.uint8)
        mask_as_rgba[m[:, :, slc] > 0, 0] = 255
        mask_as_rgba[m[:, :, slc] > 0, 3] = 255
        mimg = Image.fromarray(mask_as_rgba, mode="RGBA")
        img = Image.alpha_composite(img, mimg)
    return img


def grid_image(img, rows, cols, thumb_width=128, thumb_height=128, mask=None):
    """Make a thumbnail grid from a NIFTI image."""
    n_img = rows * cols
    n_slices = img.get_data().shape[2]
    slice_step = n_slices // n_img
    final_image = None
    for i, slc in enumerate(range(0, n_slices, slice_step)):
        im = make_image(img, slc, mask)
        im.thumbnail((thumb_width, thumb_height))
        if final_image is None:
            thumb_width, thumb_height = im.width, im.height
            final_image = Image.new("RGBA", (cols * thumb_width, rows * thumb_height))
        c = i % cols
        r = i // cols
        x = c * thumb_width
        y = r * thumb_height
        final_image.paste(im, (x, y))
    return final_image


@click.command()
@click.option("-m", "--mask", type=click.Path(exists=True))
@click.argument("image", type=click.Path(exists=True))
@click.argument("thumbnail", type=click.Path())
def make_thumbnails(image, thumbnail, mask=None):
    image = nib.load(image)
    if mask:
        mask = nib.load(mask)
    grid = grid_image(image, 5, 5, mask=mask)
    grid.save(thumbnail)


if __name__ == "__main__":
    make_thumbnails()
