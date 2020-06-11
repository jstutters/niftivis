import click
from PIL import Image, ImageOps
import nibabel as nib
import numpy as np


def make_image(nifti, slc, mask=None, rotate=True, mask_alpha=0.5):
    """Turn a single slice from a NIFTI image into an RGBA image."""
    d = nifti.get_data()
    d = d / 16
    d = d.astype(np.uint8)
    # as_rgba = np.zeros([d.shape[0], d.shape[1], 4], dtype=np.uint8)
    as_rgba = np.zeros([d.shape[0], d.shape[1], 3], dtype=np.uint8)
    as_rgba[:, :, 0] = d[:, :, slc]
    as_rgba[:, :, 1] = d[:, :, slc]
    as_rgba[:, :, 2] = d[:, :, slc]
    if rotate:
        as_rgba = np.rot90(as_rgba)
    # as_rgba[:, :, 3] = 255
    img = Image.fromarray(as_rgba, mode="RGB")
    img = ImageOps.autocontrast(img, 1)
    alpha_channel = np.zeros([d.shape[0], d.shape[1]], dtype=np.uint8)
    alpha_channel[:, :] = 255
    if rotate:
        alpha_channel = np.rot90(alpha_channel)
    alpha_channel = Image.fromarray(alpha_channel, mode="L")
    img.putalpha(alpha_channel)
    if mask:
        m = mask.get_data()
        mask_as_rgba = np.zeros([d.shape[0], d.shape[1], 4], dtype=np.uint8)
        mask_as_rgba[m[:, :, slc] > 0, 0] = 255
        mask_as_rgba[m[:, :, slc] > 0, 3] = int(255 * mask_alpha)
        if rotate:
            mask_as_rgba = np.rot90(mask_as_rgba)
        mimg = Image.fromarray(mask_as_rgba, mode="RGBA")
        img = Image.alpha_composite(img, mimg)
    return img


def grid_image(img, rows, cols, thumbnail_dims=(128, 128), mask=None, from_slice=0,
               to_slice=None, rotate=False):
    """Make a thumbnail grid from a NIFTI image."""
    n_img = rows * cols
    n_slices = img.get_data().shape[2]
    if to_slice is None:
        to_slice = n_slices
    elif to_slice > n_slices:
        to_slice = n_slices
    elif to_slice < 0:
        to_slice = n_slices + to_slice
    slice_step = (to_slice - from_slice) // n_img
    final_image = None
    thumb_width, thumb_height = thumbnail_dims
    for i, slc in enumerate(range(from_slice, to_slice, slice_step)):
        im = make_image(img, slc, mask, rotate)
        im.thumbnail((thumb_width, thumb_height))
        if final_image is None:
            tw, th = im.width, im.height
            final_image = Image.new("RGBA", (cols * tw, rows * th))
        c = i % cols
        r = i // cols
        x = c * tw
        y = r * th
        final_image.paste(im, (x, y))
    return final_image


@click.command()
@click.option("-m", "--mask", type=click.Path(exists=True))
@click.option("-r", "--rows", type=int, default=5)
@click.option("-c", "--columns", type=int, default=5)
@click.option("-s", "--thumbnail-size", "thumbnail_size", type=int)
@click.option("-w", "--thumbnail-width", "thumbnail_width", type=int, default=128)
@click.option("-h", "--thumbnail-height", "thumbnail_height", type=int, default=128)
@click.option("-f", "--from", "from_slice", type=int, default=0)
@click.option("-t", "--to", "to_slice", type=int)
@click.option("--rotate", flag_value=True, default=False)
@click.version_option(version="2020.06.1")
@click.argument("image", type=click.Path(exists=True))
@click.argument("thumbnail", type=click.Path())
def make_thumbnails(image, thumbnail, mask=None, rows=5, columns=5,
                    thumbnail_size=None, thumbnail_width=128, thumbnail_height=128,
                    from_slice=0, to_slice=None, rotate=False):
    image = nib.load(image)
    if mask:
        mask = nib.load(mask)
    if thumbnail_size:
        thumbnail_width = thumbnail_size
        thumbnail_height = thumbnail_size
    grid = grid_image(
        image,
        rows,
        columns,
        from_slice=from_slice,
        to_slice=to_slice,
        thumbnail_dims=(thumbnail_width, thumbnail_height),
        mask=mask,
        rotate=rotate
    )
    grid.save(thumbnail)


if __name__ == "__main__":
    make_thumbnails() # pylint: disable=no-value-for-parameter