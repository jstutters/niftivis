import click
from PIL import Image
import nibabel as nib
import numpy as np


def make_image(nifti, slc, mask=None, mask_alpha=1.0):
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
        mask_as_rgba[m[:, :, slc] > 0, 3] = int(255 * mask_alpha)
        mimg = Image.fromarray(mask_as_rgba, mode="RGBA")
        img = Image.alpha_composite(img, mimg)
    return img


def grid_image(img, rows, cols, start_slice=0, end_slice=None, thumbnail_dims=(128, 128), mask=None, mask_alpha=1.0):
    """Make a thumbnail grid from a NIFTI image."""
    n_img = rows * cols
    end_slice = end_slice or img.get_data().shape[2]
    if end_slice > img.get_data().shape[2]:
        end_slice = img.get_data().shape[2]
    n_slices = end_slice - start_slice
    slice_step = n_slices // n_img
    final_image = None
    thumb_width, thumb_height = thumbnail_dims
    for i, slc in enumerate(range(start_slice, end_slice, slice_step)):
        im = make_image(img, slc, mask, mask_alpha)
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
@click.option("-r", "--rows", type=int, default=5)
@click.option("-c", "--columns", type=int, default=5)
@click.option("-w", "--thumbnail-width", "thumbnail_width", type=int, default=128)
@click.option("-h", "--thumbnail-height", "thumbnail_height", type=int, default=128)
@click.option("-a", "--alpha", type=float, default=1.0)
@click.option("-s", "--start-slice", "start_slice", type=int, default=0)
@click.option("-e", "--end-slice", "end_slice", type=int)
@click.argument("image", type=click.Path(exists=True))
@click.argument("thumbnail", type=click.Path())
def make_thumbnails(image, thumbnail, mask=None, rows=5, columns=5,
                    thumbnail_width=128, thumbnail_height=128, alpha=1.0,
                    start_slice=0, end_slice=None):
    image = nib.load(image)
    if mask:
        mask = nib.load(mask)
    grid = grid_image(
        image,
        rows,
        columns,
        start_slice=start_slice,
        end_slice=end_slice,
        thumbnail_dims=(thumbnail_width, thumbnail_height),
        mask=mask,
        mask_alpha=alpha
    )
    grid.save(thumbnail)


if __name__ == "__main__":
    make_thumbnails()
