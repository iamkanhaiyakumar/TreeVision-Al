"""
TreeVision AI — Geospatial Image Tiling Module
Generates sliding-window tiles with spatial overlap while preserving
pixel offsets, original image dimensions, and georeferenced transforms.
Supports both in-memory arrays and direct rasterio windowed reads.
"""

from typing import List, Dict, Any, Generator, Tuple, Optional
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

class Tile:
    def __init__(
        self,
        tile_id: int,
        image: np.ndarray,
        row_offset: int,
        col_offset: int,
        width: int,
        height: int,
        transform: Optional[Affine] = None,
        crs: Optional[str] = None
    ):
        self.tile_id = tile_id
        self.image = image
        self.row_offset = row_offset
        self.col_offset = col_offset
        self.width = width
        self.height = height
        self.transform = transform
        self.crs = crs

def generate_tile_windows(
    img_width: int,
    img_height: int,
    tile_size: int = 640,
    overlap: float = 0.15
) -> List[Tuple[int, int, int, int]]:
    """
    Computes (col_offset, row_offset, width, height) for all tiles.
    """
    stride = int(tile_size * (1.0 - overlap))
    if stride <= 0:
        stride = tile_size

    windows = []
    
    y = 0
    while y < img_height:
        # Adjust last tile to clamp to edge
        cur_h = min(tile_size, img_height - y)
        actual_y = y if y + tile_size <= img_height else max(0, img_height - tile_size)
        actual_h = min(tile_size, img_height - actual_y)
        
        x = 0
        while x < img_width:
            actual_x = x if x + tile_size <= img_width else max(0, img_width - tile_size)
            actual_w = min(tile_size, img_width - actual_x)
            
            windows.append((actual_x, actual_y, actual_w, actual_h))
            
            if x + tile_size >= img_width:
                break
            x += stride
            
        if y + tile_size >= img_height:
            break
        y += stride
        
    # Remove any duplicate windows from edge clamping
    unique_windows = list(dict.fromkeys(windows))
    return unique_windows

def tile_image_array(
    image_rgb: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.15,
    base_transform: Optional[Affine] = None,
    crs: Optional[str] = None
) -> List[Tile]:
    """
    Extracts tiles from an in-memory RGB numpy array (H, W, 3).
    """
    h, w = image_rgb.shape[:2]
    windows = generate_tile_windows(w, h, tile_size, overlap)
    
    tiles = []
    for idx, (col_off, row_off, tile_w, tile_h) in enumerate(windows):
        tile_crop = image_rgb[row_off : row_off + tile_h, col_off : col_off + tile_w]
        
        # If tile is smaller than tile_size at border, pad with edge pixels or black
        if tile_crop.shape[0] != tile_size or tile_crop.shape[1] != tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=image_rgb.dtype)
            padded[:tile_crop.shape[0], :tile_crop.shape[1]] = tile_crop
            tile_crop = padded

        # Calculate localized affine transform for the tile
        tile_transform = None
        if base_transform is not None:
            # Shift origin by (col_offset, row_offset)
            tile_transform = base_transform * Affine.translation(col_off, row_off)

        tiles.append(
            Tile(
                tile_id=idx,
                image=tile_crop,
                row_offset=row_off,
                col_offset=col_off,
                width=tile_w,
                height=tile_h,
                transform=tile_transform,
                crs=crs
            )
        )
        
    return tiles

def stream_geotiff_tiles(
    tif_path: str,
    tile_size: int = 640,
    overlap: float = 0.15
) -> Generator[Tile, None, None]:
    """
    Streams tiles directly from a large GeoTIFF file using rasterio.windows.Window,
    preventing out-of-memory errors on massive satellite rasters.
    """
    with rasterio.open(tif_path) as src:
        windows = generate_tile_windows(src.width, src.height, tile_size, overlap)
        base_transform = src.transform
        crs = str(src.crs) if src.crs else None

        for idx, (col_off, row_off, tile_w, tile_h) in enumerate(windows):
            window = Window(col_off, row_off, tile_w, tile_h)
            # Read bands (3, H, W) and transpose to (H, W, 3)
            data = src.read([1, 2, 3], window=window)
            data = np.transpose(data, (1, 2, 0))

            if data.shape[0] != tile_size or data.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=data.dtype)
                padded[:data.shape[0], :data.shape[1]] = data
                data = padded

            tile_transform = base_transform * Affine.translation(col_off, row_off)

            yield Tile(
                tile_id=idx,
                image=data,
                row_offset=row_off,
                col_offset=col_off,
                width=tile_w,
                height=tile_h,
                transform=tile_transform,
                crs=crs
            )
