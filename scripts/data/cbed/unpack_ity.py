import argparse
from collections import namedtuple
import json
import enum
import os
import struct

import tqdm
import numpy as np

DATA_TYPES = {
    8 : 'float32',
}

DATA_MODES = {
    0 : 'spectrum',
    1 : 'series_3d',
    2 : 'stem_4d',
}

ITYHeader = namedtuple(
    'ITYHeader', [
        'version', 'head_size',
        'data_type',    'data_size',    'data_mode',
        'image_width',  'image_height', 'image_channels',
        'res1',         'res2',         'res3',
        'res1_unit',    'res2_unit',    'res3_unit',
        'depth',        'depth_w',      'depth_h',
        'zero_channel', 'voltage',      'options'
    ]
)

FILE_MAGIC = b'ity\x00'

def parse_cmdargs():
    parser = argparse.ArgumentParser(description='Unpack ITY file')

    parser.add_argument(
        'source',
        metavar = 'INPUT',
        help    = 'Path to input .ity file'
    )

    parser.add_argument(
        'out_root',
        metavar = 'OUTPUT',
        help    = 'Root path for output files'
    )

    return parser.parse_args()

def parse_header(f):
    # “ity” //4 bytes
    magic = struct.unpack("<4s", f.read(4))[0]
    if magic != FILE_MAGIC:
        raise RuntimeError(f'File is not ITY.')

    # WORD    wDataType, wDataSize; //wDataSize: e.g. float=4
    wVersion  = struct.unpack("<H", f.read(2))[0]
    wHeadSize = struct.unpack("<H", f.read(2))[0]

    # WORD    wDataType, wDataSize; //wDataSize: e.g. float=4
    wDataType = struct.unpack("<H", f.read(2))[0]
    wDataSize = struct.unpack("<H", f.read(2))[0]

    # int nWidth,nHeight;
    # // int is 4 bytes integer, width and height of the image in pixels
    nWidth  = struct.unpack("<i", f.read(4))[0]
    nHeight = struct.unpack("<i", f.read(4))[0]

    # DWORD   dwOffData;
    # //DWORD is 4 bytes unsigned integer, address of image data
    dwOffData = struct.unpack("<I", f.read(4))[0]

    # WORD    wChannel; //1 (grey image) or 4 (BGRA) for color
    wChannel = struct.unpack("<H", f.read(2))[0]

    # WORD    wResUnit, wResUnit2, wResUnit3; //DI_CM...
    wResUnit1 = struct.unpack("<H", f.read(2))[0]
    wResUnit2 = struct.unpack("<H", f.read(2))[0]
    wResUnit3 = struct.unpack("<H", f.read(2))[0]

    # int     nDepth, nDepWd, nDepHt;
    nDepth = struct.unpack("<i", f.read(4))[0]
    nDepWd = struct.unpack("<i", f.read(4))[0]
    nDepHt = struct.unpack("<i", f.read(4))[0]

    # float   f_res, f_res2, f_res3;
    f_res1 = struct.unpack("<f", f.read(4))[0]
    f_res2 = struct.unpack("<f", f.read(4))[0]
    f_res3 = struct.unpack("<f", f.read(4))[0]

    # float   zero_channel, hv;
    zero_channel = struct.unpack("<f", f.read(4))[0]
    hv           = struct.unpack("<f", f.read(4))[0]

    # int     i_dim;
    i_dim = struct.unpack("<i", f.read(4))[0]

    # int     i_opt;
    i_opt = struct.unpack("<i", f.read(4))[0]

    header = ITYHeader(
        version        = wVersion,
        head_size      = wHeadSize,
        data_type      = DATA_TYPES[wDataType],
        data_size      = wDataSize,
        data_mode      = DATA_MODES[i_dim],
        image_width    = nWidth,
        image_height   = nHeight,
        image_channels = wChannel,
        res1           = f_res1,
        res2           = f_res2,
        res3           = f_res3,
        res1_unit      = wResUnit1,
        res2_unit      = wResUnit2,
        res3_unit      = wResUnit3,
        depth          = nDepth,
        depth_w        = nDepWd,
        depth_h        = nDepHt,
        zero_channel   = zero_channel,
        voltage        = hv,
        options        = i_opt,
    )

    return header, dwOffData

def read_image(f, header):
    # assuming (H, W, C) layout
    image_size = (
          header.image_channels
        * header.image_width
        * header.image_height
        * header.data_size
    )

    # result : (H * W * C)
    data_bytes = f.read(image_size)
    result     = np.frombuffer(data_bytes, dtype = np.dtype(header.data_type))

    result = result.reshape(
        (header.image_height, header.image_width, header.image_channels)
    )

    return result

def estimate_number_of_images(f, header, data_offset):
    image_size = (
          header.image_channels
        * header.image_width
        * header.image_height
        * header.data_size
    )

    f.seek(0, os.SEEK_END)

    file_size = f.tell()
    data_size = file_size - data_offset

    n1, r = divmod(data_size, image_size)

    if r != 0:
        print('[WARNING] Data section size is not a multiple of image size')


    n2 = header.depth
    if header.data_mode == 'stem_4d':
        n2 += 1

    if n1 != n2:
        print(
            f"[WARNING] Direct number of images calculation '{n1}'"
            f" does not match the number of images from the format spec '{n2}'"
        )

    return n1

def save_image(image, path):
    np.savez_compressed(path, image)

def save_header(header, outdir):
    path = outdir.rstrip('/') + '_header.json'

    with open(path, 'wt', encoding = 'utf-8') as f:
        json.dump(header._asdict(), f, indent = 4)

def main():
    cmdargs = parse_cmdargs()

    outdir = os.path.join(cmdargs.out_root, os.path.basename(cmdargs.source))
    os.makedirs(outdir, exist_ok = True)

    with open(cmdargs.source, 'rb') as f:
        header, data_offset = parse_header(f)
        n_images = estimate_number_of_images(f, header, data_offset)

        save_header(header, outdir)

        f.seek(data_offset)

        for idx in tqdm.tqdm(range(n_images), desc = 'Extracting'):
            image = read_image(f, header)
            path  = os.path.join(outdir, f'sample_{idx:05d}.npz')
            save_image(image, path)


if __name__ == '__main__':
    main()

