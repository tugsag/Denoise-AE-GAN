import re
import argparse
import glob
import numpy as np
import cv2
import tqdm




def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def chunk_im(impaths, dest='chunked', h=200, w=200):
    # ims is list of images
    # TODO: make sure iso200 images were chunked first
    for p in tqdm.tqdm(impaths):
        count = 0
        im = cv2.imread(p)
        tag = p.split('/')[-1].split('.')[0]
        x, y = im.shape[:2]
        for i in range(0, x, h):
            for j in range(0, y, w):
                sub = im[i:i+h, j:j+w, :]
                if sub.shape == (h, w, 3):
                    # don't pad

                    # sh, sw = sub.shape[:2]
                    # sub = np.pad(sub, ((0, h-sh), (0, w-sw), (0, 0)), 'constant', constant_values=0)
                    # assert sub.shape == (h, w, 3)
                    cv2.imwrite(f'{dest}/{tag}_{count}.png', sub)
                    count += 1

def join_im(over_shape, chunks):
    # over_shape is (h, w)
    # chunks is list of ndarrays
    chunk_shape = chunks[0].shape[:2]
    h = int(np.ceil(over_shape[0]/chunk_shape[0]))
    w = int(np.ceil(over_shape[1]/chunk_shape[1]))
    rows = []
    for i in range(0, len(chunks), w):
        rows.append(np.concatenate(chunks[i:i+w], axis=1))
    return np.vstack(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help="size of chunked images", default=500)
    args = parser.parse_args()
    paths = glob.glob('NIND/*/*', recursive=True)
    chunk_im(paths[:len(paths)//5], h=args.s, w=args.s)