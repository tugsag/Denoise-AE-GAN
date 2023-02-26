import cv2
import os
import numpy as np
import re
import argparse



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def chunk_im(impaths, h=200, w=200):
    # ims is list of images
    count = 0
    for p in impaths:
        im = cv2.imread(p)
        tag = p.split('/')[-1].split('.')[0]
        base = 'noisy' if 'noisy' in tag else 'clean'
        x, y = im.shape[:2]
        for i in range(0, x, h):
            for j in range(0, y, w):
                sub = im[i:i+h, j:j+w, :]
                if sub.shape != (h, w, 3):
                    sh, sw = sub.shape[:2]
                    sub = np.pad(sub, ((0, h-sh), (0, w-sw), (0, 0)), 'constant', constant_values=0)
                assert sub.shape == (h, w, 3)
                cv2.imwrite(f'{base}/{tag}_{count}.png', sub)
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', default=all)
    # args = parser.parse_args()
    clean = [os.path.join('data/clean', i) for i in os.listdir('data/clean')]
    noisy = [os.path.join('data/noisy', i) for i in os.listdir('data/noisy')]
    chunk_im(clean)
    chunk_im(noisy)