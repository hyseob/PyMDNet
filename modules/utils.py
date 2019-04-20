from scipy.misc import imresize
import numpy as np
import cv2


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def crop_image2(img, bbox, img_size=107, padding=16, flip=False, rotate_limit=0, blur_limit=0):
    x, y, w, h = np.array(bbox, dtype='float32')

    cx, cy = x + w/2, y + h/2

    if padding > 0:
        w += 2 * padding * w/img_size
        h += 2 * padding * h/img_size

    # List of transformation matrices
    matrices = []

    # Translation matrix to move patch center to origin
    translation_matrix = np.asarray([[1, 0, -cx],
                                     [0, 1, -cy],
                                     [0, 0, 1]], dtype=np.float32)
    matrices.append(translation_matrix)

    # Scaling matrix according to image size
    scaling_matrix = np.asarray([[img_size / w, 0, 0],
                                 [0, img_size / h, 0],
                                 [0, 0, 1]], dtype=np.float32)
    matrices.append(scaling_matrix)

    # Define flip matrix
    if flip and np.random.binomial(1, 0.5):
        flip_matrix = np.eye(3, dtype=np.float32)
        flip_matrix[0, 0] = -1
        matrices.append(flip_matrix)

    # Define rotation matrix
    if rotate_limit and np.random.binomial(1, 0.5):
        angle = np.random.uniform(-rotate_limit, rotate_limit)
        alpha = np.cos(np.deg2rad(angle))
        beta = np.sin(np.deg2rad(angle))
        rotation_matrix = np.asarray([[alpha, -beta, 0],
                                      [beta, alpha, 0],
                                      [0, 0, 1]], dtype=np.float32)
        matrices.append(rotation_matrix)

    # Translation matrix to move patch center from origin
    revert_t_matrix = np.asarray([[1, 0, img_size / 2],
                                  [0, 1, img_size / 2],
                                  [0, 0, 1]], dtype=np.float32)
    matrices.append(revert_t_matrix)

    # Aggregate all transformation matrices
    matrix = np.eye(3)
    for m_ in matrices:
        matrix = np.matmul(m_, matrix)

    # Warp image, padded value is set to 128
    patch = cv2.warpPerspective(img,
                                matrix,
                                (img_size, img_size),
                                borderValue=128)

    if blur_limit and np.random.binomial(1, 0.5):
        blur_size = np.random.choice(np.arange(1, blur_limit + 1, 2))
        patch = cv2.GaussianBlur(patch, (blur_size, blur_size), 0)

    return patch


def crop_image(img, bbox, img_size=107, padding=16, valid=False):
    # This function is deprecated in favor of crop_image2

    x,y,w,h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled = imresize(cropped, (img_size, img_size))
    return scaled
