'''
Basic set of tools for real-time data augmentation on volume data using SimpleITK wrappers.
'''
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading

try:
    import SimpleITK as sitk
except ImportError:
    raise

from .. import backend as K


def random_rotation(x, rg_xy, rg_yz, rg_xz, depth_index=1, row_index=2, col_index=3, channel_index=0,
                    fill_mode='nearest', cval=0.):
    if rg_xy:
        theta = np.pi / 180 * np.random.uniform(-rg_xy, rg_xy)
        rotation_matrix_xy = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                       [np.sin(theta), np.cos(theta), 0, 0],
                                       [0, 0, 1, 0], 
                                       [0, 0, 0, 1]])
    else:
        rotation_matrix_xy = np.eye(4)
    
    if rg_yz:
        theta = np.pi / 180 * np.random.uniform(-rg_yz, rg_xz)
        rotation_matrix_yz = np.array([[1, 0, 0, 0],
                                       [0, np.cos(theta), -np.sin(theta), 0],
                                       [0, np.sin(theta), np.cos(theta), 0], 
                                       [0, 0, 0, 1]])
    else:
        rotation_matrix_yz = np.eye(4)
    
    if rg_xz:
        theta = np.pi / 180 * np.random.uniform(-rg_xz, rg_xz)
        rotation_matrix_xz = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                                       [0, 1, 0, 0],
                                       [-np.sin(theta), 0, np.cos(theta), 0], 
                                       [0, 0, 0, 1]])
    else:
        rotation_matrix_xz = np.eye(4)

    # 3D rotation matrix with separate angles
    rotation_matrix = np.dot(np.dot(rotation_matrix_xy,rotation_matrix_yz),rotation_matrix_xz)
    
    d, h, w = x.shape[depth_index], x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, d, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shift(x, drg, wrg, hrg, depth_index=1, row_index=2, col_index=3, channel_index=0,
                 fill_mode='nearest', cval=0.):
    d, h, w = x.shape[depth_index], x.shape[row_index], x.shape[col_index]
    tz = np.random.uniform(-drg, drg) * d
    ty = np.random.uniform(-hrg, hrg) * h
    tx = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, 0, tx],
                                   [0, 1, 0, ty],
                                   [0, 0, 1, tz],
                                   [0, 0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_shear(x, intensity, depth_index=1, row_index=2, col_index=3, channel_index=0,
                 fill_mode='nearest', cval=0.):
    raise NotImplementedError('random_shear 3D is not yet implemented!')
#     shear = np.random.uniform(-intensity, intensity)
#     shear_matrix = np.array([[1, -np.sin(shear), 0],
#                              [0, np.cos(shear), 0],
#                              [0, 0, 1]])
# 
#     h, w = x.shape[row_index], x.shape[col_index]
#     transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
#     x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
#     return x


def random_zoom(x, zoom_range, depth_index=1, row_index=2, col_index=3, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zz = zy = zx = 1
    else:
        zz, zy, zx = np.random.uniform(zoom_range[0], zoom_range[1], 3)
        
        # 3D zooming
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, zz],
                                [0, 0, 1]])

    d, h, w = x.shape[depth_index], x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, d, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, intensity, channel_index=0):
    '''
    Random intensity shift (uniform distribution)
    '''
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, z, y, x):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
    reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, 3]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    raise NotImplementedError('flip_axis 3D is not yet implemented!')
#     x = np.asarray(x).swapaxes(axis, 0)
#     x = x[::-1, ...]
#     x = x.swapaxes(0, axis)
#     return x


def array_to_img(x, dim_ordering=K.image_dim_ordering(), scale=False):
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 3, 0)
    
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[3] == 3:
        # RGB
        raise NotImplementedError('Multi-channel 3D images not yet supported!')
    elif x.shape[3] == 1:
        # grayscale
        img = np_to_itk(x)
    else:
        raise Exception('Unsupported channel number: ', x.shape[3])

    return img


def img_to_array(img, dim_ordering=K.image_dim_ordering()):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (depth, height, width, channels)
    print(img.GetSize())
    x = itk_to_np(img)
    x = np.asarray(x, dtype='float32')
    print(x.shape)
    if len(x.shape) == 4: 
        if dim_ordering == 'th':
            x = x.transpose(3, 0, 1, 2)
    elif len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def np_to_itk(np_img):
    '''
    Converts a numpy array to a SimpleITK image
    '''
    return sitk.GetImageFromArray(np_img)


def itk_to_np(itk_img):
    '''
    Converts a SimpleITK image to numpy array
    '''
    return sitk.GetArrayFromImage(itk_img)


def load_img(path, target_size=None):
    '''
    Load a (volume) from the path
    # Arguments
        path: the path of the image
        target_size: the target size (px) of the volume
        
    # Returns
        The image as an instance of SimpleITK.SimpleITK.Image
    '''
    img = sitk.ReadImage(path)
    
    # TODO resize the image to the target size
    if target_size is not None:
        raise NotImplementedError('Using target_size is not yet implemented!')
    
    return img


def list_pictures(directory, ext='nii|nii.gz|mha|mhd|dicom|dcm|hdr|img.gz'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


class VolumeDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range_xy: degrees (0 to 180).
        rotation_range_yz: degrees (0 to 180).
        rotation_range_xz: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        depth_shift_range: fraction of total depth.
        (unimplemented) shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select the range for each dimension.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        (unimplemented) horizontal_flip: whether to randomly flip images horizontally.
        (unimplemented) vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range_xy=0.,
                 rotation_range_yz=0.,
                 rotation_range_xz=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 depth_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 dim_ordering=K.image_dim_ordering()):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (z, y, x, channels) '
                            'or "th" (channels, z, y, x). '
                            'Received arg: ', dim_ordering)

        self.dim_ordering = dim_ordering
        # th: (samples, channels, z, y, x)
        # tf: (samples, z, y, x, channels)
        if dim_ordering == 'th':
            self.channel_index = 1
            self.depth_index = 2
            self.row_index = 3
            self.col_index = 4
        if dim_ordering == 'tf':
            self.channel_index = 4
            self.depth_index = 1
            self.row_index = 2
            self.col_index = 3


        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='mha'):
        '''
        Flow from numpy array iterator.
        '''
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def flow_from_directory(self, directory,
                            target_size=(32, 32, 32), color_mode='grayscale',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='mha'):
        raise NotImplementedError('flow_from_directory is not yet implemented!')
        # TODO implement
#         return DirectoryIterator(
#             directory, self,
#             target_size=target_size, color_mode=color_mode,
#             classes=classes, class_mode=class_mode,
#             dim_ordering=self.dim_ordering,
#             batch_size=batch_size, shuffle=shuffle, seed=seed,
#             save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


    def standardize(self, x):
        '''
        Standardize the feature range of a single 3D image x
        '''
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

        return x

    def random_transform(self, x):
        #print(x.shape)
        #print(locals())
        # x is a single image, so it doesn't have image number at index 0
        img_depth_index = self.depth_index - 1
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        
        print(img_depth_index,img_row_index,img_col_index,img_channel_index)

        # use composition of homographies to generate final transform that needs to be applied
        ####################################### rotation in XY, YZ, XZ separately
        if self.rotation_range_xy:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range_xy, self.rotation_range_xy)
            rotation_matrix_xy = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                           [np.sin(theta), np.cos(theta), 0, 0],
                                           [0, 0, 1, 0], 
                                           [0, 0, 0, 1]])
        else:
            rotation_matrix_xy = np.eye(4)
        
        if self.rotation_range_yz:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range_yz, self.rotation_range_xz)
            rotation_matrix_yz = np.array([[1, 0, 0, 0],
                                           [0, np.cos(theta), -np.sin(theta), 0],
                                           [0, np.sin(theta), np.cos(theta), 0], 
                                           [0, 0, 0, 1]])
        else:
            rotation_matrix_yz = np.eye(4)
        
        if self.rotation_range_xz:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range_xz, self.rotation_range_xz)
            rotation_matrix_xz = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                                           [0, 1, 0, 0],
                                           [-np.sin(theta), 0, np.cos(theta), 0], 
                                           [0, 0, 0, 1]])
        else:
            rotation_matrix_xz = np.eye(4)

        # 3D rotation matrix with separate angles
        rotation_matrix = np.dot(np.dot(rotation_matrix_xy,rotation_matrix_yz),rotation_matrix_xz)
        
        ####################################### translation in z,y,x separately
        if self.depth_shift_range:
            tz = np.random.uniform(-self.depth_shift_range, self.depth_shift_range) * x.shape[img_depth_index]
        else:
            tz = 0
            
        if self.height_shift_range:
            ty = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            ty = 0

        if self.width_shift_range:
            tx = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            tx = 0
            
        # 3D translation
        translation_matrix = np.array([[1, 0, 0, tx],
                                       [0, 1, 0, ty],
                                       [0, 0, 1, tz],
                                       [0, 0, 0, 1 ]])
        
#         if self.shear_range:
#             shear = np.random.uniform(-self.shear_range, self.shear_range)
#         else:
#             shear = 0
#         shear_matrix = np.array([[1, -np.sin(shear), 0],
#                                  [0, np.cos(shear), 0],
#                                  [0, 0, 1]])
        # TODO implement 3D shearing
        shear_matrix = np.eye(4)


        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zz = zy = zx = 1
        else:
            zz, zy, zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 3)
        
        # 3D zooming
        zoom_matrix = np.array([[zx, 0, 0, 0],
                                [0, zy, 0, 0],
                                [0, 0, zz, 0],
                                [0, 0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        d, h, w = x.shape[img_depth_index], x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, d, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        
        # channel shift range
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

#         if self.horizontal_flip:
#             if np.random.random() < 0.5:
#                 x = flip_axis(x, img_col_index)
# 
#         if self.vertical_flip:
#             if np.random.random() < 0.5:
#                 x = flip_axis(x, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3] * X.shape[4]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            # return the iterator object
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering=K.image_dim_ordering(),
                 save_to_dir=None, save_prefix='', save_format='mha'):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                # convert image to sitk
                img = array_to_img(batch_x[i], self.dim_ordering, scale=False)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                sitk.WriteImage(img,os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

# TODO
# class DirectoryIterator(Iterator):
# 
#     def __init__(self, directory, image_data_generator,
#                  target_size=(32, 32, 32), color_mode='grayscale',
#                  dim_ordering=K.image_dim_ordering,
#                  classes=None, class_mode='categorical',
#                  batch_size=32, shuffle=True, seed=None,
#                  save_to_dir=None, save_prefix='', save_format='mha'):
#         self.directory = directory
#         self.image_data_generator = image_data_generator
#         self.target_size = tuple(target_size)
#         if color_mode not in {'rgb', 'grayscale'}:
#             raise ValueError('Invalid color mode:', color_mode,
#                              '; expected "rgb" or "grayscale".')
#         self.color_mode = color_mode
#         self.dim_ordering = dim_ordering
#         if self.color_mode == 'rgb':
#             if self.dim_ordering == 'tf':
#                 self.image_shape = self.target_size + (3,)
#             else:
#                 self.image_shape = (3,) + self.target_size
#         else:
#             if self.dim_ordering == 'tf':
#                 self.image_shape = self.target_size + (1,)
#             else:
#                 self.image_shape = (1,) + self.target_size
#         self.classes = classes
#         if class_mode not in {'categorical', 'binary', 'sparse', None}:
#             raise ValueError('Invalid class_mode:', class_mode,
#                              '; expected one of "categorical", '
#                              '"binary", "sparse", or None.')
#         self.class_mode = class_mode
#         self.save_to_dir = save_to_dir
#         self.save_prefix = save_prefix
#         self.save_format = save_format
# 
#         white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
# 
#         # first, count the number of samples and classes
#         self.nb_sample = 0
# 
#         if not classes:
#             classes = []
#             for subdir in os.listdir(directory):
#                 if os.path.isdir(os.path.join(directory, subdir)):
#                     classes.append(subdir)
#         self.nb_class = len(classes)
#         self.class_indices = dict(zip(classes, range(len(classes))))
# 
#         for subdir in classes:
#             subpath = os.path.join(directory, subdir)
#             for fname in os.listdir(subpath):
#                 is_valid = False
#                 for extension in white_list_formats:
#                     if fname.lower().endswith('.' + extension):
#                         is_valid = True
#                         break
#                 if is_valid:
#                     self.nb_sample += 1
#         print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))
# 
#         # second, build an index of the images in the different class subfolders
#         self.filenames = []
#         self.classes = np.zeros((self.nb_sample,), dtype='int32')
#         i = 0
#         for subdir in classes:
#             subpath = os.path.join(directory, subdir)
#             for fname in os.listdir(subpath):
#                 is_valid = False
#                 for extension in white_list_formats:
#                     if fname.lower().endswith('.' + extension):
#                         is_valid = True
#                         break
#                 if is_valid:
#                     self.classes[i] = self.class_indices[subdir]
#                     self.filenames.append(os.path.join(subdir, fname))
#                     i += 1
#         super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
# 
#     def next(self):
#         with self.lock:
#             index_array, current_index, current_batch_size = next(self.index_generator)
#         # The transformation of images is not under thread lock so it can be done in parallel
#         batch_x = np.zeros((current_batch_size,) + self.image_shape)
#         grayscale = self.color_mode == 'grayscale'
#         # build batch of image data
#         for i, j in enumerate(index_array):
#             fname = self.filenames[j]
#             img = load_img(os.path.join(self.directory, fname), grayscale=grayscale, target_size=self.target_size)
#             x = img_to_array(img, dim_ordering=self.dim_ordering)
#             x = self.image_data_generator.random_transform(x)
#             x = self.image_data_generator.standardize(x)
#             batch_x[i] = x
#         # optionally save augmented images to disk for debugging purposes
#         if self.save_to_dir:
#             for i in range(current_batch_size):
#                 img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
#                 fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                   index=current_index + i,
#                                                                   hash=np.random.randint(1e4),
#                                                                   format=self.save_format)
#                 img.save(os.path.join(self.save_to_dir, fname))
#         # build batch of labels
#         if self.class_mode == 'sparse':
#             batch_y = self.classes[index_array]
#         elif self.class_mode == 'binary':
#             batch_y = self.classes[index_array].astype('float32')
#         elif self.class_mode == 'categorical':
#             batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
#             for i, label in enumerate(self.classes[index_array]):
#                 batch_y[i, label] = 1.
#         else:
#             return batch_x
#         return batch_x, batch_y
