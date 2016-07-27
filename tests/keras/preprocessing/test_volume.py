import pytest
from keras.preprocessing.volume import *
import numpy as np
import os
import shutil
import tempfile
import SimpleITK as sitk

class TestVolume:

    def setup_class(cls):
        img_w = img_h = img_d = 32
        gray_images = []
        # set up some random test volumes
        for n in range(8):
            bias = np.random.rand(img_w, img_h, img_d, 1) * 64
            variance = np.random.rand(img_w, img_h, img_d, 1) * (255-64)
            imarray = np.random.rand(img_w, img_h, img_d, 1) * variance + bias
            im = sitk.GetImageFromArray(imarray.astype('uint8').squeeze())
            gray_images.append(im)

        cls.all_test_images = [gray_images]

    def teardown_class(cls):
        del cls.all_test_images

    def test_volume_data_generator(self):
        for test_images in self.all_test_images:
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])
           
            images = np.vstack(img_list)
            generator = VolumeDataGenerator(
                rescale=1./255.,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range_xy=10.,
                rotation_range_yz=10.,
                rotation_range_xz=10.,
                width_shift_range=0.2,
                height_shift_range=0.2,
                depth_shift_range=0.2,
                shear_range=0.,
                zoom_range=0.5,#(0.,0.05),
                channel_shift_range=0.5,
                fill_mode='constant',
                cval=0.,
                horizontal_flip=False,
                vertical_flip=False)
            generator.fit(images, rounds=3, augment=True)

            tmp_folder = tempfile.mkdtemp(prefix='test_images')
            for x, y in generator.flow(images, np.arange(images.shape[0]),
                                       shuffle=True, save_to_dir=tmp_folder):
                # x is a list
                assert x[0].shape[1:] == images.shape[1:]
                break
            shutil.rmtree(tmp_folder)

if __name__ == '__main__':
    pytest.main([__file__])
