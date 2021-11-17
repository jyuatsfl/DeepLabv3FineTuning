import os
from io import BytesIO
import numpy as np
from PIL import Image
import argparse
import h5py
import cv2
import logging
from tqdm import tqdm
from inference import DeepLabV3 as DeepLabV3Retrained
from inference_rawmodel import DeepLabV3 as DeepLabV3Raw

from libs_internal.io.h5 import (load_camera_data,
                                 load_h5_group_names, load_h5_group_vals)
from mask_utils import (refine_mask, remove_noise, convert_image_color,
                        image_size_conversion, image_2Dto3D, keypoint_seg_refine)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLabModel(object):
    """
      Class to load deeplab model (retrained or original) and run inference.
    """

    def __init__(self, config):
        """Creates and loads pretrained deeplab model."""
        self.input_h5 = config.get("input_h5", "PreProcessing.h5")
        self.keyp_h5 = config.get("keyp_h5", "TwoDKeyPointsTransform.h5")
        self.output_h5 = config.get("output_h5", "BgSegmentation.h5")
        # for retraining, using the model filename
        self.modelfile = config.get(
            "modelfile", "weights_deeplabv3_retrain_coco_width480_height640.pt")
        # for original model, use "deeplabv3_resnet50", or "deeplabv3_resnet101",
        # or "deeplabv3_mobilenet_v3_large"
        # whether or not making visulization into output
        self.viz = config.get("viz", True)
        retrained = config.get("retrained", True)
        if retrained:
            # width and height are used for model retraining
            width = config.get("width", 480)
            height = config.get("height", 640)
            self.model = DeepLabV3Retrained(self.modelfile, width, height)
        else:
            self.model = DeepLabV3Raw(self.modelfile)

    def _load_h5_files(self):
        assert os.path.exists(self.keyp_h5), "%s not found" % self.keyp_h5
        frames = load_h5_group_names(self.keyp_h5)
        assert len(frames) > 0, "No frames are found! Check %s!" % self.keyp_h5
        kpslist = load_h5_group_vals(self.keyp_h5, 'data', frames)

        assert os.path.exists(self.input_h5), "%s not found" % self.input_h5
        imglist = load_h5_group_vals(self.input_h5, 'data', frames)
        # image shape is (height, width, 3), needed for saving video!!
        self.img_height, self.img_width, _ = imglist[0].shape

        with h5py.File(self.keyp_h5, 'r') as h5:
            assert "info" in h5 and "twoD_keypoints" in h5["info"]
            # class name: class index dict
            self.keyp_class_dict = {
                name.decode('ascii'): i for i, name in
                enumerate(np.array(h5["info"]["twoD_keypoints"]))
            }
        self.frame_dict = {name: img for name, img in zip(frames, imglist)}
        self.keyp_dict = {name: kps for name, kps in zip(frames, kpslist)}

    def run(self):
        """
        Run the background vs person segmentation
        """
        # load the input files into memory
        self._load_h5_files()
        mask_dict = {}
        # looping over the frames and do background segmentation inferences
        for name, frame in self.frame_dict.items():
            keyp = self.keyp_dict[name]
            # crop the frame according to the 2D keypoints?
            # frame = self.crop(frame, keyp)
            # call the corresponding model class's function 
            # masks are True or False
            mask = self.model.segment_image(frame)
            # refine mask expects the masks with 0's and 255's
            # mask = self.refine(mask, frame)
            mask_dict[name] = mask

        self.save(mask_dict)

    def save(self, mask_dict):
        """
        given the output masks after running the segmentation, 
        Save the masks to output file!
        """
        with h5py.File(self.output_h5, 'w') as f:
            group = f.create_group("info")
            dset = group.create_dataset(name=np.string_('BgSegmentation'),
                                        data=[np.string_('boolean')],
                                        compression="gzip",
                                        compression_opts=9)
            group = f.create_group("data")

            if self.viz:
                path, filename = os.path.split(self.output_h5)
                filename = filename.split(".")[0]+'.mp4'
                video = cv2.VideoWriter(os.path.join(path, filename),
                    cv2.VideoWriter_fourcc(*'mp4v'), 20,
                    (self.img_width, self.img_height)
                    )
            for name, mask in tqdm(mask_dict.items()):
                if self.viz:
                    alpha = .5
                    img_ = cv2.UMat(
                        np.array(self.frame_dict[name], dtype=np.uint8))
                    height, width = self.frame_dict[name].shape[0:2]
                    mask_ = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_[:, :, 0] = mask
                    mask_[:, :, 1] = mask
                    mask_[:, :, 2] = mask
                    dst_ = cv2.addWeighted(mask_, alpha, img_, 1 - alpha, 0)
                    video.write(dst_)
                dset = group.create_dataset(
                    name=(name),
                    data=np.array(mask, dtype=bool),
                    shape=(height, width),
                    maxshape=(height, width),
                    compression="gzip",
                    compression_opts=9,
                    dtype='float32')
            if self.viz:
                video.release()

    # def crop(self, img, keypoints):
    #     """
    #     Given the initial image an 2D keypoints, return the cropped image to
    #     Just cover the original person.
    #     """
    #     return img

    def refine(self, mask, img):
        """
        refine the mask using grabcut and noise removal given the initial
        image.
        Parameters:
            mask: np.array, initial mask
            img: np.array, initial image
        Returns:
            mask: np.array, after refinement
        """
        mask = refine_mask(np.array(mask, dtype=np.uint8), img)
        # removing noise only whan refine grabcut is already done
        # due to a mask class limitation
        mask = remove_noise(mask)
        return mask


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-hdf5', '-i', type=str,
        help="Input HDF5 file that contains images")
    parser.add_argument(
        '--keyp-hdf5', '-k', type=str,
        help="2D Key Points HDF5 file with key points")
    parser.add_argument(
        '--output-hdf5', '-o', type=str,
        help="Output HDF5 file contain masks")
    parser.add_argument(
        '--modelfile', '-m', type=str,
        help="Output HDF5 file contain masks")
    parser.add_argument(
        '--viz', action='store_true',
        help="Save visualization video")
    parser.add_argument(
        '--retrained', action='store_true', 
        help='true for retrained DeepLabV3 model, false raw model')
    args = parser.parse_args()
    return {
        "input_h5": args.input_hdf5,
        "keyp_h5": args.keyp_hdf5,
        "output_h5": args.output_hdf5,
        "viz": args.viz,
        "retrained": args.retrained,
        "modelfile": args.modelfile,
    }


def main():
    config = get_arguments()
    logger.info("Running with Config: %s" % str(config))
    dv = DeepLabModel(config)
    dv.run()


if __name__ == '__main__':
    main()
