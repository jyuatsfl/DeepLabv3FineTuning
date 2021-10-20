import os
import cv2
import glob
import torch
import argparse
import logging
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLabV3:
    def __init__(self, model_path, width, height):
        # model weights.pt
        assert os.path.exists(model_path), "Model not found."
        self.model = torch.load(model_path)
        self.model.eval()
        self.width = width
        self.height = height

    def segment(self, image_path, threshold=0.5):
        """
        Make 1 image segmentation inference with the given model.
        """
        assert os.path.exists(image_path), "%s not found!" % image_path
        img = cv2.imread(image_path)
        # image shape is height, width, 3
        rawimgshape = img.shape[0:2]
        if rawimgshape != (args.height, args.width):
            img = cv2.resize(img, (args.width, args.height))

        img = img.transpose(2, 0, 1).reshape(1, 3, self.height, self.width)
        with torch.no_grad():
            pr = self.model(torch.from_numpy(img).true_divide(
                255).type(torch.cuda.FloatTensor))
            # pr['out'] has shape like: (1, 1, 480, 640)
            mask = pr['out'].cpu().detach().numpy()[0][0]
            if mask.shape[0:2] != rawimgshape:
                logger.info("image resizing to raw")
                # resize takes shape (width, height, opposite of the shape)
                mask = cv2.resize(mask, rawimgshape[::-1])
            return (mask > threshold)


def main(args):
    os.makedirs(args.outputfolder, exist_ok=True)
    dv = DeepLabV3(args.modelpath, args.width, args.height)
    imgnames = glob.glob(args.inputfolder + "/*.png")
    for idx, fullname in enumerate(imgnames):
        basename = os.path.basename(fullname)
        # mask from segmentation is True or False values in each cell
        mask = dv.segment(fullname)
        img = cv2.imread(fullname).astype("uint8")
        # convert person, mask = True to white [255, 255, 255]
        # and background, mask = False to black [0, 0, 0]
        img[mask] = [255, 255, 255]
        img[~mask] = [0, 0, 0]
        outname = os.path.join(args.outputfolder, basename)
        cv2.imwrite(outname, img)
        if idx % 100 == 0:
            logger.info("%04d, Initial image: %s" % (idx, fullname))
            logger.info("%04d, Converted image: %s" % (idx, outname))
    logger.info("Total images: %d" % (idx+1))


if __name__ == '__main__':
    """
    To run, it needs input indexed images folder and output folder:
        python3 inference.py -i /source/images/folder \
                -o /resized/images/folder \
                -m /full/path/model/weight.pt \
                [opt: --width 480 --height 640]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelpath", "-m", default=None, type=str, required=True,
        help="Your model path for making inferences.",
    )
    parser.add_argument(
        "--inputfolder", "-i", default=None, type=str, required=True,
        help="Your inputfolder containing all Pinocchio run outputs.",
    )
    parser.add_argument(
        "--outputfolder", "-o", default=None, type=str, required=True,
        help="Your output folder.",
    )
    parser.add_argument(
        "--height", default=640, type=int, required=False,
        help="Your target image height, default=640.",
    )
    parser.add_argument(
        "--width", default=480, type=int, required=False,
        help="Your target image width, default=480.",
    )
    args = parser.parse_args()

    main(args)
