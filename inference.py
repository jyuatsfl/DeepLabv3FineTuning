import torch
import cv2

class DeepLabV3:
    def __init__(self, model_path, width, height):
        # weights.pt
        self.model = torch.load(model_path)
        self.model.eval()
        self.width = width
        self.height = height

    def inference(self, image_path, threshold=0.5):
        """
        Make 1 image inference
        """

        img = cv2.imread(image_path).transpose(2, 0, 1).reshape(
            1, 3, self.height, self.width)
        with torch.no_grad():
            pr = self.model(torch.from_numpy(img) / 255)
            # .type(torch.cuda.FloatTensor)/255)
            return (pr['out'].numpy()[0][0] > threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfolder", "-i", default=None, type=str, required=True,
        help="Your inputfolder containing all Pinocchio run outputs.",
    )
    parser.add_argument(
        "--outputfolder", "-o", default=None, type=str, required=True,
        help="Your output folder.",
    )
    parser.add_argument(
        "--type", "-t", default=None, type=str, required=True,
        help="Your image type: mask OR image",
    )
    parser.add_argument(
        "--height", default=640, type=int, required=False,
        help="Your target image height, default=640.",
    )
    parser.add_argument(
        "--width", default=480, type=int, required=False,
        help="Your target image width, default=480.",
    )
    parser.add_argument(
        "--scale", default="resize", type=str, required=False,
        help="Your scale mode to match the required size: resize or pad",
    )
    args = parser.parse_args()
    imgtype = args.type.lower()
    assert imgtype in ["mask", "image"], "image type is mask or image"
    os.makedirs(args.outputfolder, exist_ok=True)
    imgnames = glob.glob(args.inputfolder + "/*.png")
    for idx, name in enumerate(imgnames):
        basename = os.path.basename(name)
        img = cv2.imread(name)
        if imgtype == "mask":
            # background: 84  1 68 to 0, 0, 0
            # human: 36 231 253, 255, 255, 255
            img = convert_image_color(img, [84, 1, 68], [0, 0, 0])
            img = convert_image_color(img, [36, 231, 253], [255, 255, 255])
        img = pad_image_size(img, args.height, args.width, imgtype,
                             mode=args.scale)
        outname = os.path.join(args.outputfolder, basename)
        cv2.imwrite(outname, img)
        if idx % 100 == 0:
            logger.info("%04d, Initial image: %s" % (idx, name))
            logger.info("%04d, Converted image: %s" % (idx, outname))
    logger.info("Total images: %d" % (idx+1))


if __name__ == '__main__':
    """
    To run, it needs input indexed images folder and output folder:
        python3 image_conversion.py -i /source/images/folder \
                -o /resized/images/folder -t mask [OR: image] \
                [opt: --width 480 --height 640]
    """
    main()
