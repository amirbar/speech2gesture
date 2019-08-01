from audio_to_multiple_pose_gan.config import create_parser
from common.utils import set_seeds
set_seeds()
import matplotlib
matplotlib.use('Agg')
import argparse
from audio_to_multiple_pose_gan.model import PoseGAN

parser = argparse.ArgumentParser(description='train speaker specific model')
parser = create_parser(parser)
args = parser.parse_args()

def main():
    args.mode = 'train'
    pgan = PoseGAN(args)
    pgan.train(args.epochs)

if __name__ == "__main__":
    main()
