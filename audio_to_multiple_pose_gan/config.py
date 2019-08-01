from common.consts import AUDIO_SHAPE


def create_parser(parser):
    parser.add_argument('-lg', '--lambda_gan', type=float, default='1.', help='multiplier for the GAN loss (versus the regression loss) for the generator. generator_loss = regression_loss + lambda_gan * GAN_loss')
    parser.add_argument('-ld', '--lambda_d', type=float, default='1.')
    parser.add_argument('-di', '--d_input', type=str, default='motion', help='motion/pose')
    parser.add_argument('-l', '--reg_loss', type=str, default='pose', help='motion/pose/both')
    parser.add_argument('-rl', '--reg_loss_type', type=str, default='l1', help='l1/l2')
    parser.add_argument('-lmrl', '--lambda_motion_reg_loss', type=float, default=1, help='multiplier for motion reg loss in case both motion and pose are used')
    parser.add_argument('-g', '--gans', type=int, required=True)
    parser.add_argument('-n', '--name', type=str, default='debug', help='experiment name to be saved in the training directory')
    parser.add_argument('-c', '--checkpoint', type=str, help='tensorflow checkpoint path, for finetuning, inference')
    parser.add_argument('-id', '--itr_d', type=int, default=1)
    parser.add_argument('-ig', '--itr_g', type=int, default=1)
    parser.add_argument('-lrg', '--lr_g', type=float, default=1e-4)
    parser.add_argument('-lrd', '--lr_d', type=float, default=1e-4)
    parser.add_argument('-ag', '--arch_g', type=str, default='audio_to_pose_gans')
    parser.add_argument('-ad', '--arch_d', type=str, default='D_patchgan')
    parser.add_argument('-norm', '--norm', type=str, default='batch', help='Norm layer to use. Options are batch, instance and group.')
    parser.add_argument('-tc', '--train_csv', type=str)
    parser.add_argument('-s', '--speaker', type=str)
    parser.add_argument('-output_path', '--output_path', type=str, default='/tmp')
    parser.add_argument('-rp', '--config', type=str, default='audio_to_pose')
    parser.add_argument('-ov', '--output_videos', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=300)
    parser.add_argument('-seq_len', '--seq_len', type=int, default=64)
    parser.add_argument('-sample', '--sample', type=int)

    return parser

configs = {
    "audio_to_pose": {"num_keypoints": 98, "processor": "audio_to_pose", "flatten": False, "input_shape": [None, AUDIO_SHAPE]},
    "audio_to_pose_inference": {"num_keypoints": 98, "processor": "audio_to_pose_inference", "flatten": False, "input_shape": [None, AUDIO_SHAPE]}
}


def get_config(name):
    return configs[name]