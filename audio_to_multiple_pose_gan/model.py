import datetime
import logging
import subprocess
from logging import getLogger

import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from audio_to_multiple_pose_gan.config import get_config
from audio_to_multiple_pose_gan.dataset import load_train, generate_batch, get_processor
from audio_to_multiple_pose_gan.static_model_factory import get_model
from audio_to_multiple_pose_gan.tf_layers import to_motion_delta, keypoints_to_train, keypoints_regloss
from common.audio_lib import save_audio_sample
from common.audio_repr import raw_repr
from common.consts import RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS, LEFT_HAND_KEYPOINTS, \
    RIGHT_HAND_KEYPOINTS, POSE_SAMPLE_SHAPE, G_SCOPE, D_SCOPE, SR
from common.evaluation import compute_pck
from common.pose_logic_lib import translate_keypoints, get_sample_output_by_config
from common.pose_plot_lib import save_side_by_side_video, save_video_from_audio_video

logging.basicConfig()
logger = getLogger("model.logger")


class PoseGAN():

    def __init__(self, args, seq_len=64):
        self.args = args
        self.sess = tf.Session()
        self.real_pose = tf.placeholder(tf.float32, [None, seq_len, POSE_SAMPLE_SHAPE[-1]])
        self.audio_A = tf.placeholder(tf.float32, [None, None])
        self.is_training = tf.placeholder(tf.bool, ())

        g_func = get_model(args.arch_g)
        cfg = get_config(self.args.config)
        self.fake_pose = g_func({"audio": self.audio_A, "pose": self.real_pose, "config": cfg, "args": self.args}, is_training=self.is_training)

        # remove base keypoint which is always [0,0]. Keeping it may ruin GANs training due discrete problems. etc.
        training_keypoints = self._get_training_keypoints()

        training_real_pose = keypoints_to_train(self.real_pose, training_keypoints)
        training_real_pose = get_sample_output_by_config(training_real_pose, cfg)

        training_real_pose_motion = to_motion_delta(training_real_pose)

        training_fake_pose = keypoints_to_train(self.fake_pose, training_keypoints)
        training_fake_pose_motion = to_motion_delta(training_fake_pose)

        # regressin loss on motion or pose

        self.pose_regloss = 0
        if self.args.reg_loss in ['pose', 'both']:
            pose_reg = keypoints_regloss(training_real_pose, training_fake_pose, self.args.reg_loss_type)
            tf.summary.scalar(name='pose_reg', tensor=pose_reg, collections=['g_summaries'])
            self.pose_regloss += pose_reg

        if self.args.reg_loss in ['motion', 'both']:
            motion_reg = keypoints_regloss(training_real_pose_motion, training_fake_pose_motion,
                                           self.args.reg_loss_type) * self.args.lambda_motion_reg_loss
            tf.summary.scalar(name='motion_reg', tensor=motion_reg, collections=['g_summaries'])
            self.pose_regloss += motion_reg

        # Global Discriminator and Hand Discriminator
        self.G_gan_loss = tf.convert_to_tensor(0.)
        if self.args.gans:
            d_func = get_model(args.arch_d)

            # get full body keypoints
            D_training_keypoints = self._get_training_keypoints()
            D_real_pose = keypoints_to_train(self.real_pose, D_training_keypoints)
            D_fake_pose = keypoints_to_train(self.fake_pose, D_training_keypoints)

            # d motion or pose
            if self.args.d_input == 'motion':
                D_fake_pose_input = to_motion_delta(D_fake_pose)
                D_real_pose_input = to_motion_delta(D_real_pose)
            elif self.args.d_input == 'pose':
                D_fake_pose_input = D_fake_pose
                D_real_pose_input = D_real_pose
            elif self.args.d_input == 'both':
                # concatenate on the temporal axis
                D_fake_pose_input = tf.concat([D_fake_pose, to_motion_delta(D_fake_pose)], axis=1)
                D_real_pose_input = tf.concat([D_real_pose, to_motion_delta(D_real_pose)], axis=1)
            else:
                raise ValueError("d_input wrong value")

            self.fake_pose_score = d_func(D_fake_pose_input, is_training=self.is_training)
            self.real_pose_score = d_func(D_real_pose_input, reuse=True, is_training=self.is_training)

            # loss for training the global D
            self.D_loss = tf.losses.mean_squared_error(tf.ones_like(self.real_pose_score), self.real_pose_score) \
                          + args.lambda_d * tf.losses.mean_squared_error(tf.zeros_like(self.fake_pose_score),
                                                                         self.fake_pose_score)
            tf.summary.scalar(name='D_loss', tensor=self.D_loss, collections=['d_summaries'])

            # loss for training the generator from the global D - have I fooled the global D?
            self.G_gan_loss = tf.losses.mean_squared_error(tf.ones_like(self.fake_pose_score), self.fake_pose_score)
            tf.summary.scalar(name='G_gan_loss', tensor=self.G_gan_loss, collections=['g_summaries'])

            # train global D
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=D_SCOPE)):
                self.train_D = tf.train.AdamOptimizer(learning_rate=self.args.lr_d)\
                    .minimize(loss=self.D_loss,var_list=tf.trainable_variables(scope=D_SCOPE))


        # sum up ALL the losses for training the generator
        self.G_loss = self.pose_regloss + args.lambda_gan * self.G_gan_loss
        tf.summary.scalar(name='train_loss', tensor=self.G_loss, collections=['g_summaries'])

        # train the generator
        if self.args.mode != 'inference':
            trainable_variables = tf.trainable_variables(scope=G_SCOPE)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=G_SCOPE)):
                self.train_G = tf.train.AdamOptimizer(learning_rate=self.args.lr_g).minimize(loss=self.G_loss,
                                                                                             var_list=trainable_variables)

    def _get_training_keypoints(self):
        training_keypoints = []
        training_keypoints.extend(RIGHT_BODY_KEYPOINTS)
        training_keypoints.extend(LEFT_BODY_KEYPOINTS)
        for i in range(5):
            training_keypoints.extend(RIGHT_HAND_KEYPOINTS(i))
            training_keypoints.extend(LEFT_HAND_KEYPOINTS(i))
        training_keypoints = sorted(list(set(training_keypoints)))
        return training_keypoints

    def train(self, epochs=1000):
        batch_size = self.args.batch_size
        tc = datetime.datetime.now()
        base_path = os.path.join(self.args.output_path, self.args.config, self.args.name, '%s' %
                                 (str(tc).replace('.', '-').replace(' ', '--').replace(':', '-')))
        os.makedirs(base_path)
        cfg = get_config(self.args.config)
        process_row, decode_pose = get_processor(cfg)
        df = pd.read_csv(self.args.train_csv)
        if self.args.speaker != None:
            df = df[df['speaker'] == self.args.speaker]
        train_generator, num_samples_train = load_train(process_row, batch_size, df, generate_batch, workers=3,
                                                        max_queue_size=32)

        df_dev = df[df['dataset'] == 'dev'].sample(n=512, random_state=1337)
        self.sess.run(tf.global_variables_initializer())
        if hasattr(self.args, 'checkpoint') and self.args.checkpoint:
            scope_list = ['generator']
            if self.args.gans:
                scope_list += ['discriminator']
            self.restore(self.args.checkpoint, scope_list=scope_list)

        g_summaries = tf.summary.merge(tf.get_collection("g_summaries"))
        if self.args.gans:
            d_summaries = tf.summary.merge(tf.get_collection("d_summaries"))

        writer = tf.summary.FileWriter(base_path, self.sess.graph)
        ITR_PER_EPOCH = 300

        # init
        lowest_validation_loss = 10

        for j in range(epochs):
            keypoints1_list, keypoints2_list, df_loss = self.predict_df(df_dev, cfg, [0, 0], [0, 0])
            avg_loss = np.mean(df_loss)
            pck_loss = np.mean(compute_pck(pred=keypoints1_list.reshape((-1, 2, 49))[:, :, 1:],
                                           gt=keypoints2_list.reshape((-1, 2, 49))[:, :, 1:]))
            print "Epoch %s (step %s): validation loss: %s, pck: %s" % (j, ITR_PER_EPOCH * j, avg_loss, pck_loss)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="validation_loss", simple_value=avg_loss),
            ])
            writer.add_summary(summary, global_step=ITR_PER_EPOCH * j)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="pck_loss", simple_value=pck_loss),
            ])
            writer.add_summary(summary, global_step=ITR_PER_EPOCH * j)

            keypoints1_list = translate_keypoints(keypoints1_list, [900, 290])
            keypoints2_list = translate_keypoints(keypoints2_list, [1900, 280])
            if 'train_ratio' in cfg:
                train_ratio = cfg['train_ratio']
            else:
                train_ratio = None

            if lowest_validation_loss > avg_loss:
                lowest_validation_loss = avg_loss
                tf.train.Saver().save(self.sess, os.path.join(base_path,
                                                              'best_ckpt-step_%s_validation_loss_%.3f.ckp' % (
                                                              ITR_PER_EPOCH * j, avg_loss)))

            if j % 15 == 14:
                tf.train.Saver().save(self.sess, os.path.join(base_path, 'ckpt-step-%s.ckp' % (ITR_PER_EPOCH * j)))
                if self.args.output_videos:
                    self.save_prediction_video_by_percentiles(df_dev, keypoints1_list, keypoints2_list,
                                                              os.path.join(base_path, str(j)), train_ratio=train_ratio,
                                                              limit=32, loss=df_loss)
            minibatch_g_loss = []
            minibatch_loss = []
            for i in range(ITR_PER_EPOCH):
                if self.args.gans:
                    for itr_d in range(self.args.itr_d):
                        audio_X, pose_Y = train_generator.next()
                        d_loss, d_summaries_str, _ = self.sess.run([self.D_loss, d_summaries, self.train_D],
                                                                   feed_dict={self.audio_A: audio_X,
                                                                              self.real_pose: pose_Y,
                                                                              self.is_training: 1})

                    writer.add_summary(d_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)

                for itr_g in range(self.args.itr_g):
                    audio_X, pose_Y = train_generator.next()
                    g_loss, g_regloss, g_gan_loss, g_summaries_str, _ = self.sess.run(
                        [self.G_loss, self.pose_regloss, self.G_gan_loss, g_summaries, self.train_G],
                        feed_dict={self.audio_A: audio_X, self.real_pose: pose_Y, self.is_training: 1})

                minibatch_g_loss.append(g_loss)
                minibatch_loss.append(g_regloss)

            if i % 100 == 0:
                writer.add_summary(g_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)
                print "Epoch %s: Iteration: %s. g_loss: %.4f, g_gan_loss: %.4f, g_reg_loss: %.4f" % (
                j, i, np.mean(minibatch_g_loss), g_gan_loss, np.mean(minibatch_loss))
                if self.args.gans:
                    print "Epoch %s: Iteration: %s. d_loss: %.4f" % (j, i, d_loss)

    def restore(self, ckp, scope_list=('generator')):
        variables = []
        for s in scope_list:
            variables += tf.global_variables(scope=s)
        tf.train.Saver(variables).restore(self.sess, ckp)

    def save_prediction_video(self, df, keypoints_pred, keypoints_gt, save_path, limit=None, loss=None):
        if limit == None:
            limit = len(df)
        for i in range(min(len(df), limit)):
            try:
                row = df.iloc[i]
                keypoints1 = keypoints_pred[i]
                keypoints2 = keypoints_gt[i]

                dir_name = os.path.join(save_path, str(row['interval_id']))

                if not (os.path.exists(dir_name)):
                    os.makedirs(dir_name)

                video_fn = os.path.basename(row['video_fn']).split('.')[0]
                interval_id = row['interval_id']
                temp_otpt_fn = os.path.join(dir_name, '%s.mp4' % interval_id)
                otpt_fn = os.path.join(save_path,
                                       '%s_%s_%s_%s_{loss}.mp4' % (video_fn, interval_id, row['start'], row['end']))
                save_side_by_side_video(dir_name, keypoints1, keypoints2, temp_otpt_fn, delete_tmp=False)
                audio = np.load(row['pose_fn'])['audio']
                audio_out_path = '/tmp/audio_cache/%s_%s_%s.wav' % (row["interval_id"], row['start'], row['end'])
                save_audio_sample(audio, audio_out_path, 16000, 44100)
                if loss is not None:
                    otpt_fn = otpt_fn.format(loss=loss[i])
                save_video_from_audio_video(audio_out_path, temp_otpt_fn, otpt_fn)
                subprocess.call('rm -R "%s"' % (dir_name), shell=True)
            except Exception as e:
                logger.exception(e)

    def save_prediction_video_by_percentiles(self, df, keypoints_pred, keypoints_gt, save_path, loss_percentile_bgt=95,
                                             loss_percentile_smt=5, train_ratio=None, limit=None, loss=None):
        if limit is None:
            limit = len(df)

        if loss_percentile_bgt != None:
            thres = np.percentile(loss, loss_percentile_bgt)
            indices = np.where(loss > thres)[0]
            self.save_prediction_video(df.iloc[indices], keypoints_pred[indices], keypoints_gt[indices],
                                       os.path.join(save_path, str(loss_percentile_bgt)),
                                       loss=loss[indices], limit=limit / 2)

        if loss_percentile_smt != None:
            thres = np.percentile(loss, loss_percentile_smt)
            indices = np.where(loss < thres)[0]
            self.save_prediction_video(df.iloc[indices], keypoints_pred[indices], keypoints_gt[indices],
                                       os.path.join(save_path, str(loss_percentile_smt)),
                                       loss=loss[indices], limit=limit / 2)

        self.save_prediction_video(df, keypoints_pred, keypoints_gt, os.path.join(save_path, 'random'), loss=loss,
                                   limit=limit / 2)

    def predict_row(self, row, cfg, shift_pred=(0, 0), shift_gt=(0, 0)):
        process_row, decode_pose = get_processor(cfg)
        x, y = process_row(row)
        res, row_loss = self.sess.run([self.fake_pose, self.G_loss], feed_dict={self.audio_A: np.expand_dims(x, 0), self.is_training: 0, self.real_pose: np.expand_dims(y, 0)})
        return self._post_process(res, y, shift_gt, shift_pred, decode_pose, row), row_loss

    def predict_df(self, df, cfg, shift_pred=(900, 290), shift_gt=(1900, 280)):
        keypoints1_list = []
        keypoints2_list = []
        df_loss = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                (keypoints1, keypoints2), row_loss = self.predict_row(row, cfg, shift_pred=shift_pred,shift_gt=shift_gt)
            except Exception as e:
                logger.exception(e)
                continue

            keypoints1_list.append(keypoints1)
            keypoints2_list.append(keypoints2)
            df_loss.append(row_loss)

        return np.array(keypoints1_list), np.array(keypoints2_list), np.array(df_loss)

    def _post_process(self, res, y, shift_gt, shift_pred, decode_pose, row):
        return self._post_process_output(res[0], decode_pose, shift_pred, row['speaker']), \
               self._post_process_output(y, decode_pose, shift_gt, row['speaker'])

    def _post_process_output(self, res, decode_pose, shift, speaker):
        return decode_pose(res, shift, speaker)

    def predict_audio_by_fn(self, fn, cfg, speaker, shift_pred=(0, 0)):
        _, decode_pose = get_processor(cfg)
        wav, _ = raw_repr(fn, SR)
        res = self.sess.run(self.fake_pose, feed_dict={self.audio_A: np.expand_dims(wav, 0), self.is_training: 0})
        return self._post_process_output(res[0], decode_pose, shift_pred, speaker)

    def predict_audio(self, wav, cfg, speaker, shift_pred=(0, 0)):
        _, decode_pose = get_processor(cfg)
        res = self.sess.run(self.fake_pose, feed_dict={self.audio_A: np.expand_dims(wav, 0), self.is_training: 0})
        return self._post_process_output(res[0], decode_pose, shift_pred, speaker)
