import utils
utils.set_up_matplotlib()
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import numpy as np
import pickle


from models import generator
from utils import DataLoader, load, save, psnr_error, diff_mask
from constant import const
import evaluate

def calculate_score(psnrs):
   scores = np.array([], dtype=np.float32) 
   distance = psnrs
   if (distance.max() - distance.min())!=0:
      distance = (distance - distance.min()) / (distance.max() - distance.min())
   else:
      distance = (distance - 0) / (distance.max() - 0)
   scores = np.concatenate((scores[:], distance[DECIDABLE_IDX:]), axis=0)
   return scores

def visualize(gt_frames, pred_frames, labels, scores, num_vid):
   print(scores)
   labels = gt[num_vid]
   for i in range(len(frame)):
      fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(16, 8), nrows=4, ncols=1)
      plt.show()
      ax2.imshow((pred_frames[i][0]+1)/2.0)
      ax1.imshow((gt_frames[i] +1)/ 2.0)
      img1 = gt_frames[i]
      img2 = pred_frames[i][0]
      error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
      error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
      error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))
      lum_img = np.maximum(np.maximum(error_r, error_g), error_b)
      # Uncomment the next line to turn the colors upside-down
      #lum_img = np.negative(lum_img);
      ax3.imshow(lum_img)

      #compute scores
      temp_psnrs = psnrs
      #temp_psnrs[0:num_his] = temp_psnrs[num_his]
      #frame_order[0:num_his] = frame_order[num_his]
      distance = temp_psnrs
      if (distance.max() - distance.min())!=0:
         distance = (distance - distance.min()) / (distance.max() - distance.min())
      else:
         distance = (distance - 0) / (distance.max() - 0)
         #scores = np.concatenate((scores, distance[DECIDABLE_IDX:]), axis=0)
         #scores[i]=distance[DECIDABLE_IDX:]

      ax4.plot(frame_order,scores, label="scores")
      #ax4.plot(frame_order,distance, label="scores")
      ax4.axis([0, length, 0, 1])

      plt.savefig('../images/{}_{}.png'.format(video_name,'%04d'%(i)), dpi=200)
      plt.clf()


slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

num_his = const.NUM_HIS
height, width = 256, 256

snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
evaluate_name = const.EVALUATE

print(const)
# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                             dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    diff_mask_tensor = diff_mask(test_outputs, test_gt)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# ckpt_name = 'model.ckpt-1000'
# ckpt = os.path.join(snapshot_dir, ckpt_name)
video_name = '01'
i = 115
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')
    gt_loader = evaluate.GroundTruthLoader()
    print(gt_loader.NAME_MAT_MAPPING)
    gt = evaluate.get_gt(dataset=dataset_name)
    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, snapshot_dir)
    videos_info = data_loader.videos
    num_videos = len(videos_info.keys())
    total = 0
    psnr_records = []
    DECIDABLE_IDX = 4
    num_vid = -1
    for video_name, video in videos_info.items():
        #video_name = '91.mp4'
        #video = videos_info[video_name]
        num_vid = num_vid + 1
        length = video['length']
        total += length
        
        frame_order = np.array([], dtype=np.float32)
        gt_frames = []
        pred_frames = []
        diff_frames = []
        #psnrs = np.empty(shape=(length,), dtype=np.float32)
        psnrs = np.empty(shape=(length,), dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        scores = np.array([], dtype=np.float32)
        for i in range(num_his, length):
            video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1) # video clip size is (W,H,(4+1)*3)
            psnr, pred_frame, diff = sess.run([test_psnr_error, test_outputs, diff_mask_tensor],
                                              feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
            psnrs[i] = psnr
            #psnrs = np.concatenate((psnrs,[psnr]), axis=0)
            frame_order = np.concatenate((frame_order, [i]),axis=0)
            
            print('video = {} / {}, i = {} / {}'.format(
                    video_name, num_videos, i, length,))
            gt_frame = video_clip[:,:,-3:]

            gt_frames.append(gt_frame)
            pred_frames.append(pred_frame)
            diff_frames.append(diff)
        psnrs[0:num_his] = psnrs[num_his]
        psnr_records.append(psnrs)
        scores = calculate_score(psnrs)
        visualize(frame = gt_frame, pred_frame = pred_frames, labels = gt, 
                  scores = scores,
                  num_vid = num_vid)
        break



