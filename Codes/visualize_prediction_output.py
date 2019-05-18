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

def visualize(gt_frame, pred_frame, labels, scores, frame_order, num_vid):
   #print(scores)
   labels = gt[num_vid]
   length = len(scores)
   threshold = 0.5
   fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(16, 8), nrows=4, ncols=1)
   plt.show()
   ax2.imshow((pred_frame[0]+1)/2.0)
   ax1.imshow((gt_frame +1)/ 2.0)
   img1 = gt_frame
   img2 = pred_frame[0]
   error_r = np.fabs(np.subtract(img2[:,:,0], img1[:,:,0]))
   error_g = np.fabs(np.subtract(img2[:,:,1], img1[:,:,1]))
   error_b = np.fabs(np.subtract(img2[:,:,2], img1[:,:,2]))
   lum_img = np.maximum(np.maximum(error_r, error_g), error_b)
   # Uncomment the next line to turn the colors upside-down
   #lum_img = np.negative(lum_img);
   ax3.imshow(lum_img)

   #compute scores

   ax4.plot(frame_order, scores[0: len(frame_order)], label="scores")
   ax4.text(length - 350, 0.75, r'Ground_truth: {label}'.format(label='Normal' if labels[i] == 0 else 'Abnormal'),
            {'color': 'r', 'fontsize': 15})
   ax4.text(length - 350, 0.55, r'Predicted: {label}'.format(label='Normal' if scores[i] >= threshold else 'Abnormal'),
            {'color': 'b', 'fontsize': 15})
   ax4.axis([0, length, 0, 1])

   plt.savefig('../images/{}_{}/{}.png'.format(dataset_name,video_name,'%04d'%(i)), dpi=200)
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
scores = np.array([], dtype=np.float32)
gt_loader = evaluate.GroundTruthLoader()
gt_loader.AI_CITY_VIDEO_START = 20 * 30
gt = evaluate.get_gt(dataset=dataset_name)
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

with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

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
        video_name = '95.mp4'
        video = videos_info[video_name]
        num_vid = num_vid + 1
        length = video['length']
        total += length
        
        frame_order = np.array([], dtype=np.float32)
        gt_frames = []
        pred_frames = []
        diff_frames = []
        psnrs = np.empty(shape=(length,), dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        
        for i in range(num_his, length):
            video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1) # video clip size is (W,H,(4+1)*3)
            psnr, pred_frame, diff = sess.run([test_psnr_error, test_outputs, diff_mask_tensor],
                                              feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
            psnrs[i] = psnr
            frame_order = np.concatenate((frame_order, [i]),axis=0)
            
            print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
                    video_name, num_videos, i, length, psnr))
            gt_frame = video_clip[:,:,-3:]
        psnrs[0:num_his] = psnrs[num_his]
        psnr_records.append(psnrs)
        scores = calculate_score(psnrs)
        result_dir = '../images/{}_{}/'.format(dataset_name, video_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        break


with tf.Session(config=config) as sess:
   # dataset
   data_loader = DataLoader(test_folder, height, width)

   # initialize weights
   sess.run(tf.global_variables_initializer())
   print('Init global successfully!')
   
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
   video_name = '95.mp4'
   video = videos_info[video_name]
   num_vid = gt_loader.AI_CITY_VIDEO_ORDER[video_name]
   length = video['length']
   total += length
   video_order = gt_loader.AI_CITY_VIDEO_ORDER[video_name]
   frame_order = np.array([], dtype=np.float32)
   psnrs = np.empty(shape=(length,), dtype=np.float32)
   labels = np.array([], dtype=np.int8)
   scores = np.array([], dtype=np.float32)
   for i in range(num_his, length):
      video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1) # video clip size is (W,H,(4+1)*3)
      psnr, pred_frame, diff = sess.run([test_psnr_error, test_outputs, diff_mask_tensor],
                                        feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
      psnrs[i] = psnr
      frame_order = np.concatenate((frame_order, [i]),axis=0)
            
      print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
              video_name, num_videos, i, length, psnr))
      gt_frame = video_clip[:,:,-3:]
      visualize(gt_frame= gt_frame, pred_frame = pred_frame, labels = gt, 
               scores = scores,
               frame_order = frame_order,
               num_vid = num_vid)
