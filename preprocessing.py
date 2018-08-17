import os,glob
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, bbox_iou

# Right now the truth are format as:
# bool objectFound   # if truth particle is in the hard scattering process 
# bbox x                    # globe eta
# bbox y                    # globe phi
# bbox width             # Gaussian sigma (required to be 3*sigma<pi)
# bbox height            # Gaussian sigma (required to be 3*sigma<pi)
# bool class1             # truth u/d 
# bool class2             # truth s
# bool class3             # truth c
# bool class4             # truth d
# bool class_other     # truth g


class BatchGenerator(Sequence):
  def __init__(self,img_files,
         config,
         batch_size,
         shuffle=True,
         jitter=True,
         norm=None):
    self.generator = None
    
    self.config          = config
    self.filelist        = img_files
    self.evts_per_file   = config['EVTS_PER_FILE']
    self.batch_size      = batch_size
    self.nevts           = len(img_files) * config['EVTS_PER_FILE']
    self.nbatches        = int(self.nevts * (1. / self.batch_size))
    self.num_classes     = len(config['LABELS'])
    self.num_grid_x      = config['GRID_W']
    self.num_grid_y      = config['GRID_H']
    
    self.shuffle = shuffle
    self.jitter  = jitter
    self.norm    = norm
    
    self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]
    '''
    ### augmentors by https://github.com/aleju/imgaug
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    self.aug_pipe = iaa.Sequential(
      [
          # apply the following augmenters to most images
        #iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #iaa.Flipud(0.2), # vertically flip 20% of all images
        #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        sometimes(iaa.Affine(
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            #rotate=(-5, 5), # rotate by -45 to +45 degrees
            #shear=(-5, 5), # shear by -16 to +16 degrees
            #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges
                #sometimes(iaa.OneOf([
                #    iaa.EdgeDetect(alpha=(0, 0.7)),
                #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                #])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                #iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                #iaa.Grayscale(alpha=(0.0, 1.0)),
                #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
            ],
            random_order=True
          )
      ],
      random_order=True
    )
    
    if shuffle: np.random.shuffle(self.filelist)
    '''
  
  def __len__(self):
    return int(np.ceil(float(len(self.filelist))/self.config['BATCH_SIZE']))   
  
  def num_classes(self):
    return len(self.config['LABELS'])
  
  def size(self):
    return len(self.filelist)    
  
  def load_annotation(self, i):
    annots = []
    
    ifile=int(i/self.config['EVTS_PER_FILE']/self.config['BATCH_SIZE'])
    ievent=int((i-ifile*self.config['EVTS_PER_FILE']*self.config['BATCH_SIZE'])/self.config['BATCH_SIZE'])
    truths=np.load(self.filelist[ifile])
    objs = truths['truth'][ievent]
    for obj in objs:
      annot = [min(obj[1]-obj[2],0), min(obj[2]-obj[3],0), max(obj[1]+obj[2],1), max(obj[2]+obj[3],1)]
      annot += obj[5:].tolist()
      annots += [annot]
  
    if len(annots) == 0: annots = [[]]
  
    return np.array(annots)
  
  def load_image(self, i):
    ifile=int(i/self.config['EVTS_PER_FILE']/self.config['BATCH_SIZE'])
    ievent=int((i-ifile*self.config['EVTS_PER_FILE']*self.config['BATCH_SIZE'])/self.config['BATCH_SIZE'])
    ibatch=i-ifile*self.config['EVTS_PER_FILE']*self.config['BATCH_SIZE']-ievent*self.config['BATCH_SIZE']
    imagins=np.load(self.filelist[ifile])
    imgs=np.expand_dims(imagins['raw'][ievent][ibatch], axis=4)
    return imgs
  
  def __getitem__(self, idx):
    print '__getitem__',idx
    
    l_bound = idx*self.config['BATCH_SIZE']
    r_bound = (idx+1)*self.config['BATCH_SIZE']
    
    if r_bound > len(self.filelist):
        r_bound = len(self.filelist)
        l_bound = r_bound - self.config['BATCH_SIZE']
        
    im_batch = np.zeros((self.config['BATCH_SIZE']*self.config['EVTS_PER_FILE'], self.config['IMAGE_C'], self.config['IMAGE_H'], self.config['IMAGE_W']))                         # input images
    box_batch = np.zeros((self.config['BATCH_SIZE']*self.config['EVTS_PER_FILE'], 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
    y_batch = np.zeros((self.config['BATCH_SIZE']*self.config['EVTS_PER_FILE'], self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

    
    instance_count=0
    for train_instance in self.filelist[l_bound:r_bound]:
      # augment input image and fix object's position and size
      #img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
      imagin_files=np.load(train_instance)
      imagins=imagin_files['raw']
      all_objs=imagin_files['truth']
      print 'imagins, all_objs',imagins.shape, all_objs.shape
      # construct output from object's x, y, w, h

      for ievt in xrange(self.config['EVTS_PER_FILE']):
        true_box_index = 0
        objs=all_objs[ievt]
        for obj in objs:
          center_x = obj[1]
          center_x = center_x  / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
          center_y = obj[2]
          center_y = center_y  / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

          grid_x = int(np.floor(center_x))
          grid_y = int(np.floor(center_y))

          if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
            obj_indx  = np.nonzero(obj[5:]==1)
            
            center_w = 2*obj[2] / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
            center_h = 2*obj[3] / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
            
            box = [center_x, center_y, center_w, center_h]

            # find the anchor that best predicts this box
            best_anchor = -1
            max_iou     = -1
            
            shifted_box = BoundBox(0, 
                                   0,
                                   center_w,                                                
                                   center_h)
            
            for i in range(len(self.anchors)):
                anchor = self.anchors[i]
                iou    = bbox_iou(shifted_box, anchor)
                
                if max_iou < iou:
                    best_anchor = i
                    max_iou     = iou
                    
            for instance_count in xrange(self.config['EVTS_PER_FILE']):
              # assign ground truth x, y, w, h, confidence and class probs to y_batch
              y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
              y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = obj[0]
              y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx[0][0]] = 1
              
              # assign the true box to box_batch
              box_batch[instance_count, 0, 0, 0, true_box_index] = box
              
              true_box_index += 1
              true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

        if self.norm != None: 
            im_batch[instance_count] = self.norm(imagins[ievt])
        # else:
            # # plot image and bounding boxes for sanity check
            # for obj in all_objs:
              # cv2.rectangle(img[:,:,::-1], (obj[1],obj[2]), (obj[1]+2*obj[3],obj[2]+2*obj[4]), (255,0,0), 3)
              # cv2.putText(img[:,:,::-1], obj['name'], 
                          # (obj['xmin']+2, obj['ymin']+12), 
                          # 0, 1.2e-3 * img.shape[0], 
                          # (0,255,0), 2)
                    
        else: im_batch[instance_count] = self.norm(imagins[ievt])
        instance_count+=1
    
    #print(' new batch created', idx)
    print 'im_batch, box_batch, y_batch: ', im_batch.shape, box_batch.shape, y_batch.shape
    return [im_batch, box_batch], y_batch
  
  def on_epoch_end(self):
    if self.shuffle: np.random.shuffle(self.filelist)
  
  def aug_image(self, train_instance, jitter):
    image_name = train_instance['filename']
    image = cv2.imread(image_name)
  
    if image is None: print('Cannot find ', image_name)
  
    h, w, c = image.shape
    all_objs = copy.deepcopy(train_instance['object'])
  
    if jitter:
      ### scale the image
      scale = np.random.uniform() / 10. + 1.
      image = cv2.resize(image, (0,0), fx = scale, fy = scale)
  
      ### translate the image
      max_offx = (scale-1.) * w
      max_offy = (scale-1.) * h
      offx = int(np.random.uniform() * max_offx)
      offy = int(np.random.uniform() * max_offy)
      
      image = image[offy : (offy + h), offx : (offx + w)]
  
      ### flip the image
      flip = np.random.binomial(1, .5)
      if flip > 0.5: image = cv2.flip(image, 1)
          
      image = self.aug_pipe.augment_image(image)            
      
    # resize the image to standard size
    image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
    image = image[:,:,::-1]
  
    # fix object's position and size
    for obj in all_objs:
      for attr in ['xmin', 'xmax']:
          if jitter: obj[attr] = int(obj[attr] * scale - offx)
              
          obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
          obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
          
      for attr in ['ymin', 'ymax']:
          if jitter: obj[attr] = int(obj[attr] * scale - offy)
              
          obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
          obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)
  
      if jitter and flip > 0.5:
          xmin = obj['xmin']
          obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
          obj['xmax'] = self.config['IMAGE_W'] - xmin
          
    return image, all_objs


def global_to_grid(x,y,w,h,num_grid_x,num_grid_y):
   ''' convert global bounding box coords to
   grid coords. x,y = box center in relative coords
   going from 0 to 1. w,h are the box width and
   height in relative coords going from 0 to 1.
   num_grid_x,num_grid_y define the number of bins
   in x and y for the grid.
   '''
   global_coords  = [x,y]
   global_sizes   = [w,h]
   num_grid_bins  = [num_grid_x,num_grid_y]
   grid_coords    = [0.,0.]
   grid_sizes     = [0.,0.]
   for i in range(len(global_coords)):
      grid_bin_size = 1. / num_grid_bins[i]
      grid_bin = int(global_coords[i] / grid_bin_size)
      grid_coords[i] = (global_coords[i] - grid_bin_size * grid_bin) / grid_bin_size
      grid_sizes[i] = (global_sizes[i] / grid_bin_size)

   return grid_coords,grid_sizes


def grid_to_global(grid_bin_x, grid_bin_y,grid_x,grid_y,grid_w,grid_h,num_grid_x,num_grid_y):
   ''' inverse of global_to_grd '''
   grid_bins      = [grid_bin_x,grid_bin_y]
   grid_coords    = [grid_x,grid_y]
   grid_sizes     = [grid_w,grid_h]
   num_grid_bins  = [num_grid_x,num_grid_y]
   global_coords  = [0.,0.]
   global_sizes   = [0.,0.]
   for i in range(len(global_coords)):
      grid_bin_size     = 1. / num_grid_bins[i]
      global_coords[i]  = (grid_bins[i] + grid_coords[i]) * grid_bin_size
      global_sizes[i]   = grid_sizes[i] * grid_bin_size

   return global_coords,global_sizes
