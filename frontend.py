import numpy as np
import os,logging,datetime

from keras import backend as keras_backend
import math
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import tensorflow as tf
from utils import decode_netout, compute_overlap, compute_ap
from preprocessing import BatchGenerator
from backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature, Inception3Feature, VGG16Feature, ResNet50Feature
from backend import FullYoloFeatureNCHW

from callbacks import TB2

logger = logging.getLogger(__name__)


def create_config_proto(params):
   '''EJ: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = params.num_intra
   config.inter_op_parallelism_threads = params.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = params.kmp_affinity
   return config


class YOLO(object):
    def __init__(self, config_file,
                       args):
        logger.debug('YOLO init')
        self.config_file = config_file
        self.args = args
        self.input_shape = tuple(config_file['model']['input_shape'])
        
        self.labels   = config_file['model']['labels']
        self.nb_class = len(config_file['model']['labels'])
        self.nb_box   = len(config_file['model']['anchors']) // 2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors  = config_file['model']['anchors']

        self.max_box_per_image = config_file['model']['max_box_per_image']
        backend = config_file['model']['backend']


        if self.args.horovod:
           logger.debug("importing hvd")
           import horovod.keras as hvd
           self.hvd = hvd
           logger.info('horovod from: %s',hvd.__file__)
           logger.debug('hvd init')
           self.hvd.init()
           logger.info("Rank: %s",self.hvd.rank())

        if self.args.ml_comm:
           logger.debug("importing ml_comm")
           import ml_comm as mc
           from plugin_keras import InitPluginCallback, BroadcastVariablesCallback, DistributedOptimizer
           self.mc = mc
           self.InitPluginCallback = InitPluginCallback
           self.BroadcastVariablesCallback = BroadcastVariablesCallback
           self.DistributedOptimizer = DistributedOptimizer
           logger.info('ml_comm from: %s',mc.__file__)
           logger.debug('mc init')
           self.mc.init_mpi()
           logger.info("Rank: %s",self.mc.get_rank())

        self.sparse = args.sparse

        config_proto = create_config_proto(self.args)
        keras_backend.set_session(tf.Session(config=config_proto))
        logger.debug("done config proto")

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image     = Input(shape=tuple(self.input_shape))
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))

        if backend == 'Inception3':
            self.feature_extractor = Inception3Feature(self.input_shape)
        elif backend == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_shape)
        elif backend == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_shape)
        elif backend == 'Full Yolo NCHW':
            self.feature_extractor = FullYoloFeatureNCHW(self.input_shape)
        elif backend == 'Full Yolo':
            self.feature_extractor = FullYoloFeature(self.input_shape)
        elif backend == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_shape)
        elif backend == 'VGG16':
            self.feature_extractor = VGG16Feature(self.input_shape)
        elif backend == 'ResNet50':
            self.feature_extractor = ResNet50Feature(self.input_shape)
        else:
            raise Exception('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, VGG16, ResNet50, and Inception3 at the moment!')

        logger.debug("done building feature extractor")

        self.feature_extractor.feature_extractor.summary()
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        logger.info('grid h x w = %s x %s',self.grid_h,self.grid_w)
        features = self.feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1,1), strides=(1,1),
                        padding='same',
                        name='DetectionLayer',
                        kernel_initializer='lecun_normal',
                        data_format='channels_first')(features)
        # output = Reshape((self.grid_h, self.grid_w, 4 + 1 + self.nb_class))(output)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])
        
        self.model = Model([input_image, self.true_boxes], output)

        
        # initialize the weights of the detection layer
        # Taylor: Keras should initialize ALL model weights randomly
        # Tom: so where are the weights being initialized then?
        # Taylor: appears all weights are initialized at creation of a layer
        '''
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)

        layer.set_weights([new_kernel, new_bias])
        '''

        # print a summary of the whole model
        self.model.summary()
        logger.debug("done YOLO init")

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        # tf.Print(cell_x.shape,cell_x)
        cell_y = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_h), [self.grid_w]), (1, self.grid_w, self.grid_h, 1, 1)))
        cell_y = tf.transpose(cell_y, (0,2,1,3,4))

        # turam - skip concatenation with transpose, as it assumes that we have a square image
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])
        # cell_grid = tf.tile(cell_x, [self.batch_size, 1, 1, self.nb_box, 1])
        
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically
        
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half
        
        intersect_mins  = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half
        
        intersect_mins  = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        best_ious = tf.reduce_max(iou_scores, axis=4)

        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches + 1),
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                       true_box_wh + tf.ones_like(true_box_wh) *
                                       np.reshape(self.anchors, [1,1,1,self.nb_box,2]) *
                                       no_boxes_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy,
                                       true_box_wh,
                                       coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)
        
        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
            
            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            zero = tf.constant(0, dtype=tf.float32)
            where = tf.not_equal(y_true[...,4], zero)
            indices = tf.where(where)



            loss = tf.Print(loss, [y_true], message='y_true \t', summarize=1000)
            loss = tf.Print(loss, [tf.shape(y_true)], message='y_true shape \t', summarize=1000)
            loss = tf.Print(loss, [y_true[indices[0][0],indices[0][1],indices[0][2],indices[0][3]]], message='y_true value \t', summarize=1000)
            loss = tf.Print(loss, [indices[0][0],indices[0][1],indices[0][2],indices[0][3]], message='y_true value indices\t', summarize=1000)
            loss = tf.Print(loss, [y_pred[indices[0][0],indices[0][1],indices[0][2],indices[0][3]]], message='y_pred value \t', summarize=1000)

            loss = tf.Print(loss, [true_box_wh[indices[0][0],indices[0][1],indices[0][2],indices[0][3]]], message='true_box_wh \t', summarize=1000)
            loss = tf.Print(loss, [pred_box_wh[indices[0][0],indices[0][1],indices[0][2],indices[0][3]]], message='pred_box_wh \t', summarize=1000)

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
        
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to
             ):
        logger.debug("YOLO train")
        evts_per_file      = self.config_file['train']['evts_per_file']
        train_times        = self.config_file['train']['train_times']
        valid_times        = self.config_file['valid']['valid_times']
        nb_epochs          = self.config_file['train']['nb_epochs']
        learning_rate      = self.config_file['train']['learning_rate']
        self.batch_size    = self.config_file['train']['batch_size']
        warmup_epochs      = self.config_file['train']['warmup_epochs']
        self.object_scale       = self.config_file['train']['object_scale']
        self.no_object_scale    = self.config_file['train']['no_object_scale']
        self.coord_scale        = self.config_file['train']['coord_scale']
        self.class_scale        = self.config_file['train']['class_scale']
        self.saved_weights_name = self.config_file['train']['saved_weights_name']
        self.debug              = self.config_file['train']['debug']



        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_C': self.input_shape[0],
            'IMAGE_H': self.input_shape[1],
            'IMAGE_W': self.input_shape[2],
            'GRID_H': self.grid_h,
            'GRID_W': self.grid_w,
            'BOX': self.nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self.anchors,
            'BATCH_SIZE': self.batch_size,
            'TRUE_BOX_BUFFER': self.max_box_per_image,
        }
        logger.debug("create batch generators")

        train_generator = BatchGenerator(train_imgs,
                                     generator_config,
                                     evts_per_file,
                                     norm=self.config_file['train']['normalize'],
                                     sparse=self.sparse)
        valid_generator = BatchGenerator(valid_imgs,
                                     generator_config,
                                     evts_per_file,
                                     norm=self.config_file['train']['normalize'],
                                     jitter=False,
                                     sparse=self.sparse)

        logger.debug("done batch generators")
                                     
        self.warmup_batches  = warmup_epochs * (train_times * len(train_generator) + valid_times * len(valid_generator))

        ############################################
        # Compile the model
        ############################################

        logger.debug("create Adam")
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if self.args.horovod:
            logger.debug("hvd optimizer")
            optimizer = self.hvd.DistributedOptimizer(optimizer)

        if self.args.ml_comm:
            logger.debug("Distributed optimizer")
            optimizer = self.DistributedOptimizer(optimizer)

        # self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        logger.debug("compile model")
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)
        logger.debug("done compile model")

        # To use Cray Plugin we need to calculate the number of trainable variables 
        # Also useful to adjust number of epochs run
        if self.args.ml_comm:
            trainable_count = int(
              np.sum([keras_backend.count_params(p) for p in set(self.model.trainable_weights)]))
            non_trainable_count = int(
              np.sum([keras_backend.count_params(p) for p in set(self.model.non_trainable_weights)]))

            # Adjust number of Epochs based on number of ranks used
            nb_epochs = int(nb_epochs / self.mc.get_nranks())
            if nb_epochs == 0:
              nb_epochs = 1
            total_steps = int(math.ceil((warmup_epochs + nb_epochs) * (
                           (train_times*len(train_imgs) +
                           valid_times*len(valid_imgs)) / self.batch_size)))

            # if hvd.rank() == 0:
            # if mc.get_rank() == 0:
            #  print('Total params: {:,}'.format(trainable_count + non_trainable_count))
            #  print('Trainable params: {:,}'.format(trainable_count))
            #  print('Non-trainable params: {:,}'.format(non_trainable_count))
            #  print('Calculation of total_steps: {:,}'.format(total_steps))
        ############################################
        # Make a few callbacks
        ############################################

        dateString = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
        log_path = os.path.join(self.config_file['tensorboard']['log_dir'],dateString)
        

        verbose = self.config_file['train']['verbose']
        if self.args.horovod:
            logger.debug("hvd callbacks")
            callbacks = []
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            callbacks.append(self.hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            
            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            callbacks.append(self.hvd.callbacks.MetricAverageCallback())

            # create tensorboard callback
            tensorboard = TB2(evaluate=self.evaluate,generator=valid_generator,log_every=10,log_dir=log_path,update_freq='batch')
            callbacks.append(tensorboard)
            if self.hvd.rank() == 0:
               verbose = self.config_file['train']['verbose']
               os.makedirs(log_path)

               checkpoint = ModelCheckpoint(self.config_file['model']['model_checkpoint_file'].format(date=dateString),
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              period=1)
               callbacks.append(checkpoint)
            else:
               verbose = 0
        elif self.args.ml_comm:
            logger.debug("cray-plugin callbacks")
            callbacks = []
            # Cray ML Plugin: broadcast callback
            init_plugin = self.InitPluginCallback(total_steps, trainable_count)

            callbacks.append(init_plugin)
            broadcast   = self.BroadcastVariablesCallback(0)
            callbacks.append(broadcast)

            # create tensorboard callback
            tensorboard = TB2(evaluate=self.evaluate,generator=valid_generator,log_every=10,log_dir=log_path,update_freq='batch')
            callbacks.append(tensorboard)

            if self.mc.get_rank() == 0:
               verbose = self.config_file['train']['verbose']
               os.makedirs(log_path)

               checkpoint = ModelCheckpoint(self.config_file['model']['model_checkpoint_file'].format(date=dateString),
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min',
                              period=1)
               callbacks.append(checkpoint)
            else:
               verbose = 0
        else:
           os.makedirs(log_path)
           early_stop = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=3,
                              mode='min',
                              verbose=1)
           checkpoint = ModelCheckpoint(self.config_file['model']['model_checkpoint_file'].format(date=dateString),
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',
                                        period=1)
           tensorboard = TensorBoard(log_dir='./logs',
                                     histogram_freq=0,
                                     # write_batch_performance=True,
                                     write_graph=True,
                                     write_images=False)

           callbacks = [
               early_stop,
               checkpoint,
               tensorboard,
           ]
           
        logger.debug("fit generator")

        

        ############################################
        # Start the training process
        ############################################

        self.model.fit_generator(generator        = train_generator,
                                 steps_per_epoch  = len(train_generator),
                                 epochs           = nb_epochs,
                                 verbose          = verbose,
                                 validation_data  = valid_generator,
                                 validation_steps = len(valid_generator),
                                 callbacks        = callbacks,
                                 workers          = 1,
                                 max_queue_size   = 5)

        ############################################
        # Compute mAP on the validation set
        ############################################
        average_precisions = self.evaluate(valid_generator)

        # print evaluation
        for label, average_precision in average_precisions.items():
            logger.info(self.labels[label], '{:.4f}'.format(average_precision))
        logger.info('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    def evaluate(self,
                 generator,
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        all_detections     = [[None for i in range(generator.num_classes)] for j in range(generator.size())]
        all_annotations    = [[None for i in range(generator.num_classes)] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_channels, raw_height, raw_width = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin * raw_width,
                                        box.ymin * raw_height,
                                        box.xmax * raw_width,
                                        box.ymax * raw_height,
                                        box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(generator.num_classes):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)
            average_precisions[label] = average_precision

        return average_precisions

    def predict(self, image):
        # Taylor: don't use this anywhere
        # _, image_h, image_w = image.shape
        # Taylor: we don't check image size
        # image = cv2.resize(image, (self.input_shape[1], self.input_shape[2]))
        image = self.feature_extractor.normalize(image)

        # Taylor: swap channel indices
        # input_image = image[:,:,::-1]
        input_image = image[::-1,:,:]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([image, dummy_array])[0]
        boxes  = decode_netout(netout, self.anchors, self.nb_class)

        return boxes
