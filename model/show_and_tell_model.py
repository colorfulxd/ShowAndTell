# -*- coding:utf-8 -*-

# @Time    : 2019-03-01 15:33

# @Author  : Swing


import tensorflow as tf

from model.image_utils import image_embedding, image_processing
from model.data_utils import inputs as input_ops
from model.configuration import ModelConfig
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib import slim


class ShowAndTellModel(object):

    def __init__(self, config: ModelConfig, mode, train_inception=False):

        assert mode in ['train', 'eval', 'inference']
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        self.reader = tf.TFRecordReader()

        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale
        )

        # Tensor [n, h, w, c] float32
        self.images = None

        # Tensor [batch_size, padded_length] int32
        self.input_seqs = None

        # Tensor [batch_size, padded_length] int32
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        return self.mode == 'train'

    def process_image(self, encoded_image, thread_id=0):
        return image_processing.process_image(encoded_image, is_training=self.is_training(),
                                              height=self.config.image_height, width=self.config.image_width,
                                              thread_id=thread_id, image_format=self.config.image_format)

    def build_inputs(self):
        if self.mode == 'inference':
            # In interference mode, images and inputs are fed via placeholder.
            image_feed = tf.placeholder(tf.string, shape=[], name='image_feed')
            input_feed = tf.placeholder(tf.int64, shape=[None], name='input_feed')

            images = tf.expand_dims(self.process_image(image_feed), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None

        else:
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads
            )

            assert self.config.num_preprocess_threads % 2 == 0

            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encode_image, caption = input_ops.parse_sequence_example(
                    serialized_sequence_example,
                    image_feature=self.config.image_feature_name,
                    caption_feature=self.config.caption_feature_name
                )

                image = self.process_image(encode_image, thread_id=thread_id)
                images_and_captions.append([image, caption])

            queue_capacity = (2 * self.config.num_preprocess_threads * self.config.batch_size)
            images, input_seqs, target_seqs, input_mask = (
                input_ops.batch_with_dynamic_pad(images_and_captions, self.config.batch_size, queue_capacity)
            )

        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_image_embedding(self):
        inception_output = image_embedding.inception_v3(self.images,
                                                        trainable=self.train_inception,
                                                        is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV3'
        )

        # Map inception output into embedding space.
        with tf.variable_scope('image_embedding') as scope:
            image_embeddings = slim.fully_connected(
                inputs=inception_output,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope
            )

        # Save the embedding size in the graph
        tf.constant(self.config.embedding_size, name='embedding_size')

        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):
        with tf.variable_scope('seq_embedding'), tf.device('/CPU:0'):
            embedding_map = tf.get_variable(
                name='map',
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer
            )
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

    def build_model(self):
        """
        Note that this cell is not optimized for performance. Please use
        `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
        `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
        better performance on CPU.
        """
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.num_lstm_units, state_is_tuple=True)

        if self.mode == 'train':
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob
            )

        with tf.variable_scope('lstm', initializer=self.initializer) as lstm_scope:
            zero_state = lstm_cell.zero_state(
                batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
            _, initial_state = lstm_cell(self.image_embeddings, zero_state)

            lstm_scope.reuse_variables()

            if self.mode == 'inference':
                # In inference model, use concatenated states for convenient feeding and fetching
                tf.concat(axis=1, values=initial_state, name='initial_state')

                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")

                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                    state=state_tuple
                )

                tf.concat(axis=1, values=state_tuple, name='state')

            else:

                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs, _ = dynamic_rnn(
                    cell=lstm_cell,
                    inputs=self.seq_embeddings,
                    sequence_length=sequence_length,
                    initial_state=initial_state,
                    dtype=tf.float32,
                    scope=lstm_scope
                )

            lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope('logits') as logits_scope:
            logits = slim.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.config.vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope
            )

        if self.mode == 'inference':
            tf.nn.softmax(logits, name='softmax')
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            # Compute loss
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)

            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                tf.reduce_sum(weights),
                                name='batch_loss')

            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # Add summaries
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)

            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

    def setup_inception_initializer(self):
        """
        Set up the function to restore inception variables from checkpoint.
        :return:
        """

        if self.mode != 'inference':
            saver = tf.train.Saver(self.inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name='global_step',
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        self.global_step = global_step

    def build(self):

        """Create all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_embedding()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()
