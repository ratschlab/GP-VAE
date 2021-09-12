"""

Script to train the proposed GP-VAE model.

"""
import os

from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import sys
import os
import time
from datetime import datetime
import matplotlib

# from lib.motion_utils import get_lr
from lib.motion_utils.lr_schedule import get_lr

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
sys.path.append("..")
from lib.models import *



FLAGS = flags.FLAGS

# HMNIST config
# flags.DEFINE_integer('latent_dim', 256, 'Dimensionality of the latent space')
# flags.DEFINE_list('encoder_sizes', [256, 256], 'Layer sizes of the encoder')
# flags.DEFINE_list('decoder_sizes', [256, 256, 256], 'Layer sizes of the decoder')
# flags.DEFINE_integer('window_size', 3, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('sigma', 1.0, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('length_scale', 2.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('beta', 0.8, 'Factor to weigh the KL term (similar to beta-VAE)')
# flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')

# SPRITES config GP-VAE
# flags.DEFINE_integer('latent_dim', 256, 'Dimensionality of the latent space')
# flags.DEFINE_list('encoder_sizes', [32, 256, 256], 'Layer sizes of the encoder')
# flags.DEFINE_list('decoder_sizes', [256, 256, 256], 'Layer sizes of the decoder')
# flags.DEFINE_integer('window_size', 3, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('sigma', 1.0, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('length_scale', 2.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
# flags.DEFINE_float('beta', 0.1, 'Factor to weigh the KL term (similar to beta-VAE)')
# flags.DEFINE_integer('num_epochs', 20, 'Number of training epochs')

# Physionet config
flags.DEFINE_integer('latent_dim', 35, 'Dimensionality of the latent space')
flags.DEFINE_list('encoder_sizes', [128, 128], 'Layer sizes of the encoder')
flags.DEFINE_list('decoder_sizes', [256, 256], 'Layer sizes of the decoder')
flags.DEFINE_integer('window_size', 24, 'Window size for the inference CNN: Ignored if model_type is not gp-vae')
flags.DEFINE_float('sigma', 1.005, 'Sigma value for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_float('length_scale', 7.0, 'Length scale value for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_float('beta', 0.2, 'Factor to weigh the KL term (similar to beta-VAE)')
flags.DEFINE_integer('num_epochs', 40, 'Number of training epochs')

# Flags with common default values for all three datasets
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('gradient_clip', 1e5, 'Maximum global gradient norm for the gradient clipping during training')
flags.DEFINE_integer('num_steps', 0, 'Number of training steps: If non-zero it overwrites num_epochs')
flags.DEFINE_integer('print_interval', 0, 'Interval for printing the loss and saving the model during training')
flags.DEFINE_string('exp_name', "debug", 'Name of the experiment')
flags.DEFINE_string('basedir', "models", 'Directory where the models should be stored')
flags.DEFINE_string('data_dir', "", 'Directory from where the data should be read in')
flags.DEFINE_enum('data_type', 'hmnist', ['hmnist', 'physionet', 'sprites', 'motion'], 'Type of data to be trained on')
flags.DEFINE_integer('seed', 1337, 'Seed for the random number generator')
flags.DEFINE_enum('model_type', 'gp-vae', ['vae', 'hi-vae', 'gp-vae'], 'Type of model to be trained')
flags.DEFINE_integer('cnn_kernel_size', 1, 'Kernel size for the CNN preprocessor')
flags.DEFINE_list('cnn_sizes', [256], 'Number of filters for the layers of the CNN preprocessor')
flags.DEFINE_boolean('testing', False, 'Use the actual test set for testing')
flags.DEFINE_boolean('banded_covar', False,
                     'Use a banded covariance matrix instead of a diagonal one for the output of the inference network: Ignored if model_type is not gp-vae')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')

flags.DEFINE_integer('M', 1, 'Number of samples for ELBO estimation')
flags.DEFINE_integer('K', 1, 'Number of importance sampling weights')

flags.DEFINE_enum('kernel', 'cauchy', ['rbf', 'diffusion', 'matern', 'cauchy'],
                  'Kernel to be used for the GP prior: Ignored if model_type is not (m)gp-vae')
flags.DEFINE_integer('kernel_scales', 1,
                     'Number of different length scales sigma for the GP prior: Ignored if model_type is not gp-vae')

# motion
flags.DEFINE_integer('Time', 10, 'length scales sigma for the GP prior: Ignored if model_type is not gp-vae')
flags.DEFINE_boolean('use_mixer', False, 'whether use mixer mlp instead of regular mlp layers')
flags.DEFINE_boolean('from_checkpoint', False, 'whether use checkpoint model')
flags.DEFINE_string('checkpoint_path', "", 'model to use (when from_checkpoint is on)')


def main(argv):
    del argv  # unused
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    print("Testing: ", FLAGS.testing, "\t Seed: {FLAGS.seed}")

    FLAGS.cnn_sizes = [int(size) for size in FLAGS.cnn_sizes]
    FLAGS.encoder_sizes = [int(size) for size in FLAGS.encoder_sizes]
    FLAGS.decoder_sizes = [int(size) for size in FLAGS.decoder_sizes]

    if 0 in FLAGS.encoder_sizes:
        FLAGS.encoder_sizes.remove(0)
    if 0 in FLAGS.decoder_sizes:
        FLAGS.decoder_sizes.remove(0)

    # Make up full exp name
    timestamp = datetime.now().strftime("%y%m%d")
    full_exp_name = "{}_{}".format(timestamp, FLAGS.exp_name)
    outdir = os.path.join(FLAGS.basedir, full_exp_name)
    if not os.path.exists(outdir): os.mkdir(outdir)
    checkpoint_prefix = os.path.join(outdir, "ckpt").replace('\\','/')
    print("Full exp name: ", full_exp_name)

    ###################################
    # Define data specific parameters #
    ###################################

    if FLAGS.data_type == "hmnist":
        FLAGS.data_dir = "data/hmnist/hmnist_mnar.npz"
        data_dim = 784
        time_length = 10
        num_classes = 10
        decoder = BernoulliDecoder
        img_shape = (28, 28, 1)
        val_split = 50000
    elif FLAGS.data_type == "physionet":
        if FLAGS.data_dir == "":
            FLAGS.data_dir = "data/physionet/physionet.npz"
        data_dim = 35
        time_length = 48
        num_classes = 2

        decoder = GaussianDecoder
    elif FLAGS.data_type == "sprites":
        if FLAGS.data_dir == "":
            FLAGS.data_dir = "data/sprites/sprites.npz"
        data_dim = 12288
        time_length = 8
        decoder = GaussianDecoder
        img_shape = (64, 64, 3)  # --> 40,40,100 --> 25,25,256
        val_split = 8000
    elif FLAGS.data_type == "motion":
        if FLAGS.data_dir == "":
            FLAGS.data_dir = "data/motion/motion_only.npz"
            # FLAGS.data_dir = "data/motion/motion_ds_9_1000_only.npz"
        data_dim = 330
        time_length = FLAGS.Time
        decoder = GaussianDecoder
        img_shape = (110, 3)  # --> t/4,65*3,100 --> t/8,30*3,256
        val_split = 8000

    else:
        raise ValueError("Data type must be one of ['hmnist', 'physionet', 'sprites']")

    #############
    # Load data #
    #############

    data = np.load(FLAGS.data_dir)
    if FLAGS.data_type in ['motion']:
        x_train_full = data['x_full'].reshape(-1, *img_shape)
        x_train_miss = data['x_miss'].reshape(-1, *img_shape)
        m_train_miss = data['m_miss'].reshape(-1, *img_shape)
    else:
        x_train_full = data['x_train_full']
        x_train_miss = data['x_train_miss']
        m_train_miss = data['m_train_miss']
        # y_train = x_train_full

    if FLAGS.data_type in ['hmnist', 'physionet']:
        y_train = data['y_train']

    if FLAGS.testing:
        if FLAGS.data_type in ['hmnist', 'sprites']:
            x_val_full = data['x_test_full']
            x_val_miss = data['x_test_miss']
            m_val_miss = data['m_test_miss']
        if FLAGS.data_type == 'hmnist':
            y_val = data['y_test']
        elif FLAGS.data_type == 'physionet':
            x_val_full = data['x_train_full']
            x_val_miss = data['x_train_miss']
            m_val_miss = data['m_train_miss']
            y_val = data['y_train']
            m_val_artificial = data["m_train_artificial"]
    elif FLAGS.data_type in ['hmnist', 'sprites']:
        x_val_full = x_train_full[val_split:]
        x_val_miss = x_train_miss[val_split:]
        m_val_miss = m_train_miss[val_split:]
        if FLAGS.data_type == 'hmnist':
            y_val = y_train[val_split:]
        x_train_full = x_train_full[:val_split]
        x_train_miss = x_train_miss[:val_split]
        m_train_miss = m_train_miss[:val_split]
        y_train = y_train[:val_split]
    elif FLAGS.data_type == 'physionet':
        x_val_full = data["x_val_full"]  # full for artificial missings
        x_val_miss = data["x_val_miss"]
        m_val_miss = data["m_val_miss"]
        m_val_artificial = data["m_val_artificial"]
        y_val = data["y_val"]
    elif FLAGS.data_type == 'motion':
        x_val_full = data['x_full']
        x_val_miss = data['x_miss']
        m_val_miss = data['m_miss']
    else:
        raise ValueError("Data type must be one of ['hmnist', 'physionet', 'sprites']")
    if FLAGS.data_type in ['motion']:
        x_val_full = data['x_full'].reshape(-1, *img_shape)
        x_val_miss = data['x_miss'].reshape(-1, *img_shape)
        m_val_miss = data['m_miss'].reshape(-1, *img_shape)

    if FLAGS.data_type in ['mon']:
        tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss, x_train_full)) \
            .window(FLAGS.Time, stride=int(FLAGS.Time / 4)).shuffle(len(x_train_miss)).batch(FLAGS.batch_size).repeat()
        tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss, x_val_full)).window(FLAGS.Time,
                                                                                                        stride=int(
                                                                                                            FLAGS.Time / 4)).batch(
            FLAGS.batch_size).repeat()
        tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)
    else:
        tf_x_train_miss = tf.data.Dataset.from_tensor_slices((x_train_miss, m_train_miss, x_train_full)) \
            .shuffle(len(x_train_miss)).batch(FLAGS.batch_size).repeat()
        tf_x_val_miss = tf.data.Dataset.from_tensor_slices((x_val_miss, m_val_miss, x_val_full)).batch(
            FLAGS.batch_size).repeat()
        tf_x_val_miss = tf.compat.v1.data.make_one_shot_iterator(tf_x_val_miss)

    # Build Conv2D preprocessor for image data
        print("Using CNN preprocessor")
    if FLAGS.data_type in ['motion'] and FLAGS.use_mixer:
        image_preprocessor = MotionPreprocessor(img_shape, img_shape, FLAGS.cnn_kernel_size)
    elif FLAGS.data_type in ['hmnist', 'sprites', 'motion']:
        image_preprocessor = ImagePreprocessor(img_shape, FLAGS.cnn_sizes, FLAGS.cnn_kernel_size)
    elif FLAGS.data_type in ['physionet']:
        image_preprocessor = None
    else:
        raise ValueError("Data type must be one of ['hmnist', 'physionet', 'sprites']")

    ###############
    # Build model #
    ###############

    if FLAGS.model_type == "vae":
        model = VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                    encoder_sizes=FLAGS.encoder_sizes, encoder=DiagonalEncoder,
                    decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                    image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                    beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    elif FLAGS.model_type == "hi-vae":
        model = HI_VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                       encoder_sizes=FLAGS.encoder_sizes, encoder=DiagonalEncoder,
                       decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K)
    elif FLAGS.model_type == "gp-vae":
        encoder = BandedJointEncoder if FLAGS.banded_covar else JointEncoder
        model = GP_VAE(latent_dim=FLAGS.latent_dim, data_dim=data_dim, time_length=time_length,
                       encoder_sizes=FLAGS.encoder_sizes, encoder=encoder,
                       decoder_sizes=FLAGS.decoder_sizes, decoder=decoder,
                       kernel=FLAGS.kernel, sigma=FLAGS.sigma,
                       length_scale=FLAGS.length_scale, kernel_scales=FLAGS.kernel_scales,
                       image_preprocessor=image_preprocessor, window_size=FLAGS.window_size,
                       beta=FLAGS.beta, M=FLAGS.M, K=FLAGS.K, data_type=FLAGS.data_type, use_mixer=FLAGS.use_mixer)
    else:
        raise ValueError("Model type must be one of ['vae', 'hi-vae', 'gp-vae']")



    ########################
    # Training preparation #
    ########################

    print("GPU support: ", tf.config.list_physical_devices('GPU'))

    print("Training...")
    print("Training...", FLAGS.batch_size)

    # use schedular LR
    if FLAGS.num_steps == 0:
        num_steps = FLAGS.num_epochs * len(x_train_miss) // FLAGS.batch_size
    else:
        num_steps = FLAGS.num_steps

    lr = FLAGS.learning_rate
    # lr = get_lr(num_steps, warmup_step=0)
    # global_steps = len(tf_x_train_miss)*FLAGS.
    # lr_schedule = LRSchedule(FLAGS.learning_rate, int(num_steps*FLAGS.warmup), FLAGS.lr_scheduler)

    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.optimizers.Adadelta()

    print("Encoder: ", model.encoder.net.summary())
    print("Decoder: ", model.decoder.net.summary())

    if model.preprocessor is not None:
        print("Preprocessor: ", model.preprocessor.net.summary())
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net,
                                              decoder=model.decoder.net, preprocessor=model.preprocessor.net,
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())
    else:
        saver = tf.compat.v1.train.Checkpoint(optimizer=optimizer, encoder=model.encoder.net, decoder=model.decoder.net,
                                              optimizer_step=tf.compat.v1.train.get_or_create_global_step())




    summary_writer = tf.compat.v2.summary.create_file_writer(logdir=outdir, flush_millis=10000)

    if FLAGS.print_interval == 0:
        FLAGS.print_interval = num_steps // FLAGS.num_epochs

    ############
    # Training #
    ############

    losses_train = []
    losses_val = []

    t0 = time.time()
    with summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
        for i, (x_seq, m_seq, x_full_seq) in enumerate(tf_x_train_miss.take(num_steps)):
            try:
                with tf.GradientTape() as tape:
                    tape.watch(trainable_vars)
                    x_seq_reshaped = tf.cast(x_seq.reshape(-1, FLAGS.Time, data_dim), tf.float32)
                    m_seq_reshaped = tf.cast(m_seq.reshape(-1, FLAGS.Time, data_dim), tf.float32)
                    # x_seq_reshaped = tf.cast(x_seq, tf.float32)
                    # m_seq_reshaped = tf.cast(m_seq, tf.float32)
                    elbo_loss, nll, kl, mse_loss = model.compute_loss(x_seq_reshaped, m_mask=m_seq_reshaped,
                                                                      return_parts=True)
                    # mse_loss = model.compute_mse(tf.cast(x_seq,tf.float32),tf.cast(x_full_seq,tf.float32))
                    loss = elbo_loss #+ mse_loss
                    # loss = mse_loss
                    # loss = mse_loss +(i/num_steps)*elbo_loss*int(i>5000)
                    losses_train.append(loss.numpy())
                grads = tape.gradient(loss, trainable_vars)
                grads = [np.nan_to_num(grad) for grad in grads]
                grads, global_norm = tf.clip_by_global_norm(grads, FLAGS.gradient_clip)
                # optimizer.apply_gradients(zip(grads, trainable_vars),
                #                           global_step=tf.compat.v1.train.get_or_create_global_step())
                optimizer.apply_gradients(zip(grads, trainable_vars))
                # Print intermediate results
                if i % FLAGS.print_interval == 0:
                    print("================================================")
                    print("Learning rate: {} | Global gradient norm: {:.2f}".format(optimizer.lr, global_norm))
                    print("Step {}) Time = {:2f}".format(i, time.time() - t0))
                    # elbo_loss, nll, kl = model.compute_loss(x_seq, m_mask=m_seq, return_parts=True)
                    # mse_loss = model.compute_mse(tf.cast(x_seq,tf.float32),tf.cast(x_full_seq,tf.float32))
                    print("Train loss = {:.3f} | NLL = {:.3f} | KL = {:.3f} | MSE = {:.3f}".format(loss,
                                                                                                   nll, kl, mse_loss))

                    saver.save(checkpoint_prefix)
                    tf.compat.v2.summary.scalar(name="loss_train", data=loss,
                                                step=i)
                    tf.compat.v2.summary.scalar(name="kl_train", data=kl,
                                                step=i)
                    tf.compat.v2.summary.scalar(name="nll_train", data=nll,
                                                step=i)

                    # Validation loss
                    x_val_batch, m_val_batch, x_full_val_batch = tf_x_val_miss.get_next()
                    val_loss, val_nll, val_kl, val_mse = model.compute_loss(
                        tf.cast(x_val_batch.reshape(-1, FLAGS.Time, data_dim), tf.float32),
                        m_mask=tf.cast(m_val_batch.reshape(-1, FLAGS.Time, data_dim), tf.float32), return_parts=True)
                    losses_val.append(val_loss.numpy())
                    print("Validation loss = {:.3f} | NLL = {:.3f} | KL = {:.3f} |  MSE = {:.3f}".format(val_loss,
                                                                                                         val_nll,
                                                                                                         val_kl,
                                                                                                         val_mse))

                    tf.compat.v2.summary.scalar(name="loss_val", data=val_loss,
                                                step=i)
                    tf.compat.v2.summary.scalar(name="kl_val", data=val_kl,
                                                step=i)
                    tf.compat.v2.summary.scalar(name="nll_val", data=val_nll,
                                                step=i)

                    if FLAGS.data_type in ["hmnist", "sprites"]:
                        # Draw reconstructed images
                        x_hat = model.decode(model.encode(x_seq).sample()).mean()
                        tf.compat.v2.summary.image(name="input_train", data=tf.reshape(x_seq, [-1] + list(img_shape)),
                                                   step=i)
                        tf.compat.v2.summary.image(name="reconstruction_train",
                                                   data=tf.reshape(x_hat, [-1] + list(img_shape)),
                                                   step=i)
                    elif FLAGS.data_type == 'physionet':
                        # Eval MSE and AUROC on entire val set
                        x_val_miss_batches = np.array_split(x_val_miss, FLAGS.batch_size, axis=0)
                        x_val_full_batches = np.array_split(x_val_full, FLAGS.batch_size, axis=0)
                        m_val_artificial_batches = np.array_split(m_val_artificial, FLAGS.batch_size, axis=0)
                        get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_artificial_batches)

                        n_missings = m_val_artificial.sum()
                        mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m).numpy()
                                           for x, y, m in get_val_batches()]) / n_missings

                        x_val_imputed = np.vstack([model.decode(model.encode(x_batch).mean()).mean().numpy()
                                                   for x_batch in x_val_miss_batches])
                        x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]  # impute gt observed values

                        x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])
                        val_split = len(x_val_imputed) // 2
                        cls_model = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
                        cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
                        probs = cls_model.predict_proba(x_val_imputed[val_split:])[:, 1]
                        auroc = roc_auc_score(y_val[val_split:], probs)
                        print("MSE miss: {:.4f} | AUROC: {:.4f}".format(mse_miss, auroc))

                        # Update learning rate (used only for physionet with decay=0.5)
                        if i > 0 and i % (10 * FLAGS.print_interval) == 0:
                            optimizer._lr = max(0.5 * optimizer._lr, 0.1 * FLAGS.learning_rate)
                    t0 = time.time()
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                if FLAGS.debug:
                    import ipdb
                    ipdb.set_trace()
                break

    ##############
    # Evaluation #
    ##############

    print("Evaluation...")

    # Split data on batches
    x_val_miss_batches = np.array_split(x_val_miss, FLAGS.batch_size, axis=0)
    x_val_full_batches = np.array_split(x_val_full, FLAGS.batch_size, axis=0)
    if FLAGS.data_type == 'physionet':
        m_val_batches = np.array_split(m_val_artificial, FLAGS.batch_size, axis=0)
    else:
        m_val_batches = np.array_split(m_val_miss, FLAGS.batch_size, axis=0)
    get_val_batches = lambda: zip(x_val_miss_batches, x_val_full_batches, m_val_batches)

    # Compute NLL and MSE on missing values
    n_missings = m_val_artificial.sum() if FLAGS.data_type == 'physionet' else m_val_miss.sum()
    nll_miss = np.sum([model.compute_nll(x, y=y, m_mask=m).numpy()
                       for x, y, m in get_val_batches()]) / n_missings
    mse_miss = np.sum([model.compute_mse(x, y=y, m_mask=m, binary=FLAGS.data_type == "hmnist").numpy()
                       for x, y, m in get_val_batches()]) / n_missings
    print("NLL miss: {:.4f}".format(nll_miss))
    print("MSE miss: {:.4f}".format(mse_miss))

    # Save imputed values
    z_mean = [model.encode(x_batch).mean().numpy() for x_batch in x_val_miss_batches]
    np.save(os.path.join(outdir, "z_mean"), np.vstack(z_mean))
    x_val_imputed = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_mean])
    np.save(os.path.join(outdir, "imputed_no_gt"), x_val_imputed)

    # impute gt observed values
    x_val_imputed[m_val_miss == 0] = x_val_miss[m_val_miss == 0]
    np.save(os.path.join(outdir, "imputed"), x_val_imputed)

    if FLAGS.data_type == "hmnist":
        # AUROC evaluation using Logistic Regression
        x_val_imputed = np.round(x_val_imputed)
        x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])

        cls_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-10, max_iter=10000)
        val_split = len(x_val_imputed) // 2

        cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
        probs = cls_model.predict_proba(x_val_imputed[val_split:])

        auprc = average_precision_score(np.eye(num_classes)[y_val[val_split:]], probs)
        auroc = roc_auc_score(np.eye(num_classes)[y_val[val_split:]], probs)
        print("AUROC: {:.4f}".format(auroc))
        print("AUPRC: {:.4f}".format(auprc))

    elif FLAGS.data_type == "sprites" or FLAGS.data_type == "motion":
        auroc, auprc = 0, 0
        for i in range(5):
            z_sample = [model.encode(x_batch).sample().numpy() for x_batch in x_val_miss_batches]
            np.save(os.path.join(outdir, "z_sample_{}".format(i)), np.vstack(z_sample))
            x_val_imputed_sample = np.vstack([model.decode(z_batch).mean().numpy() for z_batch in z_sample])
            np.save(os.path.join(outdir, "imputed_sample_{}_no_gt".format(i)), x_val_imputed_sample)
            x_val_imputed_sample[m_val_miss == 0] = x_val_miss[m_val_miss == 0]
            np.save(os.path.join(outdir, "imputed_sample_{}".format(i)), x_val_imputed_sample)

    elif FLAGS.data_type == "motion":
        # Uncomment to preserve some z_samples and their reconstructions

        # AUROC evaluation using Logistic Regression
        x_val_imputed = x_val_imputed.reshape([-1, time_length * data_dim])
        val_split = len(x_val_imputed) // 2
        cls_model = LogisticRegression(solver='liblinear', tol=1e-10, max_iter=10000)
        cls_model.fit(x_val_imputed[:val_split], y_val[:val_split])
        probs = cls_model.predict_proba(x_val_imputed[val_split:])[:, 1]
        auprc = average_precision_score(y_val[val_split:], probs)
        auroc = roc_auc_score(y_val[val_split:], probs)

        print("AUROC: {:.4f}".format(auroc))
        print("AUPRC: {:.4f}".format(auprc))

    # Visualize reconstructions
    if FLAGS.data_type in ["hmnist", "sprites"]:
        img_index = 0
        if FLAGS.data_type == "hmnist":
            img_shape = (28, 28)
            cmap = "gray"
        elif FLAGS.data_type == "sprites":
            img_shape = (64, 64, 3)
            cmap = None

        fig, axes = plt.subplots(nrows=3, ncols=x_val_miss.shape[1], figsize=(2 * x_val_miss.shape[1], 6))

        x_hat = model.decode(model.encode(x_val_miss[img_index: img_index + 1]).mean()).mean().numpy()
        seqs = [x_val_miss[img_index:img_index + 1], x_hat, x_val_full[img_index:img_index + 1]]

        for axs, seq in zip(axes, seqs):
            for ax, img in zip(axs, seq[0]):
                ax.imshow(img.reshape(img_shape), cmap=cmap)
                ax.axis('off')

        suptitle = FLAGS.model_type + " reconstruction, NLL missing = {}".format(mse_miss)
        fig.suptitle(suptitle, size=18)
        fig.savefig(os.path.join(outdir, FLAGS.data_type + "_reconstruction.pdf"))

    results_all = [FLAGS.seed, FLAGS.model_type, FLAGS.data_type, FLAGS.kernel, FLAGS.beta, FLAGS.latent_dim,
                   FLAGS.num_epochs, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.window_size,
                   FLAGS.kernel_scales, FLAGS.sigma, FLAGS.length_scale,
                   len(FLAGS.encoder_sizes), FLAGS.encoder_sizes[0] if len(FLAGS.encoder_sizes) > 0 else 0,
                   len(FLAGS.decoder_sizes), FLAGS.decoder_sizes[0] if len(FLAGS.decoder_sizes) > 0 else 0,
                   FLAGS.cnn_kernel_size, FLAGS.cnn_sizes,
                   nll_miss, mse_miss, losses_train[-1], losses_val[-1], auprc, auroc, FLAGS.testing, FLAGS.data_dir]

    with open(os.path.join(outdir, "results.tsv"), "w") as outfile:
        outfile.write("seed\tmodel\tdata\tkernel\tbeta\tz_size\tnum_epochs"
                      "\tbatch_size\tlearning_rate\twindow_size\tkernel_scales\t"
                      "sigma\tlength_scale\tencoder_depth\tencoder_width\t"
                      "decoder_depth\tdecoder_width\tcnn_kernel_size\t"
                      "cnn_sizes\tNLL\tMSE\tlast_train_loss\tlast_val_loss\tAUPRC\tAUROC\ttesting\tdata_dir\n")
        outfile.write("\t".join(map(str, results_all)))

    with open(os.path.join(outdir, "training_curve.tsv"), "w") as outfile:
        outfile.write("\t".join(map(str, losses_train)))
        outfile.write("\n")
        outfile.write("\t".join(map(str, losses_val)))

    print("Training finished.")


if __name__ == '__main__':
    app.run(main)
