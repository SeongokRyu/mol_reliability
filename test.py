import argparse
import sys
import os
import time


from absl import logging
from absl import app


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from libs.predict_net import PredictNet
from libs.dataset import get_dataset
from libs.utils import set_cuda_visible_device
from libs.utils import get_metric_list

FLAGS = None
np.set_printoptions(3)
tf.random.set_seed(1234)
cmd = set_cuda_visible_device(1)
print ("Using ", cmd[:-1], "-th GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = cmd[:-1]


def save_outputs(model, dataset, metrics, model_name, mc_dropout=False):
    
    label_total = np.empty([0,])
    pred_total = np.empty([0,])

    st = time.time()
    for batch, (x, adj, label) in enumerate(dataset):

        pred = None
        if mc_dropout:
            pred = [model(x, adj, True) for _ in range(FLAGS.mc_sampling)]
            pred = tf.reduce_mean(pred, axis=0)
        else:    
            pred = model(x, adj, False)

        label_total = np.concatenate((label_total, label.numpy()), axis=0)
        pred_total = np.concatenate((pred_total, pred.numpy()), axis=0)

    et = time.time()

    for metric in metrics:
        metric.reset_states()
        
    model_name += '_' + str(mc_dropout)
    np.save('./outputs/'+model_name+'_label.npy', label_total)
    np.save('./outputs/'+model_name+'_pred.npy', pred_total)
    
    return


def test(model):

    model_name = FLAGS.prefix
    model_name += '_' + FLAGS.prop_train
    model_name += '_' + str(FLAGS.seed)
    model_name += '_' + str(FLAGS.use_attn)
    model_name += '_' + str(FLAGS.readout_method)
    model_name += '_' + str(FLAGS.dropout_rate)
    model_name += '_' + str(FLAGS.prior_length)
    model_name += '_' + str(FLAGS.label_smoothing)
    model_name += '_' + str(FLAGS.beta_erl)
    model_name += '_' + str(FLAGS.loss_type)
    ckpt_path = './save/'+model_name

    train_ds, test_ds, num_total, num_train = get_dataset(
        prop=FLAGS.prop_test, 
        batch_size=FLAGS.batch_size,
        train_ratio=0.0,
        seed=FLAGS.seed,
        shuffle=False
    )
    print ("Number of training and test data:", num_train, num_total-num_train)

    step = tf.Variable(0, trainable=False)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[FLAGS.decay_steps, FLAGS.decay_steps*2],
        values=[1.0, 0.1, 0.01],
    )
    lr = lambda: FLAGS.init_lr*schedule(step)

    coeff = FLAGS.prior_length * (1.0-FLAGS.dropout_rate)
    wd = lambda: coeff*schedule(step)

    optimizer = tfa.optimizers.AdamW(
        weight_decay=wd,
        learning_rate=lr,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2,
        epsilon=FLAGS.opt_epsilon
    )    

    checkpoint = tf.train.Checkpoint(
        model=model, 
        optimizer=optimizer
    )

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, 
        directory=ckpt_path, 
        max_to_keep=FLAGS.max_to_keep
    )

    metrics = get_metric_list(FLAGS.loss_type)

    status = checkpoint.restore(ckpt_manager.latest_checkpoint)

    model_name = FLAGS.prefix
    model_name += '_' + FLAGS.prop_train
    model_name += '_' + str(FLAGS.seed)
    model_name += '_' + str(FLAGS.use_attn)
    model_name += '_' + str(FLAGS.readout_method)
    model_name += '_' + str(FLAGS.dropout_rate)
    model_name += '_' + str(FLAGS.prior_length)
    model_name += '_' + str(FLAGS.label_smoothing)
    model_name += '_' + str(FLAGS.beta_erl)
    model_name += '_' + str(FLAGS.loss_type)
    model_name += '_' + FLAGS.prop_test

    save_outputs(model, test_ds, metrics, model_name, False)
    if FLAGS.mc_sampling > 0:
        save_outputs(model, test_ds, metrics, model_name, True)

    return


def main(_):

    def print_model_spec():
        print ("Target property", FLAGS.prop_test)
        print ("Random seed for data spliting", FLAGS.seed)
        print ("Number of graph convoltuion layers", FLAGS.num_layers)
        print ("Dimensionality of node features", FLAGS.node_dim)
        print ("Dimensionality of graph features", FLAGS.graph_dim)
        print ()
        print ("Whether to use attentions in node embeddings", \
                                                     FLAGS.use_attn)
        print ("Number of attention heads", FLAGS.num_heads)
        print ("Whether to use layer normalization", FLAGS.use_ln)
        print ("Whether to use feed-forward network", FLAGS.use_ffnn)
        print ("Dropout rate", FLAGS.dropout_rate)
        print ("Weight decay coeff", FLAGS.prior_length)
        print ()
        print ("Readout method", FLAGS.readout_method)
        print ()
        print ("Loss function", FLAGS.loss_type)
        print ("Train property", FLAGS.prop_train)
        print ("Test property", FLAGS.prop_test)
        print ("Random seed for data spliting", FLAGS.seed)
        print ("Number of graph convoltuion layers", FLAGS.num_layers)
        print ("Dimensionality of node features", FLAGS.node_dim)
        print ("Dimensionality of graph features", FLAGS.graph_dim)
        print ()
        print ("Whether to use attentions in node embeddings", \
                                                     FLAGS.use_attn)
        print ("Number of attention heads", FLAGS.num_heads)
        print ("Whether to use feed-forward nets", FLAGS.use_ffnn)
        print ("Whether to use layer normalization", FLAGS.use_ln)
        print ("Dropout rate", FLAGS.dropout_rate)
        print ("Weight decay coeff", FLAGS.prior_length)
        print ()
        print ("Readout method", FLAGS.readout_method)
        print ()
        print ("Loss function", FLAGS.loss_type)
        print ("Label smoothing", FLAGS.label_smoothing)
        print ("Entropy regularization", FLAGS.beta_erl)
        return
    
    last_activation = tf.nn.sigmoid
    if FLAGS.loss_type in ['mse']:
        last_activation=None
    
    model = PredictNet(
        num_layers=FLAGS.num_layers,
        node_dim=FLAGS.node_dim,
        graph_dim=FLAGS.graph_dim,
        use_attn=FLAGS.use_attn,
        num_heads=FLAGS.num_heads,
        use_ln=FLAGS.use_ln,
        use_ffnn=FLAGS.use_ffnn,
        dropout_rate=FLAGS.dropout_rate,
        readout_method=FLAGS.readout_method,
        concat_readout=FLAGS.concat_readout,
        last_activation=last_activation
    )

    print_model_spec()

    test(model)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False    
        else:
            raise argparse.ArgumentTypeEror('Boolean value expected')    

    # Hyper-parameters for prefix, prop and random seed
    parser.add_argument('--prefix', type=str, default='Exp5', 
                        help='Prefix for this training')
    parser.add_argument('--prop_train', type=str, default='egfr_dude', 
                        help='Target property to train')
    parser.add_argument('--prop_test', type=str, default='egfr_chembl_7.0', 
                        help='Target property to train')
    parser.add_argument('--seed', type=int, default=1111, 
                        help='Random seed will be used to shuffle dataset')
    parser.add_argument("--oversampling", type=str2bool, default=False, 
                        help='Whether to oversample positive/negative data points')
    parser.add_argument('--pos_ratio', type=float, default=0.5, 
                        help='Ratio of positive samples')

    # Hyper-parameters for model construction
    parser.add_argument('--num_layers', type=int, default=4, 
                        help='Number of node embedding layers')
    parser.add_argument('--node_dim', type=int, default=64, 
                        help='Dimension of node embeddings')
    parser.add_argument('--graph_dim', type=int, default=256, 
                        help='Dimension of a graph embedding')
    parser.add_argument("--use_attn", type=str2bool, default=False, 
                        help='Whether to use multi-head attentions')
    parser.add_argument('--num_heads', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument("--use_ln", type=str2bool, default=False, 
                        help='Whether to use layer normalizations')
    parser.add_argument('--use_ffnn', type=str2bool, default=False, 
                        help='Whether to use feed-forward nets')
    parser.add_argument('--dropout_rate', type=float, default=0.0, 
                        help='Dropout rates in node embedding layers')
    parser.add_argument('--prior_length', type=float, default=1e-4, 
                        help='Weight decay coefficient')
    parser.add_argument('--readout_method', type=str, default='pma', 
                        help='Readout method to be used')
    parser.add_argument('--concat_readout', type=str2bool, default=True, 
                        help='Whether to concatenate readout vectors')

    # Hyper-parameaters for loss function
    parser.add_argument('--loss_type', type=str, default='bce', 
                        help='Loss function will be used, \
                             Options: bce, mse, focal, class_balanced, max_margin')
    parser.add_argument('--label_smoothing', type=float, default=0.0, 
                        help='Coefficient for label smoothing')
    parser.add_argument('--beta_erl', type=float, default=0.0, 
                        help='Coefficient for entropy regularization (ERL)')



    # Hyper-parameters for training
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    parser.add_argument('--num_epoches', type=int, default=100, 
                        help='Number of epoches')
    parser.add_argument('--init_lr', type=float, default=1e-3, 
                        help='Initial learning rate,\
                              Do not need for warmup scheduling')
    parser.add_argument('--beta_1', type=float, default=0.9, 
                        help='Beta1 in adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, 
                        help='Beta2 in adam optimizer')
    parser.add_argument('--opt_epsilon', type=float, default=1e-7, 
                        help='Epsilon in adam optimizer')
    parser.add_argument('--decay_steps', type=int, default=40, 
                        help='Decay steps for stair learning rate scheduling')
    parser.add_argument('--decay_rate', type=float, default=0.1, 
                        help='Decay rate for stair learning rate scheduling')
    parser.add_argument('--max_to_keep', type=int, default=5, 
                        help='Maximum number of checkpoint files to be kept')
    parser.add_argument("--save_model", type=str2bool, default=True, 
                        help='Whether to save checkpoints')
    parser.add_argument("--reload_ckpt", type=str2bool, default=True, 
                        help='Whether to reload the last checkpoint')


    # Hyper-parameters for evaluation
    parser.add_argument("--save_outputs", type=str2bool, default=True, 
                        help='Whether to save final predictions for test dataset')
    parser.add_argument('--mc_dropout', type=str2bool, default=False, 
                        help='Whether to infer predictive distributions with MC-dropout')
    parser.add_argument('--mc_sampling', type=int, default=0,
                       help='Number of MC sampling')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k instances for evaluating Precision or Recall')

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

