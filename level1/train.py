from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import shutil
import os
from six.moves import xrange
import tensorflow as tf
import automated
import ae_cifar10_eval
import ae_cifar10
import parameters


'''LOCAL PARAMETERS SETTING'''
MAX_STEPS = 50
MIN_LAYER_SIZE = 6  # minimum = 2
MAX_LAYER_SIZE = 8
MAX_CLUSTER_LENGTH = 5
MAX_NODES = 500  # for one cluster
NETWORKS_TOTAL = 10  # for one generation, minimum = 2
REC_STEP = 10
GENERATION_TOTAL = 10000000  # how many times to breed
SAVER_OVERWRITE = parameters.save_overwrite  # overwrite checkpoint file (i.e) only one checkpoint file is left for all training
LOAD_GENERATION = 'none'  # load generation file set "none" if not necessary (e.g)'g_0_all_clusters.npy'
LOAD_GENERATION_C = 'none'  # load generation file set "none" if not necessary (e.g)'g_0_all_connections.npy'
# NOTE generation file must be moved in advance, because it will be removed when you start new experiment

PARENT_CHECKPOINT_DIR = parameters.parent_checkpoint_dir
CLUSTERS_CONNS_DIR = parameters.clusters_conns_dir
resultlog_txt = PARENT_CHECKPOINT_DIR + '/resultlog.txt'
BATCH_SIZE = parameters.BATCH_SIZE

def make_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def train(net_number, l_clusters, l_connections):
    print("net_number=<%d> training starts" % net_number)

    with tf.Graph().as_default():

        child_checkpoint_dir = PARENT_CHECKPOINT_DIR + "/" + str(net_number)
        make_directory(child_checkpoint_dir)

        global_step = tf.Variable(0, trainable=False)

        images, labels = ae_cifar10.distorted_inputs()

        # reshape 24 x 24 x 3 image tensor to 1-d vector
        reshaped_images = tf.reshape(images, [BATCH_SIZE, -1])

        print("INFERENCE")
        logits = automated.inference(reshaped_images, l_clusters, l_connections)

        loss = ae_cifar10.loss(logits, labels)

        print("TRAIN_OP")
        train_op = ae_cifar10.train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        init = tf.initialize_all_variables()

        sess = tf.Session()
        print("SESSION RUN")
        sess.run(init)

        # Start queue and Stop them later to avoid memory error
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in xrange(MAX_STEPS):
            start_time = time.time()
            empty, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            if step % REC_STEP == 0:
                sec_per_batch = float(duration)
                print("step %d, loss = %.2f, %.1f sec/batch" % (step, loss_value, sec_per_batch))

            #save checkpoint in the end
            if (step + 1) == MAX_STEPS:
                print("SAVE checkpoint")
                checkpoint_path = os.path.join(child_checkpoint_dir, 'model.ckpt')
                if SAVER_OVERWRITE == True:
                    print("OVERWRITE")
                    checkpoint_path = os.path.join(PARENT_CHECKPOINT_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def code_to_str(l_clusters, l_connections):
    code_str = ""
    #print("np.ravel(l_clusters)",np.ravel(l_clusters))
    for element in np.hstack((np.ravel(l_clusters), np.ravel(l_connections))):
        code_str += str(element)
    return code_str


def write_log_str(w_str):
    global resultlog_txt
    f = open(resultlog_txt, 'a')
    f.write(w_str)
    f.close()


def main(argv=None):
    ae_cifar10.maybe_download_and_extract()

    # make directory
    global PARENT_CHECKPOINT_DIR, resultlog_txt
    make_directory(PARENT_CHECKPOINT_DIR)
    make_directory(CLUSTERS_CONNS_DIR)
    if os.path.exists(resultlog_txt):
        os.remove(resultlog_txt)

    # generate_genes
    clusters_all = np.zeros(shape=(NETWORKS_TOTAL, MAX_LAYER_SIZE, MAX_CLUSTER_LENGTH)).astype(np.int32)
    connections_all = np.zeros(shape=(NETWORKS_TOTAL, MAX_LAYER_SIZE, MAX_CLUSTER_LENGTH, MAX_CLUSTER_LENGTH)).astype(np.int32)
    for i in xrange(NETWORKS_TOTAL):
        one_clusters, one_connections = automated.generate_genes(MIN_LAYER_SIZE, MAX_LAYER_SIZE, MAX_CLUSTER_LENGTH, MAX_NODES)
        clusters_all[i] = one_clusters
        connections_all[i] = one_connections

    # load generation file if needed
    if LOAD_GENERATION != "none":
        clusters_all = np.load(LOAD_GENERATION)
        connections_all = np.load(LOAD_GENERATION_C)

    # avoid training same code
    code_precision_dict = {}

    # training
    net_number = 0
    for g in xrange(GENERATION_TOTAL):
        print("GENERATION", g)
        write_log_str("GENERATION" + str(g) + os.linesep)
        precision_list = []
        for i, (clusters, connections) in enumerate(zip(clusters_all, connections_all)):
            print("clusters" + os.linesep, clusters)
            print("connections" + os.linesep, connections)
            code_str = code_to_str(clusters, connections)

            if code_precision_dict.has_key(code_str):# check whether the code was trained before
                print("TRAINED BEFORE = ", code_str)
                precision = code_precision_dict[code_str]
            else:
                trainstart = time.time()
                train(net_number, clusters, connections)
                trainelapsed_min = (time.time() - trainstart)/60
                print("TRAIN TIME = " + str(trainelapsed_min))
                precision = ae_cifar10_eval.main(net_number, clusters, connections)
                code_precision_dict[code_str] = precision

            precision_list.append(precision)
            write_log_str("netnumber=<" + str(net_number) + ">" + os.linesep + "precision=" + str(precision) + os.linesep + code_str + os.linesep)

            np.save(CLUSTERS_CONNS_DIR + '/g_' + str(g)+ 'n_' + str(net_number)+ '_clusters.npy', clusters)
            np.save(CLUSTERS_CONNS_DIR + '/g_' + str(g) + 'n_' + str(net_number) + '_connections.npy', connections)
            net_number += 1

        # save genes info of one generation
        np.save(CLUSTERS_CONNS_DIR + '/g_' + str(g) + '_all_clusters.npy', clusters_all)
        np.save(CLUSTERS_CONNS_DIR + '/g_' + str(g) + '_all_connections.npy', connections_all)

        print("UPDATE CODES")
        clusters_all, connections_all = automated.breed_genes(clusters_all, connections_all, precision_list, MIN_LAYER_SIZE, MAX_LAYER_SIZE, MAX_NODES)#generate_new_code

        print("RESULT > code, precision")
        for k, v in sorted(code_precision_dict.items(), key=lambda x: x[1]):
            print(k + os.linesep + str(v))


if __name__ == '__main__':
    tf.app.run()
