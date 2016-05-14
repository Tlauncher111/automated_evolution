from tensorflow.models.image.cifar10 import cifar10_input
import tensorflow as tf
import numpy as np

RGB_channel = 3
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
rank_decay = 0.8  # min:0 max:1 calculate the priority for breeding (e.g if 0.8, 1st = 1, 2nd = 0.8, 3rd = 0.64...)
mutation_probability = 0.4  # min:0 max:1


def find_layer_size(clusters, max_layer_size):
    layer_size = 0
    for i in xrange(max_layer_size):
        if clusters[i][0] == 0:
            break
        layer_size += 1
    return layer_size


def breed_genes(clusters_all, connections_all, precision_list, min_layer_size, max_layer_size, max_nodes):
    network_total = len(precision_list)

    max_cluster_length = clusters_all[0].shape[1]
    print("max_cluster_length", max_cluster_length)

    # dict = precision with index
    dict = {}
    for i, precision in enumerate(precision_list):
        dict[i] = precision

    # sort the dict
    rank_index_convert_list = []
    rank_factor_index = 1
    rank_sum = 0
    for k, v in sorted(dict.items(), key=lambda x: x[1]):
        rank_index_convert_list.append(k)
        rank_sum += rank_factor_index
        rank_factor_index *= rank_decay
    rank_index_convert_list.reverse()

    # find which genes are used as parents = parents_index_list
    pre_prob_rnd = np.random.uniform(0, rank_sum, network_total * 2)
    parents_index_list = []
    for prob_rnd in pre_prob_rnd:
        judge_rnd = 0
        rank_factor_index = 1
        for i in xrange(network_total):
            judge_rnd += rank_factor_index
            if prob_rnd <= judge_rnd:
                parents_index_list.append(rank_index_convert_list[i])
                break
            else:
                rank_factor_index *= rank_decay

    # breeding genes

    # new_clusters_all, new_connections_all = [], []
    new_clusters_all = np.zeros(shape=(network_total, max_layer_size, max_cluster_length)).astype(np.int32)
    new_connections_all = np.zeros(shape=(network_total, max_layer_size, max_cluster_length, max_cluster_length)).astype(np.int32)

    parent1_clusters = np.zeros([max_layer_size, max_cluster_length]).astype(np.int32)
    parent2_clusters = np.zeros([max_layer_size, max_cluster_length]).astype(np.int32)
    parent1_connections = np.zeros([max_layer_size * max_cluster_length ** 2]).astype(np.int32)
    parent1_connections = parent1_connections.reshape((max_layer_size, max_cluster_length, max_cluster_length))
    parent2_connections = np.zeros([max_layer_size * max_cluster_length ** 2]).astype(np.int32)
    parent2_connections = parent2_connections.reshape((max_layer_size, max_cluster_length, max_cluster_length))

    for i in xrange(0, network_total * 2, 2):
        parent1_index = parents_index_list[i]
        parent2_index = parents_index_list[i + 1]

        # if parent1 is the same as parent2, use the other code
        if parent1_index == parent2_index:
            parent2_index -= 1
            if parent2_index < 0:
                parent2_index = 1

        # mix genes by 2 point crossover
        parent1_clusters, parent1_connections = clusters_all[parent1_index], connections_all[parent1_index]
        parent2_clusters, parent2_connections = clusters_all[parent2_index], connections_all[parent2_index]

        # mutation
        mutation_rnd = np.random.rand()
        if mutation_rnd < mutation_probability:
            print "MUTATION"
            parent2_clusters, parent2_connections = generate_genes(min_layer_size, max_layer_size, max_cluster_length, max_nodes)


    for i in xrange(0, network_total * 2, 2):
        start_pos = np.random.randint(0, max_layer_size)
        end_pos = start_pos + np.random.randint(0, max_layer_size)
        if end_pos > max_layer_size:
            end_pos = max_layer_size

        # swap clusters and connections
        parent1_clusters[start_pos:end_pos] = parent2_clusters[start_pos:end_pos]
        parent1_connections[start_pos:end_pos] = parent2_connections[start_pos:end_pos]

        layer_size = find_layer_size(parent1_clusters, max_layer_size)

        # make new connections again for new clusters and swap only new parts
        new_connections = make_connections(parent1_clusters, layer_size, max_layer_size, max_cluster_length)
        # print "new_connections", new_connections
        connect_start_pos = max(start_pos - 1, 0)
        # print "connect_start_pos", connect_start_pos, new_connections[connect_start_pos:connect_start_pos+1]
        parent1_connections[connect_start_pos:connect_start_pos + 1] =\
            new_connections[connect_start_pos:connect_start_pos + 1]
        parent1_connections[end_pos - 1:end_pos] = new_connections[end_pos - 1:end_pos]

        new_clusters_all[i//2] = parent1_clusters
        new_connections_all[i//2] = parent1_connections

    return new_clusters_all, new_connections_all


def make_parts(min_value, max_value, size):  # return numpy which does not have all zeros
    while True:
        parts = np.random.randint(min_value, max_value + 1, size)
        if np.array_equal(parts, np.zeros([size]).astype(np.int32)):
            pass
        else:
            break
    return parts


def make_connections(m_all_layer_clusters, m_layer_size, m_max_layer_size, m_max_cluster_length):
    all_layer_connections = np.zeros([m_max_layer_size * m_max_cluster_length ** 2]).astype(np.int32)
    all_layer_connections = all_layer_connections.reshape((m_max_layer_size, m_max_cluster_length, m_max_cluster_length))

    for layer_index in xrange(m_layer_size - 1):
        one_layer_clusters = m_all_layer_clusters[layer_index]
        next_layer_clusters = m_all_layer_clusters[layer_index + 1]

        for i, cluster in enumerate(one_layer_clusters):
            if cluster == 0:  # if cluster does not exit, no connections exitst
                connections_from_one = np.zeros([m_max_cluster_length]).astype(np.int32)
            else:  # cluster exists
                while True:
                    connections_from_one = make_parts(0, 1, m_max_cluster_length)
                    # check whether the destination cluster exists
                    for k, next_cluster in enumerate(next_layer_clusters):
                        if next_cluster == 0:
                            connections_from_one[k] = 0

                    if not np.array_equal(connections_from_one, np.zeros([m_max_cluster_length]).astype(np.int32)):
                        break  # break if connections exists

            all_layer_connections[layer_index][i] = connections_from_one

        # connect if clusters receive no connections
        for k, next_cluster in enumerate(next_layer_clusters):
            if next_cluster != 0:
                # print "all_layer_connections[layer_index][k]", all_layer_connections[layer_index][k]
                connection_total = 0
                for i in xrange(m_max_cluster_length):
                    connection_total += all_layer_connections[layer_index][i][k]
                if connection_total == 0:  # no connections
                    # print "CONNECT next_layer_index=%d next_cluster_index=%d" % (layer_index+1, k)
                    all_layer_connections[layer_index][0][k] = 1  # connect to the most left cluster which always exists(but maybe it should connect randomly)

    # connect all in last layer
    for i in xrange(m_max_cluster_length):
        if m_all_layer_clusters[m_layer_size - 1][i] != 0:
            last_output = np.zeros([m_max_cluster_length]).astype(np.int32)
            last_output[0] = 1  # connect to only most left(final output)
            all_layer_connections[m_layer_size - 1][i] = last_output
    return all_layer_connections


def generate_genes(min_layer_size, max_layer_size, max_cluster_length, max_nodes):  # for one network
    # make clusters
    all_layer_clusters = np.zeros([max_layer_size, max_cluster_length]).astype(np.int32)
    layer_size = np.random.randint(min_layer_size, max_layer_size + 1)
    for layer_index in xrange(layer_size):
        cluster_length = np.random.randint(1, max_cluster_length + 1)
        clusters_parts = make_parts(1, max_nodes, cluster_length)
        all_layer_clusters[layer_index][0:cluster_length] = clusters_parts

    # make connections
    all_layer_connections = make_connections(all_layer_clusters, layer_size, max_layer_size, max_cluster_length)

    return all_layer_clusters, all_layer_connections


def inference(images, l_clusters, l_connections):
    with tf.name_scope('inference') as scope:
        def weight_variable(shape, i, j, k, kind=""):
            # print("W SHAPE!", shape, kind)
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name="W" + str(shape[0]) + "_" + kind + str(i) + str(j) + str(k))

        def bias_variable(shape, i, j, k):
            initial = tf.constant(0., shape=shape)
            return tf.Variable(initial, name="B" + str(shape[0]) + "_" + str(i) + str(j) + str(k))

        def Batch_Normalization(layer_input, variable_shape, j, k):
            beta = weight_variable(variable_shape, j, k, 0, "bn_b")
            gamma = weight_variable(variable_shape, j, k, 0, "bn_g")
            mean, variance = tf.nn.moments(layer_input, [0])
            epsilon = 0.00001
            return beta + gamma * (layer_input - mean) / tf.sqrt(variance + epsilon)

        max_layer_size = l_clusters.shape[0]
        hidden_layer_total = find_layer_size(l_clusters, max_layer_size)
        max_cluster_length = l_clusters.shape[1]

        # make all weights of connections(including inactive connections, maybe it's better not to make inactive one)

        # define weights and biases
        input_layer_w = []
        input_layer_b = []
        for next_cluster_index in xrange(max_cluster_length):
            input_size = IMAGE_SIZE * IMAGE_SIZE * RGB_channel
            output_size = l_clusters[0][next_cluster_index]
            if output_size == 0:
                break
            weights = weight_variable([input_size, output_size], -1, next_cluster_index, 0)
            biases = bias_variable([output_size], -1, next_cluster_index, 0)
            input_layer_w.append(weights)
            input_layer_b.append(biases)

        h_layer_clusters_w = []  # weight
        h_layer_clusters_b = []  # bias
        for h_layer_index in xrange(0, hidden_layer_total):
            one_layer_clusters_w = []  # weight
            one_layer_clusters_b = []  # bias
            for cluster_index in xrange(max_cluster_length):
                input_size = l_clusters[h_layer_index][cluster_index]
                one_cluster_w = []
                one_cluster_b = []
                for next_cluster_index in xrange(max_cluster_length):
                    if h_layer_index == hidden_layer_total - 1:  # last layer
                        output_size = NUM_CLASSES
                    else:
                        output_size = l_clusters[h_layer_index + 1][next_cluster_index]  # !!!TRY some patterns later

                    connection = l_connections[h_layer_index][cluster_index][next_cluster_index]
                    if connection == 0:
                        weights = "none"
                        biases = "none"
                        # need this to refer by index including empty weights
                    else:
                        weights = weight_variable([input_size, output_size], h_layer_index, next_cluster_index, cluster_index)
                        biases = bias_variable([output_size], h_layer_index, next_cluster_index, cluster_index)

                    one_cluster_w.append(weights)
                    one_cluster_b.append(biases)

                one_layer_clusters_w.append(one_cluster_w)
                one_layer_clusters_b.append(one_cluster_b)

            h_layer_clusters_w.append(one_layer_clusters_w)
            h_layer_clusters_b.append(one_layer_clusters_b)

        # calculate output from input layer
        input_data = images
        outputs_from_input_layer = []
        for input_weights, input_biases in zip(input_layer_w, input_layer_b):
            pre_out = tf.matmul(input_data, input_weights) + input_biases
            output_from_input = tf.nn.relu(pre_out)
            outputs_from_input_layer.append(output_from_input)

        # calculate output from hidden layer
        hidden_output_list = []
        for h_layer_index in xrange(0, hidden_layer_total):
            one_layer_output = []
            prev_layer_index = h_layer_index - 1
            # calculate all outputs into one cluster of next layer
            for next_cluster_index in xrange(max_cluster_length):
                outputs_to_one_cluster = []
                for cluster_index in xrange(max_cluster_length):
                    with tf.name_scope("connect" + str(h_layer_index) + str(next_cluster_index) + str(cluster_index)) as scope:
                        l_weights = h_layer_clusters_w[h_layer_index][cluster_index][next_cluster_index]
                        l_biases = h_layer_clusters_b[h_layer_index][cluster_index][next_cluster_index]

                        if isinstance(l_weights, str):  # weights = "none" means not-connected
                            pass
                        else:
                            if h_layer_index == 0:
                                input_data = outputs_from_input_layer[cluster_index]
                            else:
                                input_data = hidden_output_list[prev_layer_index][cluster_index]
                            # connected
                            pre_hidden = tf.matmul(input_data, l_weights) + l_biases

                            if h_layer_index == hidden_layer_total - 1:
                                output_size = NUM_CLASSES
                            else:
                                output_size = l_clusters[h_layer_index + 1][next_cluster_index]

                            batch_normed = Batch_Normalization(pre_hidden, [output_size], h_layer_index, next_cluster_index)
                            hidden = tf.nn.relu(batch_normed)
                            outputs_to_one_cluster.append(hidden)

                if len(outputs_to_one_cluster) > 0:
                    output_for_one_cl = outputs_to_one_cluster[0]
                    for i in xrange(1, len(outputs_to_one_cluster)):
                        output_for_one_cl += outputs_to_one_cluster[i]
                    one_layer_output.append(output_for_one_cl)

            hidden_output_list.append(one_layer_output)

        #define like this, because it seems not convenient for tensorflow to define empty variable
        final_logits = hidden_output_list[hidden_layer_total - 1][0]
        for i in xrange(1, len(hidden_output_list[hidden_layer_total-1])):
            final_logits += hidden_output_list[hidden_layer_total - 1][i]

    return final_logits