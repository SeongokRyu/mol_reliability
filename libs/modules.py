import tensorflow as tf


from libs.layers import feed_forward_net
from libs.layers import GraphConv
from libs.layers import GraphAttn
from libs.layers import GraphGatherReadout
from libs.layers import LinearReadout
from libs.layers import PMAReadout


class NodeEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 out_dim, 
                 use_attn,
                 num_heads, 
                 use_ln, 
                 use_ffnn,
                 rate):
        super(NodeEmbedding, self).__init__()

        pre_act = True
        if use_ffnn:
            pre_act = False

        if use_attn:
            self.gconv = GraphAttn(out_dim, num_heads, pre_act)
        else:     
            self.gconv = GraphConv(out_dim, pre_act)

        self.use_ffnn = use_ffnn
        if use_ffnn:
            self.ffnn = feed_forward_net(out_dim)

        self.dropout = tf.keras.layers.Dropout(rate)

        self.layer_norm = tf.keras.layers.LayerNormalization()


    def call(self, x, adj, training):            
        h = x
        h = self.gconv(h, adj)

        if self.use_ffnn:
            h = self.ffnn(h)
        h = self.dropout(h, training=training)
        h += x
        h = self.layer_norm(h)             

        return h    


class GraphEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 out_dim,
                 readout_method):
        super(GraphEmbedding, self).__init__()

        if readout_method in ['mean', 'sum', 'max']:
            self.readout = LinearReadout(out_dim, readout_method)                
        elif readout_method == 'pma':
            self.readout = PMAReadout(out_dim, 4)                

    def call(self, x):
        return self.readout(x)   
