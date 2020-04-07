import tensorflow as tf


from libs.modules import NodeEmbedding
from libs.modules import GraphEmbedding


class PredictNet(tf.keras.Model):
    
    def __init__(self,
                 num_layers,
                 node_dim,
                 graph_dim,
                 use_attn=True,
                 num_heads=4,
                 use_ln=True,
                 use_ffnn=False,
                 dropout_rate=0.1,
                 readout_method='pma',
                 concat_readout=True,
                 last_activation=None):
        super(PredictNet, self).__init__()
 
        self.num_layers = num_layers
        self.concat_readout = concat_readout

        self.first_embedding = tf.keras.layers.Dense(node_dim, use_bias=False)
        self.node_embedding = [NodeEmbedding(node_dim, use_attn, num_heads, 
                                             use_ln, use_ffnn, dropout_rate) 
                               for _ in range(num_layers)]
        
        self.graph_embedding = [GraphEmbedding(graph_dim, readout_method)
                               for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(1, activation=last_activation)

    def call(self, x, adj, training):
        h = self.first_embedding(x)
        z_list = []
        for i in range(self.num_layers):
            h = self.node_embedding[i](h, adj, training)
            z = self.graph_embedding[i](h)
            z_list.append(z)

        z = tf.concat(z_list, axis=1)     

        final_output = self.dense(z)
        final_output = tf.reshape(final_output, [-1])
        return final_output
