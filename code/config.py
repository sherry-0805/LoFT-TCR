
from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")

# Data directory
parser.add_argument("-pd", "--pridir", default="../data", help="Primary directory of data")
parser.add_argument("-sd", "--secdir", default="iedb_McPAS_vdjdb1_2_3_5folds", help="Secondary directory of data")
parser.add_argument("-td", "--terdir", default="fold0", help="Tertiary directory of data")

# Parameter setting
parser.add_argument("-m", "--mode", choices=["binary", "trinary"], default="trinary", help="Choose the classification type: 'binary' or 'trinary'")
parser.add_argument("-e", "--epochs", default=1000, type=int, help="Number of training epochs")
parser.add_argument("-w1", "--weight_decay_layer_one", default=0.0, type=float, help="Weight_decay of model.encoder.geomgcn1")
# parser.add_argument("-w2", "--weight_decay_layer_two", default=5e-06, type=float, help="Weight_decay of model.encoder.geomgcn2")
parser.add_argument("-w3", "--weight_decay_mlp", default=0.0, type=float, help="Weight_decay of model.decoder")

parser.add_argument("-geomlr", "--geomlr", default=1e-3, type=float, help="Learning rate of ESMGNN")
parser.add_argument("-l1", "--latent_weight", default=0.1, type=float, help="The weight of latent space")
parser.add_argument("-l2", "--selfloop_weight", default=0.1, type=float, help="The weight of selfloop")
parser.add_argument("-numhidden", "--num_hidden", default=1024, type=int, help="Output dimension of the first layer")
parser.add_argument("-numdivisions", "--num_divisions", default=7, type=int, help="The number of subGraph")
parser.add_argument("-heads1", "--num_heads_layer_one", default=1, type=int, help="num_heads_layer_one")
parser.add_argument("-dropout", "--dropout_rate", default=0.2, type=float, help="Dropout rate")
parser.add_argument("-layer_one_ggcn_merge", "--layer_one_ggcn_merge", default='sum', type=str, help="")
parser.add_argument("-layer_one_channel_merge", "--layer_one_channel_merge", default='mean', type=str, help="")

parser.add_argument("-dimre", "--dimReduction", default='umap', type=str, help="The way of mapping node to latent sapace.eg:umap/isomap" )
parser.add_argument("-construct_neighbor", "--construct_neighbor", default='rho', type=str, help="The way of construct neighbor.eg:rho/K")

# Models & History save directory
parser.add_argument("-md", "--modeldir", default="../model", help="Primary directory of models save directory")
parser.add_argument("-hd", "--hisdir", default="../History", help="Primary directory of history save directory")
parser.add_argument("-tmd", "--testmodeldir", default="iedb_McPAS_vdjdb1_2_3_5folds", help="Secondary directory of test model directory")
# cuda
parser.add_argument("-cu", "--cuda", default="0", help="Number of gpu device")
parser.add_argument("-valid","--validation_neg_num", default=0, type=int, help="The number of negative samples in validation set")

args = parser.parse_args()

