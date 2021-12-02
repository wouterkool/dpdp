import os
import argparse

import torch
from torch.autograd import Variable
from torch import nn

# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

import time
from datetime import timedelta

from config import get_config
from problems.vrp.vrp_reader import VRPReader
from problems.tsp.tsp_reader import TSPReader
from problems.tsptw.tsptw_reader import TSPTWReader

from models.gcn_model_vrp import ResidualGatedGCNModelVRP
from models.gcn_model import ResidualGatedGCNModel

from tqdm import tqdm
from utils.data_utils import save_dataset

from models.sparse_wrapper import wrap_sparse
from models.prep_wrapper import PrepWrapResidualGatedGCNModel

parser = argparse.ArgumentParser(description='Export heatmap')
parser.add_argument('-c','--config', type=str)
parser.add_argument('--problem', type=str, default='tsp')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--instances', type=str, required=True)
parser.add_argument('-o', '--output_filename', type=str)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--no_prepwrap', action='store_true', help='For backwards compatibility')
parser.add_argument('-f', action='store_true', help='Force overwrite existing results')
args = parser.parse_args()

assert os.path.isfile(args.checkpoint), "Make sure checkpoint file exists"

checkpoint_path = args.checkpoint
log_dir = os.path.split(checkpoint_path)[0]
config_path = args.config or os.path.join(log_dir, "config.json")

config = get_config(config_path)
print("Loaded {}:\n{}".format(config_path, config))

heatmap_filename = args.output_filename

if heatmap_filename is None:
    dataset_name = os.path.splitext(os.path.split(args.instances)[-1])[0]
    heatmap_dir = os.path.join("results", args.problem, dataset_name, "heatmaps")
    heatmap_filename = os.path.join(heatmap_dir, f"heatmaps_{config.expt_name}.pkl")
else:
    heatmap_dir = os.path.split(heatmap_filename)[0]

assert not os.path.isfile(heatmap_filename) or args.f, "Use -f to overwrite existing results"


if torch.cuda.is_available():
    print("CUDA available, using GPU")
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

do_prepwrap = not args.no_prepwrap

# Instantiate the network
model_class = ResidualGatedGCNModelVRP if args.problem == 'vrp' else ResidualGatedGCNModel
model = model_class(config, dtypeFloat, dtypeLong)
if args.problem in ('tsp', 'tsptw'):
    if 'sparse' in config and config.sparse is not None:
        model = wrap_sparse(model, config.sparse)

    if do_prepwrap:
        assert config.num_neighbors == -1, "PrepWrap only works for fully connected"
        model = PrepWrapResidualGatedGCNModel(model)
net = nn.DataParallel(model)
if torch.cuda.is_available():
    net.cuda()

if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_path)
else:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
# Load network state
if args.problem in ('tsp', 'tsptw'):
    try:
        net.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        # Backwards compatibility
        # Old checkpoints don't contain the PrepWrapModel, so load directly into the nested model
        # (but need to wrap DataParallel)
        nn.DataParallel(model.model).load_state_dict(checkpoint['model_state_dict'])
else:
    net.load_state_dict(checkpoint['model_state_dict'])

print("Loaded checkpoint with epoch", checkpoint['epoch'], 'val_loss', checkpoint['val_loss'])

# # Export heatmaps

# Set evaluation mode
net.eval()

batch_size = args.batch_size
num_nodes = config.num_nodes
num_neighbors = config.num_neighbors
beam_size = config.beam_size



# Heatmaps can make sense for clusters as well if we simply want to cache the predictions
# assert config.variant == "routes", "Heatmaps only make sense for routes"
instance_filepath = args.instances
if args.problem == 'vrp':
    reader = VRPReader(num_nodes, num_neighbors, batch_size, instance_filepath)
else:
    DataReader = DataReader = TSPTWReader if args.problem == 'tsptw' else TSPReader
    reader = DataReader(num_nodes, num_neighbors, batch_size, instance_filepath, do_prep=not do_prepwrap)

assert len(reader.filedata) % batch_size == 0, f"Number of instances {len(reader.filedata)} must be divisible by batch size {batch_size}"

dataset = iter(reader)

all_prob_preds = []
start = time.time()
for i, batch in enumerate(tqdm(dataset, total=reader.max_iter)):

    with torch.no_grad():
        if args.problem in ('tsp', 'tsptw') and do_prepwrap:
            # Convert batch to torch Variables
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
            x_nodes_timew = Variable(torch.FloatTensor(batch.nodes_timew).type(dtypeFloat), requires_grad=False) if args.problem == 'tsptw' else None

            # Forward pass
            with torch.no_grad():
                y_preds, loss, _ = net.forward(x_nodes_coord, x_nodes_timew)
        else:
            # Convert batch to torch Variables
            x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
            x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
            x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)

            # Forward pass
            with torch.no_grad():
                y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord)

        prob_preds = torch.log_softmax(y_preds, -1)[:, :, :, -1]

        all_prob_preds.append(prob_preds.cpu())
end = time.time()
duration = end - start
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

print("Took", timedelta(seconds=int(duration)), "s on ", device_count, "GPUs")
heatmaps = torch.cat(all_prob_preds, 0)
os.makedirs(heatmap_dir, exist_ok=True)
save_dataset((heatmaps.numpy(), {'duration': duration, 'device_count': device_count, 'args': args}), heatmap_filename)
print("Saved", len(heatmaps), "heatmaps to", heatmap_filename)
