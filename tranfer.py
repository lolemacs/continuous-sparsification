import torch
from models import *
from load_datasets import *

parser = argeparse.ArgumentParser(discription='tranfer learning by using continuous sparsification')
parser.add_argument('--input-path-mask' type=str, default='./checkpoint.pt', help='file path to load mask')
parser.add_argument('--input-path-parameters' type=str, default='./checkpoint.pt', help='file path to load parameters (rewind)')
parser.add_argument('--num-class', type=int ,default=100, help='class number of new task(default 100)')
parser.add_argument('--seed', type=int, default=190457, help='random seed')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--which-gpu', type=int, default=0, help='gpu ID to use')
parser.add_argument('--distributed', type=bool, default=False, help='use distributed training or not')
parser.add_argument('--mask-initial-value', type=float, default=0., help='initial value for mask parameters, which is not important at loading.')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

model = ResNet50(num_class=1000, mask_initial_value=args.mask_initial_value)
print('architecture was set up. ')

if not args.cuda:
    print('using CPU, this will be slow')
elif args.distributed:
    if args.which_gpu is not None:
        torch.cuda.set_device(args.which_gpu)
        model.cuda(args.which_gpu)
        args.batch_size = int(args.batch_size/args.world_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.which_gpu])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
elif args.which_gpu is not None:
    torch.cuda.set_device(args.which_gpu)
    model = model.cuda(args.which_gpu)
else:
    model = model.cuda()

inputfile_mask = args.input_path_paramters
loc = 'cuda:{}'.format(arg.which_gpu)
checkpoint = torch.load(inputfile, map_location=loc)
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded patramters (rewind epoch {})".format(checkpoint['epoch']))

inputfile_mask = args.input_path_mask
loc = 'cuda:{}'.format(arg.which_gpu)
checkpoint = torch.load(inputfile, map_location=loc)
model_mask.load_state_dict(checkpoint['state_dict'])
print("=> loaded mask (best accuracy epoch {})".format(checkpoint['epoch']))

mask = model_mask.output_mask()

model.classifier = nn.Linear(2048, args.num_class)
