'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
from models import multimodalcnn
import torch
from opts import parse_opts
def generate_model(opt):
    model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = torch.nn.DataParallel(model, device_ids=None)

    return model, model.parameters()

def get_model():
    opt = parse_opts()
    model, parameters = generate_model(opt)
    state = torch.load('models/lt_1head_moddrop_2.pth')
    model.load_state_dict(state['state_dict'])
    return model