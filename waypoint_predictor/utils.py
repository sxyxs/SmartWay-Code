
import torch
import numpy as np
import sys
import glob
import json

def neighborhoods(mu, x_range, y_range, sigma, circular_x=True, gaussian=False):
    """ Generate masks centered at mu of the given x and y range with the
        origin in the centre of the output
    Inputs:
        mu: tensor (N, 2)
    Outputs:
        tensor (N, y_range, s_range)
    """
    x_mu = mu[:,0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:,1].unsqueeze(1).unsqueeze(1)

    # Generate bivariate Gaussians centered at position mu
    x = torch.arange(start=0,end=x_range, device=mu.device, dtype=mu.dtype).unsqueeze(0).unsqueeze(0)
    y = torch.arange(start=0,end=y_range, device=mu.device, dtype=mu.dtype).unsqueeze(1).unsqueeze(0)

    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(-0.5 * ((x_diff/sigma[0])**2 + (y_diff/sigma[1])**2 ))
    else:
        output = torch.logical_and(
            torch.abs(x_diff) <= sigma[0], torch.abs(y_diff) <= sigma[1]
        ).type(mu.dtype)

    return output


def nms(pred, max_predictions=10, sigma=(1.0,1.0), gaussian=False):
    ''' Input (batch_size, 1, height, width) '''

    shape = pred.shape

    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0],-1))  # (BATCH_SIZE, 24*48)
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0],-1))  # (BATCH_SIZE, 24*48)

    for i in range(max_predictions):
        # Find and save max over the entire map
        flat_supp_pred = supp_pred.reshape((shape[0],-1))
        val, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0,shape[0])
        flat_output[indices,ix] = flat_pred[indices,ix]

        # Suppression
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x,y], dim=1).float()

        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)

        supp_pred *= (1-g.unsqueeze(1))

    output[output < 0] = 0
    return output




def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_checkpoint_bak(epoch, net, net_optimizer, path):
    ''' Snapshot models '''
    states = {}
    def create_state(name, model, optimizer):
        states[name] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        create_state(*param)
    torch.save(states, path)
def save_checkpoint_womap(epoch, net, net_optimizer, path):
    ''' Snapshot models '''
    states = {}
    def create_state(name, model, optimizer):
        states[name] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        create_state(*param)
    torch.save(states, path)
def save_checkpoint(epoch, net, map_encoder, net_optimizer, path):
    ''' Snapshot models '''
    states = {}

    def create_state(name, model, optimizer=None):
        states[name] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }
        if optimizer:
            states[name]['optimizer'] = optimizer.state_dict()

    create_state("predictor", net, net_optimizer)
    create_state("map_encoder", map_encoder) 

    torch.save(states, path)

def load_checkpoint_bak(net, net_optimizer, path):
    ''' Loads parameters (but not training state) '''
    states = torch.load(path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print("NOTICE: DIFFERENT KEYS FOUND")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        recover_state(*param)
    return states['predictor']['epoch'], all_tuple[0][1], all_tuple[0][2]

def load_checkpoint_orig(net, net_optimizer, path):
    ''' Loads parameters (but not training state) '''
    print(path)
    states = torch.load(path)
    def recover_state(name, model, optimizer):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print("NOTICE: DIFFERENT KEYS FOUND")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        optimizer.load_state_dict(states[name]['optimizer'])
    all_tuple = [("predictor", net, net_optimizer)]
    for param in all_tuple:
        recover_state(*param)
    return states['predictor']['epoch'], all_tuple[0][1], all_tuple[0][2]

def load_checkpoint(net, map_encoder, net_optimizer, path):
    ''' Loads parameters (but not training state) '''
    states = torch.load(path)

    def recover_state(name, model, optimizer=None):
        state = model.state_dict()
        model_keys = set(state.keys())
        load_keys = set(states[name]['state_dict'].keys())
        if model_keys != load_keys:
            print(f"NOTICE: DIFFERENT KEYS FOUND IN {name}")
        state.update(states[name]['state_dict'])
        model.load_state_dict(state)
        if optimizer:
            optimizer.load_state_dict(states[name]['optimizer'])

    
    recover_state("predictor", net, net_optimizer)
    recover_state("map_encoder", map_encoder)

    return states['predictor']['epoch'], net, map_encoder, net_optimizer


def get_attention_mask(num_imgs=24, neighbor=2):
    assert neighbor <= 5

    mask = np.zeros((num_imgs,num_imgs))
    t = np.zeros(num_imgs)
    t[:neighbor+1] = np.ones(neighbor+1)
    if neighbor != 0:
        t[-neighbor:] = np.ones(neighbor)
    for ri in range(num_imgs):
        mask[ri] = t
        t = np.roll(t, 1)

    return torch.from_numpy(mask).reshape(1,1,num_imgs,num_imgs).long()


def load_gt_navigability(path):
    ''' waypoint ground-truths '''
    all_scans_nav_map = {}
    gt_dir = glob.glob('%s*'%(path))
    for gt_dir_i in gt_dir:
        with open(gt_dir_i, 'r') as f:
            nav_map = json.load(f)
        for scan_id, values in nav_map.items():
            all_scans_nav_map[scan_id] = values
    return all_scans_nav_map
