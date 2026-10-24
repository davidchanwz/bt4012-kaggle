"""Meaningful leakage and propagation checks for the message network."""
import copy
import json
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from message_model import MessageNet, torch_matrix, normalized, prepare


def run():
    torch.manual_seed(4012)
    a = sparse.csr_matrix((np.ones(4, np.float32), ([0, 1, 3, 4], [1, 2, 4, 5])), shape=(6, 6))
    x = torch.randn(6, 165)
    visible = torch.tensor([1., 0., 1., 1., 0., 1.])
    g = dict(x=x.clone(), visible=visible, metadata=torch.zeros(6, 5),
             incoming=torch_matrix(normalized(a.T)), outgoing=torch_matrix(normalized(a)),
             y=torch.tensor([0., 1., 0., 1., 0., 1.]), target=torch.randn(6, 165))
    model = MessageNet(True).eval()
    original = model(g).detach()
    changed = copy.deepcopy(g)
    changed['x'][visible==0] = 10000.
    torch.testing.assert_close(model(changed), original, rtol=0, atol=0)
    changed = copy.deepcopy(g)
    changed['y'] = 1-changed['y']
    changed['target'] *= -1000
    torch.testing.assert_close(model(changed), original, rtol=0, atol=0)
    changed = copy.deepcopy(g)
    changed['x'][3:] += 100
    torch.testing.assert_close(model(changed)[:3], original[:3], rtol=0, atol=0)
    changed = copy.deepcopy(g)
    changed['x'][0] += torch.randn(165)*3
    difference = float(abs(model(changed)[2].detach()-original[2]))
    assert difference > 1e-6, difference
    differentiable = copy.deepcopy(g)
    differentiable['x'].requires_grad_()
    model(differentiable)[2].backward()
    gradient = float(differentiable['x'].grad[0].abs().sum())
    assert gradient > 1e-8, gradient
    control = MessageNet(False).eval()
    torch.testing.assert_close(control(changed)[2], control(g)[2], rtol=0, atol=0)
    nodes = pd.DataFrame(dict(txId=np.arange(6),time_step=[1,1,1,2,2,2],
        visible=visible.numpy().astype(bool),supplied=np.ones(6,bool),
        label=[0.,np.nan,1.,0.,np.nan,1.],query=visible.numpy().astype(bool)))
    groups, _ = prepare(nodes,x.numpy(),a,1,2,include_targets=True)
    assert len(groups)==2 and 'target' in groups[0] and 'target' not in groups[1]
    np.testing.assert_array_equal(groups[0]['x'][1],np.zeros(165))
    from mixed_training import fit_mixed
    first,_=fit_mixed(groups,1,4012,True,epochs=1)
    changed_groups=copy.deepcopy(groups)
    changed_groups[1]['y']=1-changed_groups[1]['y']
    changed_groups[1]['x']*=1000
    second,_=fit_mixed(changed_groups,1,4012,True,epochs=1)
    for key,value in first.state_dict().items():
        torch.testing.assert_close(value,second.state_dict()[key],rtol=0,atol=0)
    checks = dict(hidden_vector_invariance=True, fraud_label_input_invariance=True,
                  reconstruction_target_input_invariance=True, disconnected_component_isolation=True,
                  two_hop_through_hidden_node=True, neighbour_encoder_gradient=True,
                  self_control_has_no_neighbour_dependence=True,
                  graph_preparation_respects_prefix_and_visibility=True,
                  mixed_training_ignores_future_labels_and_features=True,
                  two_hop_prediction_change=difference, remote_feature_gradient=gradient)
    print(json.dumps(checks, indent=2))
    return checks


if __name__=='__main__':
    run()
