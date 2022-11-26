import context

import torch

import src

if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.Linear(10, 3),
        torch.nn.Linear(3, 1),
    )
    print(list(model.parameters())[0])
    with open('test.pt', 'wb') as f:
        torch.save(model.state_dict(), f)

    ss_model = torch.nn.Sequential(
        src.model.SSLinear(5, 10),
        src.model.SSLinear(10, 3),
        src.model.SSLinear(3, 1),
    )
    print(list(ss_model.parameters())[0])

    with open('test.pt', 'rb') as f:
        ss_model.load_state_dict(torch.load(f), strict=False)
    print(list(ss_model.parameters())[0])

    x = torch.randn(1, 5)
    print(model(x))
    print(ss_model(x))

    torch.nn.Linear = src.model.SSLinear
    layer = torch.nn.Linear(3, 1)
    print(type(layer))

    with open(src.config.Paths.models / 'm35_e_d.pth', 'rb') as f:
        states_dict = torch.load(f, map_location=lambda storage, loc: storage)
    e_states = states_dict['encoder']
    d_states = states_dict['decoder']
    encoder = src.api.get_configured_encoder()
    encoder.load_state_dict(e_states, strict=False)
    paras = []
    for i in encoder.modules():
        if isinstance(i, src.model.SSLinear):
            i.reset_ss()
            paras.append(i.gamma)
            paras.append(i.beta)
    print(paras)
    pass
