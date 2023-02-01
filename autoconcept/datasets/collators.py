
import torch


class CollateEmulator:

    def __init__(self):
        pass

    def __call__(self, batch):
        batch_dict = dict()
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])

        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict
