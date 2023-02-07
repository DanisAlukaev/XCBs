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


class CollateBOW:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, batch):
        batch_dict = dict()
        indices = []
        bows = []
        max_len = -1
        for sample in batch:
            for key in sample.keys():
                if key not in batch_dict:
                    batch_dict[key] = list()
                batch_dict[key].append(sample[key])
            tokens = self.vocabulary.tokenizer(sample["report"])
            if len(tokens) > max_len:
                max_len = len(tokens)
            id_ = self.vocabulary.vocab.lookup_indices(tokens)
            indices.append(id_)

            # TODO: compute on the fly
            bow = [0] * 8800
            for i in id_:
                bow[i] += 1
            bows.append(torch.tensor(bow).float())
        batch_dict["bow"] = bows
        for key in batch_dict.keys():
            if not all(isinstance(x, torch.Tensor) for x in batch_dict[key]):
                continue
            batch_dict[key] = torch.stack(batch_dict[key])

        return batch_dict
