from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report)


class AllMulticlassClfMetrics:
    """Wrapper for sklearn balanced accuracy for PyTorch."""

    def __call__(self, targets, preds, prefix='train'):
        ret = {}
        d = classification_report(
            targets, preds, output_dict=True, zero_division=0)
        d.pop('accuracy')
        for k1, v1 in d.items():
            for k2, v2 in v1.items():
                if k1.isnumeric():
                    k = f'{k2}/{k1}'
                else:
                    if 'support' in k2:
                        continue
                    k = k1.replace(' ', '_') + f'/{k2}'
                ret[f'{prefix}/{k}'] = float(v2)

        for m, tag in zip([accuracy_score,
                           balanced_accuracy_score,
                           ],
                          ['accuracy',
                           'balanced_accuracy'
                           ]):
            ret[prefix+'/'+tag] = m(targets, preds)
        return ret


def retrieve(x):
    return x.detach().cpu().numpy().flatten()
