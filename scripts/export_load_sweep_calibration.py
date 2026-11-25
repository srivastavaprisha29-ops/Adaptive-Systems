"""Export calibration scores for independent auditing after the sweep completes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from adaptive_systems import confidence_exit,experiment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--head',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    lock=json.loads((args.run/'lock.json').read_text())
    cache=torch.load(args.cache,weights_only=True,map_location='cpu')
    if cache['signature']['preparation']!=lock['preparation'] or experiment.digest(args.head)!=lock['head_sha256']:
        raise ValueError('Calibration inputs mismatch')
    head=torch.load(args.head,weights_only=True,map_location='cpu')
    features,teacher,labels=cache['values']
    logits=features@head['weight']+head['bias']
    margin=confidence_exit.margin(logits)
    output=dict(preparation=lock['preparation'],head_sha256=lock['head_sha256'],cache_sha256=experiment.digest(args.cache),
                records=[dict(image_id=image,margin=float(margin[i]),early_prediction=int(logits[i].argmax()),
                              full_prediction=int(teacher[i].argmax()),label=int(labels[i]))
                         for i,image in enumerate(cache['signature']['indices'])])
    target=args.run/'calibration-scores.json'
    with target.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print('Exported',len(output['records']),'calibration records')


if __name__=='__main__':main()
