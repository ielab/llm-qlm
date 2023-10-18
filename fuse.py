import argparse
from ranx import fuse, Run

parser = argparse.ArgumentParser()
parser.add_argument('--run1', type=str, help='Path to the first run file.')
parser.add_argument('--run2', type=str, help='Path to the second run file.')
parser.add_argument('--weight1', type=float, help='Weight of the first run.')
parser.add_argument('--weight2', type=float, help='Weight of the second run.')
parser.add_argument('--retrieval_fusion', action='store_true', default=False)
parser.add_argument('--output', type=str, help='Path to the output file.')
args = parser.parse_args()


run1 = Run.from_file(args.run1, kind='trec').to_dict()
run2 = Run.from_file(args.run2, kind='trec').to_dict()

if args.retrieval_fusion:
    pass
else:
    # make two runs with same cutoff.
    # Note this is only for this project and for the sake of simplicity.
    # We assume the two runs are having the same candidates of documents for each query.
    for qid in run1.keys():
        new_dict = {}
        if len(run1[qid]) > len(run2[qid]):
            for doc in run2[qid]:
                new_dict[doc] = run1[qid][doc]
            run1[qid] = new_dict
        else:
            for doc in run1[qid]:
                new_dict[doc] = run2[qid][doc]
            run2[qid] = new_dict

run1 = Run.from_dict(run1)
run2 = Run.from_dict(run2)

combined_run = fuse(
        runs=[run1, run2],
        norm="min-max",  # Default normalization strategy
        params={'weights': [args.weight1, args.weight2]},
    )

combined_run.save(args.output, kind='trec')