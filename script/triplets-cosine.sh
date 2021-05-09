#!/bin/sh -x

../build/triplets \
--trip ../dataset/triplets/wikipedia/wikipedia-triplets.txt \
--docs ../dataset/triplets/wikipedia/wikipedia-papers.txt \
--emb ../dataset/triplets/wikipedia/wikipedia-embeddings.txt \
--verbose false --num_clusters 289 --max_iter -1 --func cosine --r 16
