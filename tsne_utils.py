'''
Util file for TSNE
'''

import pandas as pd
from MulticoreTSNE import MulticoreTSNE as TSNE

def run_tsne(latent_codes, category_ids, output_path="./tsne.csv"):
    # take as input latent codes and category labels
    # runs tsne
    # save as a csv (from pandas dataframe)
    # compute embeddings
    print("starting tsne.....")
    embeddings = TSNE(n_jobs=4, random_state=7).fit_transform(latent_codes)
    print("done with tsne")
    # inspired by: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-\
    #    and-t-sne-in-python-8ef87e7915b
    # uses multicore tsne = only 2 dims in projection
    emb_df = pd.DataFrame(
        data={"emb1": embeddings[:, 0], "emb2": embeddings[:, 1], "id": category_ids}
    )
    emb_df.to_csv(output_path)
    return embeddings