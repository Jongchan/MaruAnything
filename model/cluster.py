import os
import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np

from sklearn.cluster import MiniBatchKMeans, KMeans


class Cluster:

    def __init__(
        self,
        minibatch_size: int = 1024,
        cluster_path: str = Path("./cluster.pkl"),
        n_clusters: int = 400,
    ):
        self.minibatch_size = minibatch_size
        self.cluster_path = cluster_path
        # self.cluster_method = MiniBatchKMeans(
        #     n_clusters=n_clusters,
        #     verbose=True,
        # )
        self.cluster_method = KMeans(n_clusters=n_clusters, verbose=True)

    def fit_cluster(
        self, 
        segment_paths: list,
        num_epochs: int = 1,
    ):
        if self.cluster_path.exists():
            self.cluster_method = pickle.loads(self.cluster_path.read_bytes())
            return
        
        # num_minibatches = (len(segment_paths) - 1) // self.minibatch_size + 1
        # for epoch in range(num_epochs):
        #     for i in tqdm(range(num_minibatches), desc=f"epoch {epoch}"):
        #         if i == num_minibatches - 1:
        #             # last batch
        #             minibatch_segment_paths = segment_paths[-self.minibatch_size:]
        #         else:    
        #             minibatch_segment_paths = segment_paths[i*self.minibatch_size:(i+1)*self.minibatch_size]

        #         minibatch = []
        #         for segment_path in minibatch_segment_paths:
        #             segment = pickle.loads(segment_path.read_bytes())["roi_feature"]
        #             minibatch.append(segment)
        #         self.cluster_method.partial_fit(minibatch)
        # self.save_cluster()

        all_features = []
        for segment_path in segment_paths:
            all_features.append(pickle.loads(segment_path.read_bytes())["roi_feature"])
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=16)
        all_features = self.pca.fit_transform(all_features)
        self.cluster_method.fit(all_features)
        self.save_cluster()

    def save_cluster(self):
        self.cluster_path.write_bytes(pickle.dumps(self.cluster_method))
        print(f"cluster saved to {self.cluster_path}")

    def inference_segment(self, feature):
        feature = self.pca.transform([feature])[0]
        inf = self.cluster_method.predict([feature])
        score = self.cluster_method.score([feature])
        return inf, score

if __name__ == "__main__":
    n_clusters = 500
    import glob
    segment_paths_all = [Path(path) for path in glob.glob("./segments/*/*.pkl")]
    import random
    random.shuffle(segment_paths_all)
    segment_paths = []
    for segment_path in segment_paths_all:
        segment = pickle.loads(segment_path.read_bytes())
        binary_mask = segment["binary_mask"].astype(np.float32)
        if binary_mask.sum() < binary_mask.size / 50:
            continue
        segment_paths.append(segment_path)
    print(f"all segments {len(segment_paths_all)}, valid segments {len(segment_paths)}")
    cluster = Cluster(
        cluster_path=Path(f"./cluster_{n_clusters}.pkl"),
        n_clusters=n_clusters,
    )
    cluster.fit_cluster(segment_paths, num_epochs=3)

    for segment_path in tqdm(segment_paths, desc="inference cluster"):
        segment = pickle.loads(segment_path.read_bytes())
        assigned_cluster, score = cluster.inference_segment(segment["roi_feature"])
        segment_cluster_info = {
            "image_id": segment["image_id"],
            "segment_idx": segment["segment_idx"],
            "cluster": assigned_cluster[0],
            "score": score,
        }
        segment_cluster_path = Path(f"./segment_cluster_{n_clusters}_PCA/{segment['image_id']}/{segment['segment_idx']}.pkl")
        segment_cluster_path.parent.mkdir(parents=True, exist_ok=True)
        segment_cluster_path.write_bytes(pickle.dumps(segment_cluster_info))
        