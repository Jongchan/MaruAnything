import os
import pickle
from pathlib import Path
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans


class Cluster:

    def __init__(
        self,
        minibatch_size: int = 1024,
        cluster_path: str = Path("./cluster.pkl"),
        n_clusters: int = 400,
    ):
        self.minibatch_size = minibatch_size
        self.cluster_path = cluster_path
        self.cluster_method = MiniBatchKMeans(
            n_clusters=n_clusters,
            verbose=True,
        )

    def fit_cluster(
        self, 
        segment_paths: list,
        num_epochs: int = 1,
    ):
        if self.cluster_path.exists():
            self.cluster_method = pickle.loads(self.cluster_path.read_bytes())
            return
        
        num_minibatches = (len(segment_paths) - 1) // self.minibatch_size + 1
        for epoch in range(num_epochs):
            for i in tqdm(range(num_minibatches), desc=f"epoch {epoch}"):
                if i == num_minibatches - 1:
                    # last batch
                    minibatch_segment_paths = segment_paths[-self.minibatch_size:]
                else:    
                    minibatch_segment_paths = segment_paths[i*self.minibatch_size:(i+1)*self.minibatch_size]

                minibatch = []
                for segment_path in minibatch_segment_paths:
                    segment = pickle.loads(segment_path.read_bytes())["roi_feature"]
                    minibatch.append(segment)
                self.cluster_method.partial_fit(minibatch)
        self.save_cluster()

    def save_cluster(self):
        self.cluster_path.write_bytes(pickle.dumps(self.cluster_method))
        print(f"cluster saved to {self.cluster_path}")

    def inference_segment(self, feature):
        
        inf = self.cluster_method.predict([feature])
        return inf

if __name__ == "__main__":
    n_clusters = 400
    import glob
    segment_paths = [Path(path) for path in glob.glob("./segments/*/*.pkl")]
    import random
    random.shuffle(segment_paths)
    cluster = Cluster(
        cluster_path=Path(f"./cluster_{n_clusters}.pkl"),
        n_clusters=n_clusters,
    )
    cluster.fit_cluster(segment_paths, num_epochs=10)

    for segment_path in tqdm(segment_paths, desc="inference cluster"):
        segment = pickle.loads(segment_path.read_bytes())
        assigned_cluster = cluster.inference_segment(segment["roi_feature"])
        segment_cluster_info = {
            "image_id": segment["image_id"],
            "segment_idx": segment["segment_idx"],
            "cluster": assigned_cluster,
        }
        segment_cluster_path = Path(f"./segment_cluster/{segment['image_id']}/{segment['segment_idx']}.pkl")
        segment_cluster_path.parent.mkdir(parents=True, exist_ok=True)
        segment_cluster_path.write_bytes(pickle.dumps(segment_cluster_info))
        