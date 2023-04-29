from tqdm import tqdm

from dataset import Dataset
from segmenter import Segmenter

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    dataset = Dataset(args.config_path)
    segmenter = Segmenter()
    # clusterer = Clusterer()

    segment_save_dir = "./segments"

    pbar = tqdm(dataset, desc="Segmenting")
    for image_id, image_path in pbar:
        pbar.set_description(f"Processing {image_id}")
        segments = segmenter.get_segments(image_id, image_path)
        for segment in segments:
            segment.save_to(segment_save_dir)