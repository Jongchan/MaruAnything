import yaml

# TODO: we may add Image class to handle other metadata (e.g. annotations)

class Dataset:

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_dir = config["image_dir"]
        with open(config["image_ids_txt"], "r") as f:
            self.image_ids = f.readlines()
            self.image_ids = sorted([image_id.strip() for image_id in self.image_ids])
        print(f"Dataset initialized with {len(self.image_ids)} images")
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = f"{self.image_dir}/{image_id}.jpg"
        return image_id, image_path
    
    def __len__(self):
        return len(self.image_ids)
    def __iter__(self):
        self.current_index = 0
        return self
    def __next__(self):
        if self.current_index < len(self):
            x = self[self.current_index]
            self.current_index += 1
            return x
        raise StopIteration