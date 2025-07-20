import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from PIL import Image
from torchvision.transforms import ToTensor

def resize_image(img, max_width):
    img = Image.fromarray(img)
    new_height = int(max_width * img.height / img.width)
    img = img.resize((max_width, new_height), resample=Image.BILINEAR)
    tensor = ToTensor()(img)
    return tensor[:, :(new_height // 14) * 14]

if __name__ == "__main__":
    import numpy as np
    from hydra import initialize, compose
    from types import SimpleNamespace
    from data.datasets.co3d import Co3dDataset
    import rerun as rr

    img_size = 224
    with initialize(config_path="config"):
        cfg = compose(config_name="default")
        common_config = SimpleNamespace(**cfg["data"]["train"]["common_config"])
        common_config.augs = SimpleNamespace(**common_config.augs)
        common_config.augs.color_jitter = SimpleNamespace(**common_config.augs.color_jitter) \
            if common_config.augs.color_jitter is not None else None
        common_config.img_size = img_size
    
    dataset = Co3dDataset(
        common_config, 
        split="test", 
        CO3D_DIR="/storage/group/dataset_mirrors/01_incoming/Co3D",
        CO3D_ANNOTATION_DIR="/usr/prakt/s0018/3d_scene_editing/submodules/dust3r/data/co3d_anno"
    )
    batch = dataset.get_data(img_per_seq=4)
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    print(batch.keys())
    with torch.no_grad():
        # model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = VGGT(img_size=img_size, enable_camera=False, enable_depth=False, enable_point=True, enable_track=False).to(device)
        model.load_state_dict(torch.load("logs/exp001/ckpts/checkpoint_15.pt")["model"])
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            img_tensor = torch.stack([resize_image(img, img_size) for img in batch["images"]])[None].to(device)
            aggregated_tokens_list, ps_idx = model.aggregator(img_tensor)
        assert model.point_head is not None
        point_map, _ = model.point_head(aggregated_tokens_list, img_tensor, ps_idx)
        point_mask = torch.stack([torch.tensor(mask) for mask in batch["point_masks"]])[None]
        colors = img_tensor.numpy().transpose(0, 1, 3, 4, 2)[point_mask].reshape(-1, 3)
        rr.init("VGGT test")
        rr.connect_grpc()
        rr.log("world", rr.ViewCoordinates.RUF)
        rr.log("world/pts", rr.Points3D(point_map[point_mask].reshape(-1, 3), colors=colors.reshape(-1, 3)))
        for i, img in enumerate(batch["images"], 1):
            rr.log(f"world/img{i}", rr.Image(img))

