from torch.utils.data import DataLoader
from data_loaders.tensors import t2m_collate, t2m_prefix_collate

def get_dataset_class(name):
    if name == "motion_stat_300":
        from .motion_stat_300_dataset import MotionStat300
        return MotionStat300
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    # 仅支持 motion_stat_300，沿用 t2m 风格的 collate
    if name != "motion_stat_300":
        raise ValueError(f'Unsupported dataset name [{name}] for collate_fn (only motion_stat_300 is kept).')
    if pred_len > 0:
        return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
    return lambda x: t2m_collate(x, batch_size)


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0,
                device=None, autoregressive=False, cache_path=None, data_dir: str = ""):
    DATA = get_dataset_class(name)
    if name == "motion_stat_300":
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len,
                       device=device, autoregressive=autoregressive, cache_path=cache_path, data_dir=data_dir)
    else:
        raise ValueError(f'Unsupported dataset name [{name}] (only motion_stat_300 is kept).')
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0,
                       device=None, autoregressive=False, cache_path=None, data_dir: str = ""):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len,
                          device=device, autoregressive=autoregressive, cache_path=cache_path, data_dir=data_dir)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader