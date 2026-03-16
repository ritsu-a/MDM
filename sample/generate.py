# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
import shutil
from data_loaders.tensors import collate
import subprocess


def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    # 当前工程仅保留 motion_stat_300
    if args.dataset != 'motion_stat_300':
        raise ValueError(f"当前代码仅支持 dataset=motion_stat_300，收到: {args.dataset}")
    n_joints = 21
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    # motion_stat_300: 固定 300 帧、60fps
    max_frames = 300
    fps = 60
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    if args.context_len > 0:
        is_using_data = True  # For prefix completion, we need to sample a prefix
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        elif args.dynamic_text_path != '':
            out_path += '_' + os.path.basename(args.dynamic_text_path).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    texts = None
    if args.text_prompt != '':
        texts = [args.text_prompt] * args.num_samples
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.dynamic_text_path != '':
        assert os.path.exists(args.dynamic_text_path)
        assert args.autoregressive, "Dynamic text sampling is only supported with autoregressive sampling."
        with open(args.dynamic_text_path, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        n_frames = len(texts) * args.pred_len  # each text prompt is for a single prediction
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    sample_fn = diffusion.p_sample_loop
    if args.autoregressive:
        sample_cls = AutoRegressiveSampler(args, sample_fn, n_frames)
        sample_fn = sample_cls.sample

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

    if is_using_data:
        iterator = iter(data)
        input_motion, model_kwargs = next(iterator)
        input_motion = input_motion.to(dist_util.dev())
        if texts is not None:
            model_kwargs['y']['text'] = texts
    else:
        # 仅支持文本条件（t2m），不再支持 a2m
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)

    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
    init_image = None    
    
    all_motions = []
    all_lengths = []
    all_text = []

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    
    if 'text' in model_kwargs['y'].keys():
        # encoding once instead of each iteration saves lots of time
        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
    
    if args.dynamic_text_path != '':
        # Rearange the text to match the autoregressive sampling - each prompt fits to a single prediction
        # Which is 2 seconds of motion by default
        model_kwargs['y']['text'] = [model_kwargs['y']['text']] * args.num_samples
        if args.text_encoder_type == 'bert':
            model_kwargs['y']['text_embed'] = (model_kwargs['y']['text_embed'][0].unsqueeze(0).repeat(args.num_samples, 1, 1, 1), 
                                               model_kwargs['y']['text_embed'][1].unsqueeze(0).repeat(args.num_samples, 1, 1))
        else:
            raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
    
    all_qpos = []
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        sample = sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # motion_stat_300: 直接在原始特征空间输出 qpos（不做 SMPL / rot2xyz / recover_from_ric）
        qpos = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float().numpy()  # (B,T,60) or (B,T,1,60)
        if qpos.ndim == 4 and qpos.shape[2] == 1:
            qpos = np.squeeze(qpos, axis=2)
        if qpos.ndim != 3:
            raise ValueError(f"Unexpected qpos shape for motion_stat_300: {qpos.shape}")
        all_qpos.append(qpos)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(all_qpos[-1])
        _len = model_kwargs['y']['lengths'].cpu().numpy()
        if 'prefix' in model_kwargs['y'].keys():
            _len[:] = sample.shape[-1]
        all_lengths.append(_len)

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    if args.dynamic_text_path != '':
        text_file_content = '\n'.join(['#'.join(s) for s in all_text])
    else:
        text_file_content = '\n'.join(all_text)
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write(text_file_content)
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    # motion_stat_300: 保存 qpos 结果与可选的视频渲染
    if args.save_npz:
        npz_dir = os.path.join(out_path, "npz")
        os.makedirs(npz_dir, exist_ok=True)
        for i in range(total_num_samples):
            np.savez(os.path.join(npz_dir, f"sample{i:04d}.npz"), qpos=all_motions[i])
    if args.render_video:
        if not args.save_npz:
            raise ValueError("--render_video requires --save_npz for motion_stat_300.")
        if not os.path.isfile(args.render_script_path):
            raise FileNotFoundError(f"Render script not found: {args.render_script_path}")
        videos_dir = os.path.join(out_path, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        for i in range(total_num_samples):
            npz_path = os.path.join(out_path, "npz", f"sample{i:04d}.npz")
            mp4_path = os.path.join(videos_dir, f"sample{i:04d}.mp4")
            subprocess.run(
                [
                    "python",
                    args.render_script_path,
                    "--npz_path",
                    npz_path,
                    "--video_path",
                    mp4_path,
                    "--motion_fps",
                    str(args.motion_fps),
                ],
                check=True,
            )
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')
    return out_path


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='val',
                              hml_mode='train' if args.pred_len > 0 else 'text_only',  # We need to sample a prefix from the dataset
                              fixed_len=args.pred_len + args.context_len, pred_len=args.pred_len, device=dist_util.dev(),
                              data_dir=args.data_dir if getattr(args, "data_dir", "") else "")
    data.fixed_length = n_frames
    return data


def is_substr_in_list(substr, list_of_strs):
    return np.char.find(list_of_strs, substr) != -1  # [substr in string for string in list_of_strs]

if __name__ == "__main__":
    main()
