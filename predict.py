import argparse
import pathlib
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchaudio
from score import Score
import torch

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", required=False, default=None, type=int)
    parser.add_argument("--mode", required=True, choices=["predict_file", "predict_dir"], type=str)
    parser.add_argument("--ckpt_path", required=False, default="epoch=3-step=7459.ckpt", type=pathlib.Path)
    parser.add_argument("--inp_dir", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--inp_path", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--out_path", required=True, type=pathlib.Path)
    parser.add_argument("--num_workers", required=False, default=0, type=int)
    return parser.parse_args()


class Dataset(Dataset):
    def __init__(self, dir_path: pathlib.Path):
        self.wavlist = list(dir_path.glob("*.wav"))
        _, self.sr = torchaudio.load(self.wavlist[0])

    def __len__(self):
        return len(self.wavlist)

    def __getitem__(self, idx):
        fname = self.wavlist[idx]
        wav, _ = torchaudio.load(fname)
        sample = {
            "wav": wav}
        return sample
    
    def collate_fn(self, batch):
        max_len = max([x["wav"].shape[1] for x in batch])
        out = []
        # Performing repeat padding
        for t in batch:
            wav = t["wav"]
            amount_to_pad = max_len - wav.shape[1]
            padding_tensor = wav.repeat(1,1+amount_to_pad//wav.size(1))
            out.append(torch.cat((wav,padding_tensor[:,:amount_to_pad]),dim=1))
        return torch.stack(out, dim=0)


def main():
    args = get_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "predict_file":
        assert args.inp_path is not None, "inp_path is required when mode is predict_file."
        assert args.inp_dir is None, "inp_dir should be None."
        assert args.inp_path.exists()
        assert args.inp_path.is_file()
        wav, sr = torchaudio.load(args.inp_path)
        scorer = Score(ckpt_path=args.ckpt_path, input_sample_rate=sr, device=device)
        score = scorer.score(wav.to(device))
        with open(args.out_path, "w") as fw:
            fw.write(str(score[0]))
    else:
        assert args.inp_dir is not None, "inp_dir is required when mode is predict_dir."
        assert args.bs is not None, "bs is required when mode is predict_dir."
        assert args.inp_path is None, "inp_path should be None."
        assert args.inp_dir.exists()
        assert args.inp_dir.is_dir()
        dataset = Dataset(dir_path=args.inp_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.bs,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            num_workers=args.num_workers)
        sr = dataset.sr
        scorer = Score(ckpt_path=args.ckpt_path, input_sample_rate=sr, device=device)
        with open(args.out_path, 'w'):
            pass
        for batch in tqdm.tqdm(loader):
            scores = scorer.score(batch.to(device))
            with open(args.out_path, 'a') as fw:
                for s in scores:
                    fw.write(str(s) + "\n")


if __name__ == '__main__':
    main()