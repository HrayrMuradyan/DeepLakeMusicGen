import deeplake
import torch
import torch.nn.functional as F
from audiocraft.data.info_audio_dataset import AudioInfo
from audiocraft.modules.conditioners import WavCondition
from audiocraft.data.music_dataset import MusicInfo
from audiocraft.data.audio_dataset import SegmentInfo, AudioMeta
from audiocraft.data.audio_utils import convert_audio
import copy


class DeepLakeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_workers=1, batch_size=1, target_sr=32000, target_channels=1, segment_duration=30, pad=True, return_info=False, num_samples=10000, shuffle=True):
        self.segment_duration = segment_duration
        self.target_channels = target_channels
        self.target_sr = target_sr
        self.deep_lake_ds = deeplake.load(dataset_path)
        self.pad = pad
        self.return_info = return_info
        self.num_samples = num_samples
        self.target_frames = segment_duration * target_sr



    def collater(self, samples):
        """The collater function has to be provided to the dataloader
        if AudioDataset has return_info=True in order to properly collate
        the samples of a batch.
        """
        if self.segment_duration is None and len(samples) > 1:
            assert self.pad, "Must allow padding when batching examples of different durations."

        # In this case the audio reaching the collater is of variable length as segment_duration=None.
        to_pad = self.segment_duration is None and self.pad
        if to_pad:
            max_len = max([wav.shape[-1] for wav, _ in samples])

            def _pad_wav(wav):
                return F.pad(wav, (0, max_len - wav.shape[-1]))

        if self.return_info:
            if len(samples) > 0:
                assert len(samples[0]) == 2
                assert isinstance(samples[0][0], torch.Tensor)
                assert isinstance(samples[0][1], SegmentInfo)

            wavs = [wav for wav, _ in samples]
            segment_infos = [copy.deepcopy(info) for _, info in samples]

            if to_pad:
                # Each wav could be of a different duration as they are not segmented.
                for i in range(len(samples)):
                    # Determines the total length of the signal with padding, so we update here as we pad.
                    segment_infos[i].total_frames = max_len
                    wavs[i] = _pad_wav(wavs[i])

            wav = torch.stack(wavs)
            return wav, segment_infos
        else:
            assert isinstance(samples[0], torch.Tensor)
            if to_pad:
                samples = [_pad_wav(s) for s in samples]
            return torch.stack(samples)



    def __getitem__(self, index):
        index = index % len(self.deep_lake_ds)

        audio = self.deep_lake_ds.audio[index].data()['value'].reshape(-1)
        metadata = self.deep_lake_ds.metadata[index].data()['value']['metadata']
        info = self.deep_lake_ds.metadata[index].data()['value']['info']

        audio = convert_audio(torch.tensor(audio).unsqueeze(0).to(torch.float32), metadata['sample_rate'], self.target_sr, self.target_channels)
        n_frames = audio.shape[-1]

        if self.pad:
            audio = F.pad(audio, (0, self.target_frames - n_frames))
            n_frames = audio.shape[-1]

        audio_meta = AudioMeta.from_dict(metadata)
        segment_info = SegmentInfo(audio_meta, seek_time=0., n_frames=n_frames, total_frames=n_frames,
                                    sample_rate=self.target_sr, channels=self.target_channels)
        audio_info = AudioInfo(**segment_info.to_dict())


        info.update(audio_info.to_dict())
        music_info = MusicInfo.from_dict(info)

        music_info.self_wav = WavCondition(
            wav=audio[None], length=torch.tensor([audio_info.n_frames]),
            sample_rate=[audio_info.sample_rate], path=[audio_info.meta.path], seek_time=[audio_info.seek_time])


        return audio, music_info


    def __len__(self):
        return self.num_samples
