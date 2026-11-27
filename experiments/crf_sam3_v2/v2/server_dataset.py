"""Use expression-dependent mask/bbox rendering, as the inference pipeline does."""
def install():
    import crf_sam3_attn.train_data as td
    from crf_sam3_attn.dataset import CRFSAM3AttnDataset
    from crf_sam3_demo import contains_color_attribute
    class AlignedDataset(CRFSAM3AttnDataset):
        def _get_offline_listwise_item(self, idx):
            sample = self._samples[idx]
            meta = self._load_offline_meta(sample["video_id"], sample["exp_id"])
            if not meta:
                raise RuntimeError("Missing cache metadata; refusing random fallback")
            old = self.mask_style
            self.mask_style = "bbox" if contains_color_attribute(meta.get("expression", "")) else "mask"
            try:
                result = super()._get_offline_listwise_item(idx)
                if (result.get("video_id"), str(result.get("exp_id"))) != (sample["video_id"], str(sample["exp_id"])):
                    raise RuntimeError("Invalid cache caused dataset fallback; rebuild cache")
                return result
            finally:
                self.mask_style = old
    td.CRFSAM3AttnDataset = AlignedDataset
