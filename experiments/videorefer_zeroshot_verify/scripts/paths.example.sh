# Copy to paths.sh and edit. Example values are the original server paths from the CRF-SAM3 configs.
export CRF_ROOT=/mnt/csip-500/share/kongjiawei/Sa2VA-main
export SAM3_CKPT=/mnt/csip-500/share/kongjiawei/sam3-main/sam3-main/checkpoints/sam3/sam3.pt
export REVOS=/mnt/csip-500/share/kongjiawei/Sa2VA/data/video_datas/revos
export HIST=$CRF_ROOT/crf_sam3_revos_valid_revos_query_ranker_20260806
export VR_REPO=/path/to/VideoRefer/VideoRefer          # dir that contains the `videorefer` package
export VR_CKPT=/path/to/VideoRefer-7B                  # or DAMO-NLP-SG/VideoRefer-7B
export SIGLIP=/path/to/siglip-so400m-patch14-384       # name must contain "siglip"
export WORK=/path/to/vr_zeroshot_work
export CRF_PY=/mnt/csip-500/share/kongjiawei/Sa2VA/.venv/bin/python   # CRF-SAM3 env python
export VR_PY=python                                    # python of the `videorefer` conda env
