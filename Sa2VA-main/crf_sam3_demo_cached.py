"""
Cached wrapper around CRF-SAM3 pipeline.

This module keeps the original pipeline implementation untouched and adds
lightweight per-video caches for:
1) category extraction (expression -> category/num_targets)
2) SAM3 category recall (video+category -> all_masks)

The goal is to avoid repeated SAM3 full-video propagation for expressions that
map to the same category inside one video, without increasing GPU memory usage.
"""

import os
from typing import Dict, Tuple, Any

from crf_sam3_demo import CRFSAM3Pipeline as _BaseCRFSAM3Pipeline


class CRFSAM3Pipeline(_BaseCRFSAM3Pipeline):
    """
    Backward-compatible pipeline with optional category-level recall cache.

    Notes:
    - Caches are reset when switching to a new video path.
    - Cache is in CPU memory (Python dict), not GPU memory.
    """

    def __init__(
        self,
        *args,
        enable_category_recall_cache: bool = True,
        cache_verbose: bool = False,
        **kwargs,
    ):
        self.enable_category_recall_cache = bool(enable_category_recall_cache)
        self.cache_verbose = bool(cache_verbose)

        self._cache_video_path = None
        self._category_cache: Dict[Tuple[str, str], Any] = {}
        self._recall_cache: Dict[Tuple[str, str], Any] = {}

        self._cache_stats = {
            "category_hits": 0,
            "category_misses": 0,
            "recall_hits": 0,
            "recall_misses": 0,
        }
        super().__init__(*args, **kwargs)

    @staticmethod
    def _norm_video_path(video_path: str) -> str:
        return os.path.abspath(str(video_path))

    @staticmethod
    def _norm_category(category: str) -> str:
        return str(category).strip().lower()

    @staticmethod
    def _norm_expression(expression: str) -> str:
        return str(expression).strip()

    def reset_video_cache(self):
        self._category_cache.clear()
        self._recall_cache.clear()

    def set_current_video(self, video_path: str):
        norm_path = self._norm_video_path(video_path)
        if self._cache_video_path != norm_path:
            self._cache_video_path = norm_path
            self.reset_video_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self._cache_stats.items()}

    def predict(self, video_path: str, referring_expression: str, return_details: bool = False):
        # Ensure per-video cache scope.
        self.set_current_video(video_path)

        if not self.enable_category_recall_cache:
            return super().predict(video_path, referring_expression, return_details=return_details)

        orig_extract_category = self.concept_extractor.extract_category
        orig_recall_all = self.sam3_recall.recall_all_instances

        def _cached_extract_category(expression, vpath=None):
            vpath = self._norm_video_path(vpath if vpath is not None else video_path)
            exp_key = self._norm_expression(expression)
            key = (vpath, exp_key)
            if key in self._category_cache:
                self._cache_stats["category_hits"] += 1
                if self.cache_verbose:
                    print(f"[CRF-CACHE] category HIT video={os.path.basename(vpath)} exp='{exp_key[:48]}'")
                return self._category_cache[key]

            self._cache_stats["category_misses"] += 1
            result = orig_extract_category(expression, vpath)
            self._category_cache[key] = result
            if self.cache_verbose:
                print(f"[CRF-CACHE] category MISS video={os.path.basename(vpath)} exp='{exp_key[:48]}'")
            return result

        def _cached_recall_all_instances(vpath, category):
            vpath = self._norm_video_path(vpath)
            cat_key = self._norm_category(category)
            key = (vpath, cat_key)
            if key in self._recall_cache:
                self._cache_stats["recall_hits"] += 1
                if self.cache_verbose:
                    print(f"[CRF-CACHE] recall HIT video={os.path.basename(vpath)} category='{cat_key}'")
                return self._recall_cache[key]

            self._cache_stats["recall_misses"] += 1
            result = orig_recall_all(vpath, category)
            self._recall_cache[key] = result
            if self.cache_verbose:
                print(f"[CRF-CACHE] recall MISS video={os.path.basename(vpath)} category='{cat_key}'")
            return result

        # Temporarily monkey-patch internals for this prediction call.
        self.concept_extractor.extract_category = _cached_extract_category
        self.sam3_recall.recall_all_instances = _cached_recall_all_instances
        try:
            return super().predict(video_path, referring_expression, return_details=return_details)
        finally:
            self.concept_extractor.extract_category = orig_extract_category
            self.sam3_recall.recall_all_instances = orig_recall_all
