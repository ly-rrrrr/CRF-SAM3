"""
MAPP-SAM3: Moment-Adaptive Phrase Prompting for SAM 3
Designed for RVOS (Referring Video Object Segmentation)

Core Innovation:
1. Q-Frame: Coarse temporal filtering with Gumbel-Max sampling
2. Best View Reasoning: MLLM finds optimal initialization frames
3. SAM3 Propagation: Bidirectional tracking with Memory Bank
"""

import os
import sys
import json
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add project paths
sys.path.append("/mnt/csip-200/share/kongjiawei/q-frame-master")
sys.path.append("/mnt/csip-200/share/kongjiawei/sam3-main")
sys.path.append("/mnt/csip-200/share/kongjiawei/Sa2VA-main")


class QFrameSelector:
    """Module 1: Coarse Temporal Filtering using Q-Frame"""

    def __init__(self, clip_model_path: str, device: str = "cuda", apply_rebalancing: bool = False):
        """
        Initialize Q-Frame selector with Long-CLIP

        Args:
            clip_model_path: Path to Long-CLIP checkpoint
            device: Device to run on (default: cuda)
            apply_rebalancing: Whether to apply adaptive rebalancing when sampling is concentrated
        """
        self.device = device
        self.apply_rebalancing = apply_rebalancing

        # Load Long-CLIP model
        # Import from the correct path
        longclip_path = "/mnt/csip-200/share/kongjiawei/q-frame-master/Long-CLIP"
        if longclip_path not in sys.path:
            sys.path.insert(0, longclip_path)

        from model import longclip
        self.longclip = longclip  # Store module for later use

        print(f"[Q-Frame] Loading Long-CLIP from {clip_model_path}...")
        print(f"[Q-Frame] Adaptive rebalancing: {'enabled' if self.apply_rebalancing else 'disabled'}")
        self.clip_model, self.clip_processor = longclip.load(
            clip_model_path, device=device
        )
        self.clip_model.eval()

    def select_keyframes(
        self,
        video_frames: List[np.ndarray],
        referring_expression: str,
        num_candidates: int = 8,
        tau: float = 0.8
    ) -> List[int]:
        """
        Select Top-K keyframes using Gumbel-Max sampling

        Args:
            video_frames: List of video frames (H, W, 3)
            referring_expression: Complex referring expression
            num_candidates: Number of frames to select
            tau: Temperature for Gumbel sampling

        Returns:
            List of selected frame indices (sorted)
        """
        # Step 1: Uniform sampling to reduce computation
        total_frames = len(video_frames)
        if total_frames > 128:
            sample_indices = np.linspace(0, total_frames - 1, 128, dtype=int)
            sampled_frames = [video_frames[i] for i in sample_indices]
        else:
            sample_indices = np.arange(total_frames)
            sampled_frames = video_frames

        # Step 2: Compute image-text similarity with Long-CLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Encode text
            text_tokens = self.longclip.tokenize([referring_expression]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)

            # Encode images
            image_pil = [Image.fromarray(frame) for frame in sampled_frames]
            images = torch.stack([
                self.clip_processor(img) for img in image_pil
            ]).to(self.device)
            image_features = self.clip_model.encode_image(images)

            # Compute similarity
            logits = text_features @ image_features.T  # (1, N)
            probs = (logits / tau).softmax(dim=1)[0]

        # Step 3: Gumbel-Max sampling for diversity
        gumbel_noise = -torch.log(-torch.log(
            torch.rand(len(probs), device=self.device) + 1e-10
        ) + 1e-10)
        gumbel_probs = torch.log(probs) + gumbel_noise

        # Select top-K
        _, top_indices = torch.topk(gumbel_probs, num_candidates)
        top_indices = top_indices.cpu().numpy()

        # Map back to original frame indices
        selected_frame_indices = sample_indices[top_indices]
        selected_frame_indices = np.sort(selected_frame_indices)

        print(f"[Q-Frame] Initial selection: {num_candidates} keyframes from {total_frames} frames")
        print(f"[Q-Frame] Initial indices: {selected_frame_indices.tolist()}")

        # Adaptive adjustment: avoid dense sampling, ensure video-level coverage
        # Only apply if self.apply_rebalancing is True
        if not self.apply_rebalancing:
            print(f"[Q-Frame] Adaptive rebalancing disabled, returning raw Q-Frame selection")
            return selected_frame_indices.tolist()

        selected_list = selected_frame_indices.tolist()
        q_frame_span = selected_list[-1] - selected_list[0]
        video_span = total_frames - 1

        print(f"[Q-Frame] Q-Frame span: {q_frame_span} frames (from {selected_list[0]} to {selected_list[-1]})")
        print(f"[Q-Frame] Video span: {video_span} frames")

        # If Q-Frame sequence is too concentrated (< 50% of video length), rebalance
        if q_frame_span < video_span * 0.5:
            print(f"[Q-Frame] ⚠ Detected concentrated sampling (span={q_frame_span}/{video_span}={q_frame_span/video_span:.1%})")
            print(f"[Q-Frame] Applying adaptive rebalancing...")

            # Target: half from context (before+after), half from event region
            num_context_target = num_candidates // 2
            num_event_target = num_candidates - num_context_target

            # Step 1: Sample context frames from unexplored regions (before and after)
            before_region = list(range(0, selected_list[0]))
            after_region = list(range(selected_list[-1] + 1, total_frames))

            num_before = num_context_target // 2
            num_after = num_context_target - num_before

            context_indices = []

            # Sample from before region
            if len(before_region) > 0 and num_before > 0:
                step = max(1, len(before_region) // num_before)
                before_samples = before_region[::step][:num_before]
                context_indices.extend(before_samples)
                print(f"[Q-Frame] Sampled {len(before_samples)} frames from before region [0, {selected_list[0]})")
            else:
                print(f"[Q-Frame] No frames available in before region")

            # Sample from after region
            if len(after_region) > 0 and num_after > 0:
                step = max(1, len(after_region) // num_after)
                after_samples = after_region[::step][:num_after]
                context_indices.extend(after_samples)
                print(f"[Q-Frame] Sampled {len(after_samples)} frames from after region ({selected_list[-1]}, {total_frames})")
            else:
                print(f"[Q-Frame] No frames available in after region")

            # Step 2: Check if context sampling is sufficient
            num_context_actual = len(context_indices)
            num_deficit = num_context_target - num_context_actual

            if num_deficit > 0:
                print(f"[Q-Frame] Context sampling deficit: {num_deficit} frames, will compensate from event region")
                num_event_target += num_deficit

            # Step 3: Sample from event region (Q-Frame results)
            # Use stride sampling to ensure diversity
            stride = max(1, len(selected_list) // num_event_target)
            event_indices = selected_list[::stride][:num_event_target]
            print(f"[Q-Frame] Sampled {len(event_indices)} frames from event region (stride={stride})")

            # Combine and sort
            final_indices = sorted(event_indices + context_indices)
            print(f"[Q-Frame] Rebalanced indices: {final_indices} (total: {len(final_indices)})")
            print(f"[Q-Frame] Event frames ({len(event_indices)}): {event_indices}")
            print(f"[Q-Frame] Context frames ({len(context_indices)}): {context_indices}")

            return final_indices
        else:
            print(f"[Q-Frame] Sampling is well-distributed, no adjustment needed")
            return selected_list


class BestViewReasoner:
    """Module 2: Best View Reasoning with MLLM + SAM3 Presence Head"""

    def __init__(
        self,
        mllm_path: str,
        sam3_checkpoint: str,
        device: str = "cuda"
    ):
        """
        Initialize MLLM and SAM3 Presence Head

        Args:
            mllm_path: Path to MLLM model (e.g., InternVL2.5)
            sam3_checkpoint: Path to SAM3 checkpoint
            device: Device to run on
        """
        self.device = device

        # Load MLLM (using transformers) - following official Qwen2.5-VL example
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"[BestView] Loading MLLM from {mllm_path}...")

        # Use bfloat16 and prioritize GPU 0 and 1 by using sequential strategy
        # This will fill GPU 0 first, then GPU 1, then GPU 2, then GPU 3
        # Long-CLIP on GPU 0, SAM3 on GPU 1
        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mllm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Fill GPUs sequentially: 0 -> 1 -> 2 -> 3
            # max_memory={
            #     0: "28GiB",  # GPU 0 (shared with Long-CLIP) - fill this first
            #     1: "28GiB",  # GPU 1 (shared with SAM3) - fill this second
            #     2: "6GiB",   # GPU 2 (has other processes) - use only if needed
            #     3: "4GiB",   # GPU 3 (has other processes) - use only if needed
            # },
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            mllm_path,
            trust_remote_code=True
        )

        # Note: SAM3 will be loaded later when needed to save memory
        self.sam3_checkpoint = sam3_checkpoint
        self.sam3_model = None
        print("[BestView] SAM3 will be loaded on-demand to save memory")

    def filter_by_presence(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        referring_expression: str,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Filter frames using SAM3 Presence Token

        Args:
            frames: List of frames
            frame_indices: Candidate frame indices
            referring_expression: Text description
            threshold: Presence confidence threshold

        Returns:
            Filtered frame indices with high presence scores
        """
        print(f"[BestView] Filtering {len(frame_indices)} frames with Presence Head...")

        # Load SAM3 on-demand if not yet loaded
        if self.sam3_model is None:
            from sam3.model_builder import build_sam3_image_model
            print(f"[BestView] Loading SAM3 from {self.sam3_checkpoint}...")
            self.sam3_model = build_sam3_image_model(
                checkpoint_path=self.sam3_checkpoint,
                device=self.device,
                eval_mode=True,
                enable_segmentation=True
            )

        # For now, skip Presence Head filtering as it requires specific SAM3 API
        # Return all frames (can be implemented later)
        print(f"[BestView] Presence Head filtering skipped (using all candidates)")
        return frame_indices

    def find_best_frames(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        referring_expression: str
    ) -> Dict[int, Tuple[int, str]]:
        """
        Find best initialization frame for each object using MLLM

        The MLLM will:
        1. Analyze the referring expression AND video frames
        2. Determine how many objects are mentioned
        3. For each object, find the best initialization frame
        4. Generate simplified phrases for each object

        Args:
            frames: Full video frames
            frame_indices: Filtered candidate frame indices
            referring_expression: Complex referring expression

        Returns:
            Dict mapping object_id to (best_frame_idx, simple_phrase)
        """
        print(f"[BestView] MLLM reasoning to determine number of objects and best frames...")

        # Prepare multi-frame prompt for MLLM
        candidate_frames = [frames[i] for i in frame_indices]

        # Construct prompt - MLLM determines number of objects
        # Construct prompt - MLLM determines number of objects
        prompt = f"""Analyze these video frames to segment objects mentioned in the referring expression.
Referring Expression: "{referring_expression}"
Your Task:
1. **Count objects**: Look at ALL these frames and count how many distinct objects match the referring expression
    If referring expression like:
       - "Cows moving forward" → Count individual cows you see (could be 2, 3, 4, or more)
       - "The white cow" → 1 object
   - Don't guess from language - COUNT from visual observation

2. **Choose best frame for EACH object**: Select the frame where each object is:
    most clearly positioned relative to the scene, such that natural phrases like “the 'category name' on the left / in the center / on the right” best apply.
    This frame should show the object:
        - fully visible
        - minimally occluded
        - spatially separated from similar objects
        - with a stable and unambiguous left–center–right position

3. **Generate simple phrase for EACH object**: 
   - Good examples: "the cow on the left", "the cow in the center", "the cow on the right"
   - MUST use "the" before the category name to indicate a specific object
   - Use prepositions: "on the left", "in the center", "on the right"
   - AVOID: colors, actions ("walking", "running"), phrases without "the"
   - Keep it natural and concise: 4-7 words maximum

Output Format (JSON):
{{
  "num_objects": <int>,
  "objects": [
    {{
      "object_id": 0,
      "best_frame_index": <frame_number>,
      "simple_phrase": "the + <category name> + <position>",
    }},
    ...
  ]
}}
Example 1 - Single object:
For "the person walking", output:
{{
  "num_objects": 1,
  "objects": [
    {{
      "object_id": 0,
      "best_frame_index": 5,
      "simple_phrase": "the person in the center",
    }}
  ]
}}
Example 2 - Multiple objects (CRITICAL - this is what you need to do well!):
For "cows in the field", after visually counting 3 cows in frames:
{{
  "num_objects": 3,
  "objects": [
    {{
      "object_id": 0,
      "best_frame_index": 2,
      "simple_phrase": "the cow on the left",
    }},
    {{
      "object_id": 1,
      "best_frame_index": 5,
      "simple_phrase": "the cow in the center",
    }},
    {{
      "object_id": 2,
      "best_frame_index": 8,
      "simple_phrase": "the cow on the right",
    }}
  ]
}}
Now analyze the following {len(candidate_frames)} candidate frames (frame indices: {frame_indices}):
"""

        # Prepare images for MLLM (create a grid visualization)
        # For actual implementation, pass frames to MLLM's vision encoder

        # Placeholder: Use MLLM to generate structured output
        with torch.no_grad():
            # This is a simplified call - actual implementation needs proper
            # multi-image handling based on the MLLM's API
            response = self._call_mllm_with_images(
                prompt, candidate_frames, frame_indices
            )

        # Parse JSON response
        import re

        # Log full MLLM response for debugging
        print(f"\n[BestView] ===== MLLM Raw Response =====")
        print(response)
        print(f"[BestView] ===== End of Response =====\n")

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group()
                # Clean up trailing commas that break JSON parsing
                # Remove commas before closing braces or brackets
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                result_full = json.loads(json_str)

                # Check if new three-stage format or old format
                if "stage3_output" in result_full:
                    # New format with three stages
                    print(f"\n[BestView] ===== Stage 1: Discriminators =====")
                    discriminators = result_full.get("stage1_discriminators", {})
                    print(f"  Category: {discriminators.get('category', 'N/A')}")
                    print(f"  Spatial: {discriminators.get('spatial', [])}")
                    print(f"  Temporal: {discriminators.get('temporal', [])}")
                    print(f"  Appearance: {discriminators.get('appearance', [])}")
                    print(f"  Relational: {discriminators.get('relational', [])}")

                    print(f"\n[BestView] ===== Stage 2: Tracking & Elimination =====")
                    tracking = result_full.get("stage2_tracking", {})
                    print(f"  Initial Candidates: {tracking.get('initial_candidates', [])}")
                    print(f"  Elimination Reasoning:")
                    for obj, reasoning in tracking.get("elimination_reasoning", {}).items():
                        print(f"    - {obj}: {reasoning}")
                    print(f"  Locked Targets: {tracking.get('locked_targets', [])}")

                    # Extract stage3_output for processing
                    result = result_full["stage3_output"]
                else:
                    # Old format, use directly
                    result = result_full

                num_objects = result["num_objects"]
                print(f"\n[BestView] MLLM determined {num_objects} objects")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[BestView] Warning: Failed to parse MLLM response: {e}")
                print(f"[BestView] Matched JSON string: {json_match.group()[:200]}...")
                # Fallback: analyze expression for plural
                result = self._fallback_object_detection(referring_expression, frame_indices)
        else:
            print(f"[BestView] Warning: No JSON found in MLLM response")
            # Fallback: analyze expression for plural
            result = self._fallback_object_detection(referring_expression, frame_indices)

        # Convert to output format
        best_frames = {}
        print(f"\n[BestView] ===== Stage 3: Selected Initialization Frames =====")
        for obj_info in result["objects"]:
            obj_id = obj_info["object_id"]
            frame_idx = obj_info["best_frame_index"]
            phrase = obj_info["simple_phrase"]
            reasoning = obj_info.get("reasoning", "N/A")
            best_frames[obj_id] = (frame_idx, phrase)
            print(f"[BestView] Object {obj_id}:")
            print(f"  - Frame: {frame_idx}")
            print(f"  - Phrase: '{phrase}'")
            print(f"  - Reasoning: {reasoning}")
        print(f"[BestView] ==========================================\n")

        return best_frames

    def _fallback_object_detection(
        self,
        referring_expression: str,
        frame_indices: List[int]
    ) -> dict:
        """
        Fallback method when MLLM fails to parse
        Uses heuristics to estimate number of objects
        """
        import re

        expression_lower = referring_expression.lower()

        # Strategy 1: Look for explicit numbers
        numbers = {
            "one": 1, "a": 1, "an": 1, "single": 1,
            "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "both": 2, "couple": 2, "pair": 2
        }

        for word, count in numbers.items():
            if word in expression_lower.split():
                return self._create_fallback_result(count, frame_indices, referring_expression)

        # Strategy 2: Look for digits
        digit_match = re.search(r'\b(\d+)\s+', referring_expression)
        if digit_match:
            count = int(digit_match.group(1))
            return self._create_fallback_result(count, frame_indices, referring_expression)

        # Strategy 3: Detect plural forms (CRITICAL for "Cows moving forward")
        # Common plural patterns
        if re.search(r'\b\w+s\s', expression_lower):  # words ending in 's' followed by space
            # Plural detected but no count specified
            # For RVOS, plural without count typically means 2-4 objects
            # Use conservative estimate: 2
            print(f"[BestView] Fallback: Detected plural form in '{referring_expression}', estimating 2+ objects")
            return self._create_fallback_result(2, frame_indices, referring_expression)

        # Strategy 4: Quantifiers
        multiple_indicators = ["all", "several", "some", "many", "multiple", "group"]
        for indicator in multiple_indicators:
            if indicator in expression_lower:
                return self._create_fallback_result(2, frame_indices, referring_expression)

        # Default: single object
        return self._create_fallback_result(1, frame_indices, referring_expression)

    def _create_fallback_result(
        self,
        num_objects: int,
        frame_indices: List[int],
        referring_expression: str
    ) -> dict:
        """Create a fallback result structure"""
        # Distribute objects across different frames for diversity
        objects = []
        for i in range(num_objects):
            # Select different frames for each object
            frame_idx = frame_indices[min(i * len(frame_indices) // num_objects, len(frame_indices) - 1)]
            objects.append({
                "object_id": i,
                "best_frame_index": frame_idx,
                "simple_phrase": f"{referring_expression} (object {i})" if num_objects > 1 else referring_expression,
                "reasoning": "Fallback: distributed across candidate frames"
            })

        return {
            "num_objects": num_objects,
            "reasoning_for_count": f"Heuristic-based estimation from expression analysis",
            "objects": objects
        }

    def _call_mllm_with_images(
        self,
        prompt: str,
        frames: List[np.ndarray],
        frame_indices: List[int]
    ) -> str:
        """
        Call Qwen2.5-VL with multi-image input for best view reasoning

        This is the CORE function that enables vision-language reasoning:
        - Analyzes multiple video frames simultaneously
        - Counts objects by visual inspection
        - Determines spatial positions from actual frame content
        - Generates simplified phrases based on visual attributes
        """
        from qwen_vl_utils import process_vision_info

        print(f"[BestView] Calling MLLM with {len(frames)} candidate frames...")

        # Convert frames to PIL
        pil_frames = [Image.fromarray(frame) for frame in frames]

        # Qwen2.5-VL multi-image input format
        # Each frame is added as a separate image in the content list
        content = []

        # Add all frames as images
        for idx, (pil_img, frame_idx) in enumerate(zip(pil_frames, frame_indices)):
            content.append({
                "type": "image",
                "image": pil_img
            })

        # Add the text prompt at the end
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs using processor (following official example)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Move to device (official way)
        inputs = inputs.to(self.device)

        # Generate with MLLM
        print(f"[BestView] Generating response from MLLM...")
        with torch.no_grad():
            generated_ids = self.mllm.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"[BestView] MLLM response received ({len(response)} chars)")

        return response


class SAM3Propagator:
    """Module 3: SAM3 Tracker with Bidirectional Propagation"""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize SAM3 tracker

        Args:
            checkpoint_path: Path to SAM3 checkpoint
            device: Device to run on (default: cuda)
        """
        self.device = device

        # Load SAM3 Video Predictor
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        print(f"[SAM3] Loading SAM3 Tracker from {checkpoint_path}...")
        self.predictor = Sam3VideoPredictor(
            checkpoint_path=checkpoint_path,
            has_presence_token=True,
            apply_temporal_disambiguation=True
        )

    def segment_and_propagate(
        self,
        video_path: str,
        best_frames: Dict[int, Tuple[int, str]],
        num_frames: int,
        video_id: str = "default"
    ) -> List[np.ndarray]:
        """
        Initialize SAM3 on best frames and propagate to full video

        Args:
            video_path: Path to video directory (MeViS JPEGImages/video_id)
            best_frames: Dict mapping obj_id to (frame_idx, phrase)
            num_frames: Total number of frames in video
            video_id: Video identifier for session

        Returns:
            List of masks (N, H, W) for each frame, with instance IDs
        """
        print(f"[SAM3] Starting propagation for {len(best_frames)} objects...")

        # Create base session ID
        session_id = f"{video_id}_{int(time.time())}"

        # Step 2: Process each object separately with independent sessions
        # Strategy: Run SAM3 propagation for each object individually, then merge masks

        all_object_masks = {}  # {frame_idx: {obj_id: mask}}

        for our_obj_id, (frame_idx, phrase) in best_frames.items():
            print(f"\n[SAM3] ========== Processing Object {our_obj_id} ==========")
            print(f"[SAM3] Text prompt: '{phrase}' at frame {frame_idx}")

            # Create a separate session for this object
            obj_session_id = f"{session_id}_obj{our_obj_id}"

            # Start session
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                    session_id=obj_session_id
                )
            )
            print(f"[SAM3] Session {obj_session_id} started")

            # Add prompt for this specific object
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=obj_session_id,
                    frame_index=frame_idx,
                    text=phrase,
                )
            )

            outputs = response.get("outputs", {})
            out_obj_ids = outputs.get("out_obj_ids", [])

            if len(out_obj_ids) == 0:
                print(f"[SAM3] ⚠⚠⚠ INITIALIZATION FAILED ⚠⚠⚠")
                print(f"[SAM3] Object {our_obj_id} could not be initialized")
                print(f"[SAM3] Frame: {frame_idx}, Phrase: '{phrase}'")
                print(f"[SAM3] This will result in empty masks for this object!")
                print(f"[SAM3] Response: {response}")
                _ = self.predictor.handle_request(
                    request=dict(type="close_session", session_id=obj_session_id)
                )
                continue

            print(f"[SAM3] ✓ Object {our_obj_id} initialized, propagating...")

            # Propagate through the entire video
            for response in self.predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=obj_session_id,
                )
            ):
                frame_idx_prop = response["frame_index"]
                outputs = response["outputs"]

                out_obj_ids = outputs.get("out_obj_ids", np.array([]))
                out_binary_masks = outputs.get("out_binary_masks", None)

                if out_binary_masks is not None and len(out_obj_ids) > 0:
                    # Initialize frame dict if needed
                    if frame_idx_prop not in all_object_masks:
                        all_object_masks[frame_idx_prop] = {}

                    # Store the mask for this object (use first detected object from SAM3)
                    all_object_masks[frame_idx_prop][our_obj_id] = out_binary_masks[0]

            # Close this object's session
            _ = self.predictor.handle_request(
                request=dict(type="close_session", session_id=obj_session_id)
            )
            print(f"[SAM3] Object {our_obj_id} propagation complete, session closed")

        # Close the main session (if still open)
        print(f"\n[SAM3] All objects processed, merging masks...")

        # Get video frame shape from first frame
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {video_path}")
        first_frame_path = os.path.join(video_path, frame_files[0])
        first_frame = Image.open(first_frame_path)
        H, W = first_frame.height, first_frame.width

        if len(all_object_masks) == 0:
            print("[SAM3] ⚠ No masks generated! Returning empty masks with original video resolution.")
            return [np.zeros((H, W), dtype=np.uint8) for _ in range(num_frames)]

        # Step 3: Merge all object masks into final frame masks
        final_masks = []

        # Verify mask shape matches video resolution
        first_frame_masks = list(all_object_masks.values())[0]
        sample_mask = list(first_frame_masks.values())[0]
        mask_h, mask_w = sample_mask.shape[-2:]
        if (mask_h, mask_w) != (H, W):
            print(f"[SAM3] Warning: SAM3 output resolution ({mask_h}, {mask_w}) differs from video resolution ({H}, {W})")
            H, W = mask_h, mask_w  # Use SAM3's output resolution

        for frame_idx in range(num_frames):
            # Create empty frame mask
            frame_mask = np.zeros((H, W), dtype=np.uint8)

            if frame_idx in all_object_masks:
                # Merge all objects for this frame
                for obj_id, binary_mask in all_object_masks[frame_idx].items():
                    # Assign instance ID (obj_id + 1, since 0 is background)
                    frame_mask[binary_mask > 0.5] = obj_id + 1

            final_masks.append(frame_mask)

            # Progress indicator
            if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                num_objects_in_frame = len(all_object_masks.get(frame_idx, {}))
                print(f"[SAM3] Merged frame {frame_idx + 1}/{num_frames}, objects: {num_objects_in_frame}")

        print(f"[SAM3] Propagation complete! Generated {len(final_masks)} masks with {len(best_frames)} objects")
        return final_masks


class MAPPSAM3Pipeline:
    """Complete MAPP-SAM3 Pipeline"""

    def __init__(
        self,
        clip_model_path: str,
        mllm_path: str,
        sam3_checkpoint: str,
        device: str = "cuda"
    ):
        """
        Initialize full pipeline

        Args:
            clip_model_path: Path to Long-CLIP checkpoint
            mllm_path: Path to MLLM model
            sam3_checkpoint: Path to SAM3 checkpoint
            device: Device to run on
        """
        self.device = device

        # Initialize modules
        print("=" * 60)
        print("Initializing MAPP-SAM3 Pipeline")
        print("=" * 60)

        self.qframe_selector = QFrameSelector(clip_model_path, device)
        self.bestview_reasoner = BestViewReasoner(mllm_path, sam3_checkpoint, device)
        self.sam3_propagator = SAM3Propagator(sam3_checkpoint, device)

        print("Pipeline initialized successfully!\n")

    def predict(
        self,
        video_frames: List[np.ndarray],
        video_path: str,
        referring_expression: str,
        video_id: str,
        num_candidates: int = 8
    ) -> List[np.ndarray]:
        """
        End-to-end prediction

        Args:
            video_frames: List of video frames (H, W, 3) - for Q-Frame and Best View
            video_path: Path to video directory - for SAM3
            referring_expression: Complex referring expression
            video_id: Video identifier
            num_candidates: Number of candidate frames for Q-Frame

        Returns:
            List of instance masks for each frame (H, W) with instance IDs
        """
        print("\n" + "=" * 60)
        print(f"Running MAPP-SAM3 on video with {len(video_frames)} frames")
        print(f"Video ID: {video_id}")
        print(f"Referring Expression: '{referring_expression}'")
        print("=" * 60 + "\n")

        # Module 1: Coarse Temporal Filtering
        print("[Stage 1/3] Q-Frame Temporal Filtering")
        print("-" * 60)
        candidate_indices = self.qframe_selector.select_keyframes(
            video_frames, referring_expression, num_candidates
        )
        print()

        # Module 2: Best View Reasoning
        print("[Stage 2/3] Best View Reasoning")
        print("-" * 60)

        # 2.1: Filter by presence
        filtered_indices = self.bestview_reasoner.filter_by_presence(
            video_frames, candidate_indices, referring_expression
        )

        # 2.2: Find best frames per object
        # MLLM will determine number of objects by analyzing expression + frames
        best_frames = self.bestview_reasoner.find_best_frames(
            video_frames, filtered_indices, referring_expression
        )

        # Keep MLLM loaded for subsequent expressions (following Sa2VA's approach)
        # Each predict() call is stateless - PyTorch will auto-release intermediate activations
        torch.cuda.empty_cache()
        print()

        # Module 3: SAM3 Propagation
        print("[Stage 3/3] SAM3 Bidirectional Propagation")
        print("-" * 60)
        masks = self.sam3_propagator.segment_and_propagate(
            video_path=video_path,
            best_frames=best_frames,
            num_frames=len(video_frames),
            video_id=video_id
        )
        print()

        print("=" * 60)
        print("MAPP-SAM3 Inference Complete!")
        print("=" * 60 + "\n")

        return masks

    def save_masks(self, masks: List[np.ndarray], output_dir: str):
        """
        Save mask predictions to PNG files in MeViS format

        Args:
            masks: List of instance masks (H, W) with instance IDs (0=background, 1,2,3...=objects)
            output_dir: Directory to save mask images
        """
        os.makedirs(output_dir, exist_ok=True)

        for i, mask in enumerate(masks):
            # Convert instance mask to binary mask (following Sa2VA format)
            # All non-zero values (1, 2, 3, ...) -> 255 (foreground)
            # Zero values -> 0 (background)
            binary_mask = (mask > 0).astype(np.uint8) * 255

            # Save with frame index naming: 00000.png, 00001.png, etc.
            frame_name = f"{i:05d}.png"
            mask_img = Image.fromarray(binary_mask)
            mask_img.save(os.path.join(output_dir, frame_name))

        print(f"[Pipeline] Saved {len(masks)} masks to {output_dir}")


# ============================================================================
# Utility Functions
# ============================================================================

def load_video_frames(video_path: str):
    """
    Load video frames from a directory containing image files

    Args:
        video_path: Path to directory containing video frames (image files)

    Returns:
        List of frames as numpy arrays (H, W, 3)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    # Get all image files in the directory
    frame_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        frame_files.extend(glob.glob(os.path.join(video_path, ext)))

    if not frame_files:
        raise FileNotFoundError(f"No image files found in: {video_path}")

    # Sort files to ensure consistent frame ordering
    frame_files.sort()

    # Load frames
    frames = []
    print(f"[load_video_frames] Loading {len(frame_files)} frames from {video_path}")

    for frame_file in frame_files:
        try:
            frame = np.array(Image.open(frame_file).convert("RGB"))
            frames.append(frame)
        except Exception as e:
            print(f"[load_video_frames] Warning: Failed to load {frame_file}: {e}")
            continue

    if not frames:
        raise ValueError(f"No valid frames could be loaded from: {video_path}")

    print(f"[load_video_frames] Successfully loaded {len(frames)} frames")
    return frames


# ============================================================================
# Demo Script
# ============================================================================

def load_mevis_sample(data_root: str, video_id: str, exp_id: int):
    """Load a sample from MeViS dataset"""
    # Load meta data
    meta_file = os.path.join(data_root, "meta_expressions.json")
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    # Get expression and frames
    video_meta = meta_data["videos"][video_id]
    exp_data = video_meta["expressions"][str(exp_id)]
    expression = exp_data["exp"]
    frame_names = video_meta["frames"]

    # Load video frames
    frames_dir = os.path.join(data_root, "JPEGImages", video_id)
    frames = []
    for frame_name in frame_names:
        frame_path = os.path.join(frames_dir, f"{frame_name}.jpg")
        frame = np.array(Image.open(frame_path).convert("RGB"))
        frames.append(frame)

    return frames, expression, frame_names


def main():
    """Demo on MeViS dataset"""
    import argparse
    import yaml

    # Parse arguments
    parser = argparse.ArgumentParser(description='MAPP-SAM3 Demo')
    parser.add_argument('--config', type=str, default='mapp_sam3_config.yaml',
                       help='Path to config file')
    parser.add_argument('--video_id', type=str, default='e339e02b5b3c',
                       help='Video ID to process')
    parser.add_argument('--exp_id', type=int, default=0,
                       help='Expression ID')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract paths from config
    clip_model_path = config['models']['clip_model_path']
    mllm_path = config['models']['mllm_path']
    sam3_checkpoint = config['models']['sam3_checkpoint']
    data_root = config['data']['mevis_root']
    device = config['models']['device']

    # Load a sample
    video_id = args.video_id
    exp_id = args.exp_id

    print("Loading MeViS sample...")
    frames, expression, frame_names = load_mevis_sample(
        data_root, video_id, exp_id
    )

    # Initialize pipeline
    pipeline = MAPPSAM3Pipeline(
        clip_model_path=clip_model_path,
        mllm_path=mllm_path,
        sam3_checkpoint=sam3_checkpoint,
        device=device
    )

    # Run inference
    num_candidates = config['q_frame']['num_candidates']
    print(f"Using {num_candidates} candidate frames for Q-Frame")
    video_path = os.path.join(data_root, "JPEGImages", video_id)
    masks = pipeline.predict(
        video_frames=frames,
        video_path=video_path,
        referring_expression=expression,
        video_id=video_id,
        num_candidates=num_candidates
    )

    # Save results
    output_dir = f"./mapp_sam3_results/{video_id}/{exp_id}"
    os.makedirs(output_dir, exist_ok=True)

    for i, (mask, frame_name) in enumerate(zip(masks, frame_names)):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(os.path.join(output_dir, f"{frame_name}.png"))

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
