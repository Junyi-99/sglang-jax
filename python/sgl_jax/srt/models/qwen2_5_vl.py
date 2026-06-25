import functools
import logging
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import _count_mm_item_rows, _extract_mm_value
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vit import Qwen2_5_VL_VisionModel
from sgl_jax.srt.multimodal.models.qwen2_5VL.qwen2_5_vl_generation import (
    Qwen2_5_VL_Generation,
)

logger = logging.getLogger(__name__)


def _merge_local_image_features(
    input_ids,
    text_embeds,
    placeholders,
    vision_features,
):
    if placeholders.shape[0] == 0 or vision_features.shape[0] == 0:
        return text_embeds

    per_dp_vision_size = int(vision_features.shape[0])
    mask = jnp.any(input_ids[:, None] == placeholders[None, :], axis=1)
    gather_idx = jnp.cumsum(mask.astype(jnp.int32), axis=0) - 1
    # `clip` is a final safety net only; the contract (per-rank placeholder count
    # <= per-shard vision rows) is asserted host-side in `merge_mm` so a
    # violation surfaces instead of being silently capped here.
    safe_gather_idx = jnp.clip(
        jnp.where(mask, gather_idx, 0),
        0,
        per_dp_vision_size - 1,
    )
    vision_embeds = vision_features[safe_gather_idx]
    return jnp.where(mask[:, None], vision_embeds, text_embeds)


@functools.cache
def _build_merge_shard_map(mesh):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            PartitionSpec("data"),
            PartitionSpec("data", None),
            PartitionSpec(None),
            PartitionSpec("data", None),
        ),
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )
    def merge(input_ids, text_embeds, placeholders, vision_features):
        return _merge_local_image_features(
            input_ids,
            text_embeds,
            placeholders,
            vision_features,
        )

    return merge


def _normalize_image_grid_thw(grid_thw) -> tuple[tuple[int, int, int], ...]:
    if grid_thw is None:
        return ()
    if isinstance(grid_thw, tuple):
        if len(grid_thw) == 3 and all(np.isscalar(value) for value in grid_thw):
            return (tuple(int(value) for value in grid_thw),)
        return tuple(tuple(int(value) for value in row) for row in grid_thw)

    grid = np.asarray(grid_thw)
    if grid.size == 0:
        return ()
    if grid.ndim == 1:
        return (tuple(int(value) for value in grid.tolist()),)
    return tuple(tuple(int(value) for value in row) for row in grid.tolist())


def _get_item_image_grid_thw(item: Any) -> tuple[tuple[int, int, int], ...]:
    model_specific_data = _extract_mm_value(item, "model_specific_data") or {}
    if isinstance(model_specific_data, dict):
        grid_thw = model_specific_data.get("image_grid_thw")
    else:
        grid_thw = getattr(model_specific_data, "image_grid_thw", None)
    return _normalize_image_grid_thw(grid_thw)


def _count_grid_feature_rows(
    grid_thw: tuple[tuple[int, int, int], ...],
    spatial_merge_size: int,
) -> int:
    return sum(
        int(t) * (int(h) // spatial_merge_size) * (int(w) // spatial_merge_size)
        for t, h, w in grid_thw
    )


def _get_visual_config(config: Any) -> QwenVLModelVitConfig:
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        vision_config = getattr(config, "vision_config_dict", None)
    if vision_config is None:
        return QwenVLModelVitConfig()
    if isinstance(vision_config, QwenVLModelVitConfig):
        return vision_config
    if isinstance(vision_config, dict):
        config_obj = QwenVLModelVitConfig()
        for key, value in vision_config.items():
            setattr(config_obj, key, value)
        return config_obj
    return vision_config


def _copy_config_attrs(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def _config_attr(*configs: Any, name: str, default=None):
    for config in configs:
        if config is None:
            continue
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _build_visual_loader_config(
    *,
    visual_config: Any,
    text_config: Any,
    model_config: Any,
) -> Any:
    config = SimpleNamespace(**_copy_config_attrs(visual_config))
    model_path = _config_attr(model_config, visual_config, name="model_path")
    if model_path is not None:
        config.model_path = model_path

    vocab_size = _config_attr(text_config, model_config, name="vocab_size")
    hidden_size = _config_attr(text_config, model_config, name="hidden_size")
    text_hidden_size = _config_attr(
        text_config,
        model_config,
        name="text_hidden_size",
        default=hidden_size,
    )
    if vocab_size is not None:
        config.vocab_size = vocab_size
    if text_hidden_size is not None:
        config.text_hidden_size = text_hidden_size
    return config


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VL_Generation):
    """In-model Qwen2.5-VL normal image path.

    The visual encode and merge surfaces stay outside the backbone JIT while the
    language model and MRoPE path continue to come from the staged generation
    implementation.
    """

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__(config=config, dtype=dtype, mesh=mesh)
        self.mesh = getattr(self, "mesh", mesh)
        self.dtype = getattr(self, "dtype", dtype or jnp.bfloat16)
        self.visual_config = _get_visual_config(config)
        self.visual = Qwen2_5_VL_VisionModel(
            config=self.visual_config,
            dtype=self.dtype,
            rngs=rngs,
            mesh=mesh,
        )
        # Lazy cache for the single encode shard_map, keyed on the batch-dependent
        # static values `(per_dp_vision_size, slot_count)`. Built on demand by
        # `_get_encode_fn` and cleared in `load_weights` so reloaded ViT weights
        # are re-captured (mirrors the `_get_merge_fn` lazy-cache pattern).
        self._encode_fn_cache: dict[tuple[int, int], Any] = {}

    def _encode_shard_map_mesh(self):
        if getattr(self, "mesh", None) is not None:
            return self.mesh
        return Mesh(np.asarray(jax.devices()[:1]), ("data",))

    def _merge_shard_map_mesh(self):
        if getattr(self, "mesh", None) is not None:
            return self.mesh
        return Mesh(np.asarray(jax.devices()[:1]), ("data",))

    def _get_encode_fn(self, per_dp_vision_size: int, slot_count: int):
        """Lazily build (and cache) the single encode shard_map.

        `per_dp_vision_size` (the `segment_sum` `num_segments`) and `slot_count`
        (the unrolled Python slot loop count) are batch-dependent static values, so the
        shard_map closure depends on them. Caching on `(per_dp_vision_size,
        slot_count)` keeps it ONE build per distinct static pair (not a per-call
        rebuild) and lets JAX hit its trace/compile cache. Cleared in
        `load_weights` so reloaded ViT weights are re-captured.
        """
        key = (int(per_dp_vision_size), int(slot_count))
        cache = getattr(self, "_encode_fn_cache", None)
        if cache is None:
            cache = {}
            self._encode_fn_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        fn = self._build_encode_fn(per_dp_vision_size, slot_count)
        cache[key] = fn
        return fn

    def _build_encode_fn(self, per_dp_vision_size: int, slot_count: int):
        visual_def, visual_state = nnx.split(self.visual.visual)
        visual_state_leaves, visual_state_def = jax.tree_util.tree_flatten(visual_state)
        mesh = self._encode_shard_map_mesh()
        per_dp_vision_size = int(per_dp_vision_size)
        slot_count = int(slot_count)

        @jax.shard_map(
            mesh=mesh,
            in_specs=(
                PartitionSpec("data", None, None, None),
                PartitionSpec("data", None, None),
                PartitionSpec("data", None, None, None),
                PartitionSpec("data", None, None),
                PartitionSpec("data", None, None),
                PartitionSpec("data", None),
                PartitionSpec("data", None),
            ),
            out_specs=PartitionSpec("data", None),
            check_vma=False,
        )
        def encode_vision(
            pixel_values,
            window_index,
            rotary_pos_emb,
            cu_seqlens,
            cu_window_seqlens,
            valid_patch_rows,
            valid_feature_rows,
        ):
            visual_state = jax.tree_util.tree_unflatten(visual_state_def, visual_state_leaves)
            visual = nnx.merge(visual_def, visual_state)

            # This shard sees one DP rank's local slice: leading `slot_count`
            # axis stripped of the `data` shard. Run the per-image ViT for each
            # slot (static loop, unrolled at trace time), then compact this
            # rank's real feature rows locally -- no collective.
            feature_max_rows = int(window_index.shape[1])
            slot_features = []
            for slot in range(slot_count):
                features = visual.compute_hidden_states(
                    pixel_values[slot],
                    window_index[slot],
                    rotary_pos_emb[slot],
                    cu_seqlens[slot],
                    cu_window_seqlens[slot],
                    valid_patch_rows=valid_patch_rows[slot],
                )
                slot_features.append(
                    Qwen2_5_VLForConditionalGeneration._pad_rows_to(features, feature_max_rows)
                )
            # `[slot_count, feature_max_rows, hidden]`.
            slot_features = jnp.stack(slot_features, axis=0)
            hidden_size = int(slot_features.shape[-1])

            # Compaction (folded in from the former
            # `_compact_slot_features_shard_map`): keep only valid feature rows
            # per slot and pack them densely into `[per_dp_vision_size, hidden]`.
            valid_feature_rows_local = jnp.reshape(valid_feature_rows, (slot_count,))
            row_ids = jnp.arange(feature_max_rows)
            valid_mask = row_ids[None, :] < valid_feature_rows_local[:, None]
            flat_features = slot_features.reshape(slot_count * feature_max_rows, hidden_size)
            flat_mask = valid_mask.reshape(slot_count * feature_max_rows)
            compact_idx = jnp.cumsum(flat_mask.astype(jnp.int32), axis=0) - 1
            safe_idx = jnp.where(flat_mask, compact_idx, 0)
            weighted_features = flat_features * flat_mask[:, None].astype(flat_features.dtype)
            return jax.ops.segment_sum(
                weighted_features,
                safe_idx,
                num_segments=per_dp_vision_size,
            )

        return encode_vision

    def load_weights(self, model_config):
        super().load_weights(model_config)
        visual_config = getattr(self, "visual_config", model_config)
        visual_loader_config = _build_visual_loader_config(
            visual_config=visual_config,
            text_config=getattr(self, "text_config", None),
            model_config=model_config,
        )
        self.visual.load_weights(visual_loader_config)
        # Clear the encode shard_map cache so reloaded ViT weights are
        # re-captured by the next `_get_encode_fn` build.
        self._encode_fn_cache = {}

    def _data_hidden_sharding(self):
        if getattr(self, "mesh", None) is None:
            return None
        return NamedSharding(self.mesh, PartitionSpec("data", None))

    def _data_axis_size(self) -> int:
        if getattr(self, "mesh", None) is None:
            return 1
        mesh_shape = getattr(self.mesh, "shape", {})
        return int(mesh_shape.get("data", 1))

    def _device_put_data_axis(self, value, spec):
        if getattr(self, "mesh", None) is None:
            return value
        return jax.device_put(value, NamedSharding(self.mesh, spec))

    def _spatial_merge_size(self) -> int:
        visual_model = getattr(self, "visual", None)
        visual = getattr(visual_model, "visual", None)
        return int(
            _config_attr(
                visual,
                getattr(visual_model, "config", None),
                getattr(self, "visual_config", None),
                name="spatial_merge_size",
                default=1,
            )
        )

    def _prepare_encode_item(self, item):
        pixel_values = jnp.asarray(_extract_mm_value(item, "feature")).astype(self.dtype)
        image_grid_thw = _get_item_image_grid_thw(item)
        window_index, rotary_pos_emb, cu_seqlens, cu_window_seqlens = (
            self.visual.visual.compute_aux_arrays(image_grid_thw)
        )
        placeholder_rows = _count_mm_item_rows(item)
        return {
            "pixel_values": pixel_values,
            "window_index": window_index,
            "rotary_pos_emb": rotary_pos_emb,
            "cu_seqlens": cu_seqlens,
            "cu_window_seqlens": cu_window_seqlens,
            "valid_patch_rows": int(pixel_values.shape[0]),
            "placeholder_rows": placeholder_rows,
            "valid_feature_rows": _count_grid_feature_rows(
                image_grid_thw,
                Qwen2_5_VLForConditionalGeneration._spatial_merge_size(self),
            ),
        }

    @staticmethod
    def _pad_rows_to(value, target_rows: int):
        """Zero-pad `value` (a `[rows, ...]` array) up to `target_rows` rows.

        Pure-JAX variant of `_pad_rows` for use inside the encode shard_map body
        (no `self`, no host-side int validation -- shapes are static under trace).
        """
        rows = int(value.shape[0])
        if rows == target_rows:
            return value
        pad_shape = (target_rows - rows, *value.shape[1:])
        padding = jnp.zeros(pad_shape, dtype=value.dtype)
        return jnp.concatenate([value, padding], axis=0)

    def _pad_rows(self, value, target_rows: int, *, dtype=None):
        rows = int(value.shape[0])
        if rows == target_rows:
            return value.astype(dtype) if dtype is not None else value
        if rows > target_rows:
            raise ValueError(f"cannot pad {rows} rows down to {target_rows}")
        pad_shape = (target_rows - rows, *value.shape[1:])
        padding = jnp.zeros(pad_shape, dtype=dtype or value.dtype)
        value = value.astype(dtype) if dtype is not None else value
        return jnp.concatenate([value, padding], axis=0)

    def _pad_window_index(self, window_index, target_feature_rows: int):
        rows = int(window_index.shape[0])
        if rows == target_feature_rows:
            return window_index
        if rows > target_feature_rows:
            raise ValueError(f"window_index rows {rows} exceed target rows {target_feature_rows}")
        padding = jnp.arange(rows, target_feature_rows, dtype=window_index.dtype)
        return jnp.concatenate([window_index, padding], axis=0)

    def _zero_encode_lane(
        self,
        *,
        max_patch_rows: int,
        max_feature_rows: int,
        patch_dim: int,
        rotary_dim: int,
        cu_seqlens_size: int,
        cu_window_seqlens_size: int,
    ):
        return {
            "pixel_values": jnp.zeros((max_patch_rows, patch_dim), dtype=self.dtype),
            "window_index": jnp.arange(max_feature_rows, dtype=jnp.int32),
            "rotary_pos_emb": jnp.zeros((max_patch_rows, rotary_dim), dtype=jnp.float32),
            "cu_seqlens": jnp.zeros((cu_seqlens_size,), dtype=jnp.int32),
            "cu_window_seqlens": jnp.zeros((cu_window_seqlens_size,), dtype=jnp.int32),
            "valid_patch_rows": 0,
            "valid_feature_rows": 0,
        }

    def _pad_encode_lane(
        self,
        prepared,
        *,
        max_patch_rows: int,
        max_feature_rows: int,
        cu_seqlens_size: int,
        cu_window_seqlens_size: int,
    ):
        return {
            "pixel_values": Qwen2_5_VLForConditionalGeneration._pad_rows(
                self,
                prepared["pixel_values"],
                max_patch_rows,
                dtype=self.dtype,
            ),
            "window_index": Qwen2_5_VLForConditionalGeneration._pad_window_index(
                self,
                prepared["window_index"],
                max_feature_rows,
            ),
            "rotary_pos_emb": Qwen2_5_VLForConditionalGeneration._pad_rows(
                self,
                prepared["rotary_pos_emb"],
                max_patch_rows,
            ),
            "cu_seqlens": Qwen2_5_VLForConditionalGeneration._pad_rows(
                self,
                prepared["cu_seqlens"],
                cu_seqlens_size,
            ),
            "cu_window_seqlens": Qwen2_5_VLForConditionalGeneration._pad_rows(
                self,
                prepared["cu_window_seqlens"],
                cu_window_seqlens_size,
            ),
            "valid_patch_rows": prepared["valid_patch_rows"],
            "valid_feature_rows": prepared["valid_feature_rows"],
        }

    def _prepare_rank_slots(self, rank_items, slot_count, rank_vision_rows):
        """Prepare every present (rank, slot) item and validate the feature-row
        contracts. Returns `prepared_by_slot[slot][dp_rank] -> prepared dict`.
        """
        dp_size = len(rank_items)
        prepared_by_slot: list[dict[int, dict[str, Any]]] = []
        encoded_rows_by_rank = [0 for _ in range(dp_size)]
        for slot_idx in range(slot_count):
            prepared_by_rank: dict[int, dict[str, Any]] = {}
            for dp_rank, items in enumerate(rank_items):
                if slot_idx >= len(items):
                    continue
                prepared = Qwen2_5_VLForConditionalGeneration._prepare_encode_item(
                    self, items[slot_idx]
                )
                # Per (rank, slot) feature-row contract: actual encoded rows must
                # match the placeholder offsets for this item.
                valid_feature_rows = int(prepared["valid_feature_rows"])
                placeholder_rows = int(prepared["placeholder_rows"])
                if valid_feature_rows != placeholder_rows:
                    raise ValueError(
                        f"rank {dp_rank} image item actual feature rows "
                        f"{valid_feature_rows}, but offsets require "
                        f"placeholder rows {placeholder_rows}"
                    )
                encoded_rows_by_rank[dp_rank] += valid_feature_rows
                prepared_by_rank[dp_rank] = prepared
            prepared_by_slot.append(prepared_by_rank)

        # Per-rank total feature-row contract: the rows this rank encodes across
        # all its slots must equal the sidecar's requested rows.
        for dp_rank, expected_rows in enumerate(rank_vision_rows or []):
            expected_rows = int(expected_rows)
            if encoded_rows_by_rank[dp_rank] != expected_rows:
                raise ValueError(
                    f"rank {dp_rank} encoded {encoded_rows_by_rank[dp_rank]} "
                    f"vision rows, but sidecar requires {expected_rows}"
                )
        return prepared_by_slot

    def _compute_global_slot_padding(self, prepared_by_slot):
        """Global padding maxima across ALL (rank, slot) pairs. Padding to a
        single global max (vs per-slot max) is slightly more padding but lets the
        whole batch flow through ONE shard_map call with a uniform stacked shape.
        """
        all_prepared = [
            prepared
            for prepared_by_rank in prepared_by_slot
            for prepared in prepared_by_rank.values()
        ]
        representative = all_prepared[0]
        return {
            "p_max": max(prepared["valid_patch_rows"] for prepared in all_prepared),
            "f_max": max(int(prepared["window_index"].shape[0]) for prepared in all_prepared),
            "cu_max": max(int(prepared["cu_seqlens"].shape[0]) for prepared in all_prepared),
            "cuw_max": max(
                int(prepared["cu_window_seqlens"].shape[0]) for prepared in all_prepared
            ),
            "patch_dim": int(representative["pixel_values"].shape[-1]),
            "rotary_dim": int(representative["rotary_pos_emb"].shape[-1]),
        }

    def _build_padded_encode_inputs(self, prepared_by_slot, dims, dp_size, slot_count):
        """Pad every (rank, slot) lane to the global maxima, stack to
        `[dp_size, slot_count, ...]`, and shard on the data axis. Returns the 7
        inputs in `encode` call order; missing (rank, slot) pairs use a dummy lane.
        """
        p_max = dims["p_max"]
        f_max = dims["f_max"]
        cu_max = dims["cu_max"]
        cuw_max = dims["cuw_max"]
        dummy_lane = Qwen2_5_VLForConditionalGeneration._zero_encode_lane(
            self,
            max_patch_rows=p_max,
            max_feature_rows=f_max,
            patch_dim=dims["patch_dim"],
            rotary_dim=dims["rotary_dim"],
            cu_seqlens_size=cu_max,
            cu_window_seqlens_size=cuw_max,
        )

        pixel_values_by_rank = []
        window_index_by_rank = []
        rotary_pos_emb_by_rank = []
        cu_seqlens_by_rank = []
        cu_window_seqlens_by_rank = []
        valid_patch_rows_by_rank = []
        valid_feature_rows_by_rank = []
        for dp_rank in range(dp_size):
            pixel_values_slots = []
            window_index_slots = []
            rotary_pos_emb_slots = []
            cu_seqlens_slots = []
            cu_window_seqlens_slots = []
            valid_patch_rows_slots = []
            valid_feature_rows_slots = []
            for slot_idx in range(slot_count):
                prepared = prepared_by_slot[slot_idx].get(dp_rank)
                lane = (
                    Qwen2_5_VLForConditionalGeneration._pad_encode_lane(
                        self,
                        prepared,
                        max_patch_rows=p_max,
                        max_feature_rows=f_max,
                        cu_seqlens_size=cu_max,
                        cu_window_seqlens_size=cuw_max,
                    )
                    if prepared is not None
                    else dummy_lane
                )
                pixel_values_slots.append(lane["pixel_values"])
                window_index_slots.append(lane["window_index"])
                rotary_pos_emb_slots.append(lane["rotary_pos_emb"])
                cu_seqlens_slots.append(lane["cu_seqlens"])
                cu_window_seqlens_slots.append(lane["cu_window_seqlens"])
                valid_patch_rows_slots.append(lane["valid_patch_rows"])
                valid_feature_rows_slots.append(lane["valid_feature_rows"])
            pixel_values_by_rank.append(jnp.stack(pixel_values_slots, axis=0))
            window_index_by_rank.append(jnp.stack(window_index_slots, axis=0))
            rotary_pos_emb_by_rank.append(jnp.stack(rotary_pos_emb_slots, axis=0))
            cu_seqlens_by_rank.append(jnp.stack(cu_seqlens_slots, axis=0))
            cu_window_seqlens_by_rank.append(jnp.stack(cu_window_seqlens_slots, axis=0))
            valid_patch_rows_by_rank.append(valid_patch_rows_slots)
            valid_feature_rows_by_rank.append(valid_feature_rows_slots)

        # Stack across ranks + shard on the data axis. Drive the per-input
        # `device_put` with a (host array, PartitionSpec) spec list (replaces the
        # 7 near-identical device_put blocks).
        specs = (
            (jnp.stack(pixel_values_by_rank, axis=0), PartitionSpec("data", None, None, None)),
            (jnp.stack(window_index_by_rank, axis=0), PartitionSpec("data", None, None)),
            (jnp.stack(rotary_pos_emb_by_rank, axis=0), PartitionSpec("data", None, None, None)),
            (jnp.stack(cu_seqlens_by_rank, axis=0), PartitionSpec("data", None, None)),
            (jnp.stack(cu_window_seqlens_by_rank, axis=0), PartitionSpec("data", None, None)),
            (jnp.asarray(valid_patch_rows_by_rank, dtype=jnp.int32), PartitionSpec("data", None)),
            (jnp.asarray(valid_feature_rows_by_rank, dtype=jnp.int32), PartitionSpec("data", None)),
        )
        return tuple(
            Qwen2_5_VLForConditionalGeneration._device_put_data_axis(self, array, spec)
            for array, spec in specs
        )

    def encode_mm(
        self,
        mm_items_by_rank: list[list[object]],
        rank_vision_rows: list[int],
    ):
        dp_size = max(len(mm_items_by_rank or []), len(rank_vision_rows or []), 1)
        per_dp_vision_size = max(rank_vision_rows or [0])
        hidden_size = int(self.text_config.hidden_size)
        if getattr(self, "mesh", None) is not None and self._data_axis_size() != dp_size:
            raise ValueError(
                f"encode sidecar dp_size {dp_size} must match mesh data axis "
                f"{self._data_axis_size()}"
            )

        rank_items = [
            (
                mm_items_by_rank[dp_rank]
                if mm_items_by_rank is not None and dp_rank < len(mm_items_by_rank)
                else []
            )
            for dp_rank in range(dp_size)
        ]
        slot_count = max((len(items) for items in rank_items), default=0)

        # Empty image batch: no shard_map needed, return zeros that match the
        # `[dp_size * per_dp_vision_size, hidden]` contract.
        if slot_count == 0:
            features = jnp.zeros((dp_size * per_dp_vision_size, hidden_size), dtype=self.dtype)
            sharding = self._data_hidden_sharding()
            if sharding is not None:
                features = jax.device_put(features, sharding)
            return features

        prepared_by_slot = Qwen2_5_VLForConditionalGeneration._prepare_rank_slots(
            self, rank_items, slot_count, rank_vision_rows
        )
        dims = Qwen2_5_VLForConditionalGeneration._compute_global_slot_padding(
            self, prepared_by_slot
        )
        (
            pixel_values,
            window_index,
            rotary_pos_emb,
            cu_seqlens,
            cu_window_seqlens,
            valid_patch_rows,
            valid_feature_rows,
        ) = Qwen2_5_VLForConditionalGeneration._build_padded_encode_inputs(
            self, prepared_by_slot, dims, dp_size, slot_count
        )

        encode = self._get_encode_fn(per_dp_vision_size, slot_count)
        features = encode(
            pixel_values,
            window_index,
            rotary_pos_emb,
            cu_seqlens,
            cu_window_seqlens,
            valid_patch_rows,
            valid_feature_rows,
        )

        sharding = self._data_hidden_sharding()
        if sharding is not None:
            features = jax.device_put(features, sharding)
        return features

    def merge_mm(
        self,
        input_ids,
        placeholder_values,
        vision_features,
    ):
        text_embeds = self.model.embed_tokens(input_ids)
        placeholders = jnp.asarray(sorted(placeholder_values), dtype=input_ids.dtype)
        dp_size = self._data_axis_size()
        token_size = int(input_ids.shape[0])
        vision_size = int(vision_features.shape[0])
        if token_size % dp_size != 0:
            raise ValueError(f"input_ids size {token_size} must be divisible by dp_size {dp_size}")
        if vision_size % dp_size != 0:
            raise ValueError(
                f"vision_features size {vision_size} must be divisible by dp_size {dp_size}"
            )

        # Contract guard: each rank's local cumsum-gather can only address its own
        # `vision_size // dp_size` vision rows. If a shard's placeholder count
        # exceeds that, the in-shard `clip` would silently cap the gather index and
        # mis-gather. Surface it here (host-side count requires a device->host sync,
        # acceptable for a correctness guard) instead of corrupting embeddings.
        per_dp_token = token_size // dp_size
        per_dp_vision = vision_size // dp_size
        per_rank_ids = jnp.asarray(input_ids).reshape(dp_size, per_dp_token)
        per_rank_mask = jnp.any(per_rank_ids[:, :, None] == placeholders[None, None, :], axis=2)
        per_rank_placeholders = np.asarray(jnp.sum(per_rank_mask.astype(jnp.int32), axis=1))
        for dp_rank in range(dp_size):
            count = int(per_rank_placeholders[dp_rank])
            if count > per_dp_vision:
                raise ValueError(
                    f"rank {dp_rank} has {count} image placeholders but only "
                    f"{per_dp_vision} local vision rows are available"
                )

        # `input_ids` (from forward_batch) is 1-D `P("data")`; its delivery
        # sharding is not guaranteed to already match the shard_map in_spec, so
        # keep this alignment to stay safe at dp>1.
        input_ids = Qwen2_5_VLForConditionalGeneration._device_put_data_axis(
            self, input_ids, PartitionSpec("data")
        )
        # `text_embeds` comes straight from `embed_tokens`; its output sharding is
        # not guaranteed to be `P("data", None)`, so keep this alignment.
        text_embeds = Qwen2_5_VLForConditionalGeneration._device_put_data_axis(
            self, text_embeds, PartitionSpec("data", None)
        )
        # `vision_features` already arrives as `P("data", None)` from
        # `encode_mm`'s final `device_put`, which is exactly the merge in_spec, so
        # no further reshard is needed.
        merge = Qwen2_5_VLForConditionalGeneration._get_merge_fn(self)
        inputs_embeds = merge(input_ids, text_embeds, placeholders, vision_features)
        # `out_specs=P("data", None)` already produces the target sharding, so the
        # shard_map result is returned as-is (no redundant reshard).
        return inputs_embeds

    def _get_merge_fn(self):
        mesh = Qwen2_5_VLForConditionalGeneration._merge_shard_map_mesh(self)
        cached = getattr(self, "_merge_fn", None)
        if cached is not None and getattr(self, "_merge_fn_mesh", None) is mesh:
            return cached
        # `_build_merge_shard_map` is memoized on `mesh`, so the same mesh reuses
        # the same shard_map function and JAX hits the trace/compile cache.
        merge = _build_merge_shard_map(mesh)
        self._merge_fn = merge
        self._merge_fn_mesh = mesh
        return merge


EntryClass = Qwen2_5_VLForConditionalGeneration
