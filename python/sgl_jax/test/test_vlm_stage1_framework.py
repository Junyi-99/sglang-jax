from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
    _build_mm_sidecars,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.models import qwen2_5_vl
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sgl_jax.srt.multimodal.models.qwen2_5VL import qwen2_5_vit


def _fake_visual():
    def compute_aux_arrays(grid_thw):
        rows = sum(int(t) * int(h) * int(w) for t, h, w in grid_thw)
        return (
            jnp.arange(rows, dtype=jnp.int32),
            jnp.zeros((rows, 1), dtype=jnp.float32),
            jnp.arange(len(grid_thw) + 1, dtype=jnp.int32),
            jnp.arange(rows + 1, dtype=jnp.int32),
        )

    return SimpleNamespace(visual=SimpleNamespace(compute_aux_arrays=compute_aux_arrays))


def _make_fake_encode_fn(per_dp_vision_size, slot_count, *, record=None):
    """Emulate the single encode shard_map (per-image ViT + folded compaction).

    The real device path runs the ViT then compacts each rank's valid feature
    rows locally into `[per_dp_vision_size, hidden]`. Tests treat the stacked
    `pixel_values[dp, slot, P_max, patch_dim]` rows as the per-slot features
    (grid (1,1,1) -> patch rows == feature rows), keep `valid_feature_rows`
    rows per slot, and pack them densely per rank.
    """

    def encode(
        pixel_values,
        window_index,
        rotary_pos_emb,
        cu_seqlens,
        cu_window_seqlens,
        valid_patch_rows,
        valid_feature_rows,
    ):
        pixel_values = np.asarray(pixel_values)
        valid_feature_rows = np.asarray(valid_feature_rows)
        if record is not None:
            record(pixel_values, window_index, valid_patch_rows, valid_feature_rows)
        dp_size = pixel_values.shape[0]
        hidden = pixel_values.shape[-1]
        by_rank = []
        for dp_rank in range(dp_size):
            local = []
            for slot_idx in range(slot_count):
                rows = int(valid_feature_rows[dp_rank, slot_idx])
                if rows:
                    local.append(pixel_values[dp_rank, slot_idx, :rows])
            if local:
                rank_features = np.concatenate(local, axis=0)
            else:
                rank_features = np.zeros((0, hidden), dtype=pixel_values.dtype)
            if rank_features.shape[0] < per_dp_vision_size:
                rank_features = np.concatenate(
                    [
                        rank_features,
                        np.zeros(
                            (per_dp_vision_size - rank_features.shape[0], hidden),
                            dtype=pixel_values.dtype,
                        ),
                    ],
                    axis=0,
                )
            by_rank.append(rank_features)
        return jnp.asarray(np.concatenate(by_rank, axis=0))

    return encode


def test_generate_req_getitem_preserves_media_fields():
    req = GenerateReqInput(
        text=["a", "b"],
        sampling_params=[{}, {}],
        rid=["r0", "r1"],
        return_logprob=[False, False],
        logprob_start_len=[-1, -1],
        top_logprobs_num=[0, 0],
        token_ids_logprob=[None, None],
        return_routed_experts=[False, False],
        image_data=[["image0"], ["image1"]],
        video_data=[["video0"], ["video1"]],
        audio_data=[["audio0"], ["audio1"]],
    )
    req.input_embeds = [["emb0"], ["emb1"]]

    item = req[1]

    assert item.image_data == ["image1"]
    assert item.video_data == ["video1"]
    assert item.audio_data == ["audio1"]
    assert item.input_embeds == ["emb1"]


def test_build_mm_sidecars_accepts_dict_and_dataclass_mm_inputs():
    item0 = SimpleNamespace(modality="image", offsets=[(1, 2)])
    item1 = SimpleNamespace(modality="image", offsets=[(5, 5), (7, 8)])
    reqs_info = [
        ScheduleReqsInfo(reqs=[SimpleNamespace(mm_inputs={"mm_items": [item0]})]),
        ScheduleReqsInfo(reqs=[SimpleNamespace(mm_inputs=SimpleNamespace(mm_items=[item1]))]),
    ]

    mm_items_by_rank, rank_vision_rows = _build_mm_sidecars(reqs_info, dp_size=2)

    assert mm_items_by_rank == [[item0], [item1]]
    assert rank_vision_rows == [2, 3]


def test_build_mm_sidecars_counts_only_image_items():
    image_item = SimpleNamespace(modality="image", offsets=[(1, 2)])
    audio_item = SimpleNamespace(modality="audio", offsets=[(3, 4)])
    reqs_info = [
        ScheduleReqsInfo(reqs=[SimpleNamespace(mm_inputs={"mm_items": [image_item, audio_item]})])
    ]

    mm_items_by_rank, rank_vision_rows = _build_mm_sidecars(reqs_info, dp_size=1)

    assert mm_items_by_rank == [[image_item]]
    assert rank_vision_rows == [2]


def test_model_worker_batch_carries_mm_sidecars_without_forward_batch_tree_child():
    item = SimpleNamespace(modality="image", offsets=[(0, 0)])
    batch = ScheduleBatch(
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[SimpleNamespace(mm_inputs={"mm_items": [item]}, lora_id="0")],
                input_ids=np.array([1], dtype=np.int32),
                seq_lens=np.array([1], dtype=np.int32),
                out_cache_loc=np.array([1], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([1], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
            )
        ],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
    )
    batch._merge_sampling_info = lambda per_dp_bs_size, total_bs: None
    batch._merge_cache_loc = lambda *args: np.array([1], dtype=np.int32)

    mwb = batch.get_model_worker_batch(
        token_paddings=[1],
        bs_paddings=[1],
        cache_loc_paddings=[1],
        page_size=1,
    )

    assert mwb.mm_items_by_rank == [[item]]
    assert mwb.rank_vision_rows == [1]
    assert "mm_items_by_rank" not in ForwardBatch.__dataclass_fields__
    assert "rank_vision_rows" not in ForwardBatch.__dataclass_fields__


def test_forward_batch_input_embedding_uses_data_axis_sharding():
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, ("data",))
    batch = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        input_ids=np.array([1], dtype=np.int32),
        real_input_ids_len=1,
        seq_lens=np.array([1], dtype=np.int32),
        out_cache_loc=np.array([1], dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
        sampling_info=None,
        positions=np.array([0], dtype=np.int32),
        cache_loc=np.array([1], dtype=np.int32),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=np.array([1], dtype=np.int32),
        extend_prefix_lens=np.array([0], dtype=np.int32),
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.array([0], dtype=np.int32),
        real_bs=1,
        real_bs_per_dp=[1],
        input_embedding=np.ones((1, 4), dtype=np.float32),
    )
    runner = SimpleNamespace(
        mesh=mesh,
        attn_backend=None,
        model_config=SimpleNamespace(
            is_embedding=False,
            hf_config=SimpleNamespace(architectures=[]),
        ),
    )
    captured_specs = []

    def fake_device_array(values, sharding):
        captured_specs.append(sharding.spec)
        return values

    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=fake_device_array,
    ):
        ForwardBatch.init_new(batch, runner)

    assert PartitionSpec("data", None) in captured_specs


def test_model_runner_multimodal_hook_noops_and_calls_encode_merge():
    class FakeMode:
        def __init__(self, is_extend):
            self._is_extend = is_extend

        def is_extend(self):
            return self._is_extend

    class FakeModel:
        image_token_id = 151655

        def __init__(self):
            self.calls = []

        def encode_mm(self, *, mm_items_by_rank, rank_vision_rows):
            self.calls.append(("encode", mm_items_by_rank, rank_vision_rows))
            return jnp.ones((1, 2), dtype=jnp.float32)

        def merge_mm(self, *, input_ids, placeholder_values, vision_features):
            self.calls.append(("merge", tuple(placeholder_values), vision_features.shape))
            return jnp.ones((input_ids.shape[0], 2), dtype=jnp.float32)

    runner = object.__new__(ModelRunner)
    runner.model = FakeModel()
    runner.model_config = SimpleNamespace(image_token_id=151655)

    text_batch = SimpleNamespace(mm_items_by_rank=None, rank_vision_rows=None)
    text_forward = SimpleNamespace(forward_mode=FakeMode(True), input_ids=jnp.array([1]))
    runner.prepare_inmodel_multimodal_forward(text_batch, text_forward)
    assert text_forward.__dict__.get("input_embedding") is None
    assert runner.model.calls == []

    image_item = SimpleNamespace(offsets=[(0, 0)])
    image_batch = SimpleNamespace(mm_items_by_rank=[[image_item]], rank_vision_rows=[1])
    image_forward = SimpleNamespace(
        forward_mode=FakeMode(True),
        input_ids=jnp.array([151655], dtype=jnp.int32),
        input_embedding=None,
    )
    runner.prepare_inmodel_multimodal_forward(image_batch, image_forward)

    assert [call[0] for call in runner.model.calls] == ["encode", "merge"]
    assert image_forward.input_embedding.shape == (1, 2)

    decode_forward = SimpleNamespace(
        forward_mode=FakeMode(False),
        input_ids=jnp.array([151655], dtype=jnp.int32),
        input_embedding=None,
    )
    runner.prepare_inmodel_multimodal_forward(image_batch, decode_forward)
    assert len(runner.model.calls) == 2


def test_model_runner_hook_ignores_empty_image_sidecar():
    class FakeMode:
        def __init__(self, is_extend):
            self._is_extend = is_extend

        def is_extend(self):
            return self._is_extend

    class FakeModel:
        image_token_id = 151655

        def __init__(self):
            self.calls = []

        def encode_mm(self, *, mm_items_by_rank, rank_vision_rows):
            self.calls.append(("encode", mm_items_by_rank, rank_vision_rows))
            return jnp.ones((1, 2), dtype=jnp.float32)

        def merge_mm(self, *, input_ids, placeholder_values, vision_features):
            self.calls.append(("merge", tuple(placeholder_values), vision_features.shape))
            return jnp.ones((input_ids.shape[0], 2), dtype=jnp.float32)

    runner = object.__new__(ModelRunner)
    runner.model = FakeModel()
    runner.model_config = SimpleNamespace(image_token_id=151655)
    batch = SimpleNamespace(mm_items_by_rank=[[]], rank_vision_rows=[0])
    forward = SimpleNamespace(
        forward_mode=FakeMode(True),
        input_ids=jnp.array([151655], dtype=jnp.int32),
        input_embedding=None,
    )

    runner.prepare_inmodel_multimodal_forward(batch, forward)

    assert runner.model.calls == []
    assert forward.input_embedding is None


def test_qwen2_5_vl_merge_uses_rank_local_vision_rows():
    image_token_id = 151655
    merged = qwen2_5_vl._merge_local_image_features(
        input_ids=jnp.array([image_token_id, 3, image_token_id], dtype=jnp.int32),
        text_embeds=jnp.full((3, 1), -1, dtype=jnp.float32),
        placeholders=jnp.array([image_token_id], dtype=jnp.int32),
        vision_features=jnp.array([[20.0], [21.0]], dtype=jnp.float32),
    )

    np.testing.assert_array_equal(
        np.asarray(merged).reshape(-1),
        np.array([20.0, -1.0, 21.0], dtype=np.float32),
    )


def test_qwen2_5_vl_merge_builder_uses_real_shard_map_specs():
    captured = {}

    def fake_shard_map(**kwargs):
        captured.update(kwargs)

        def decorate(fn):
            return fn

        return decorate

    with patch.object(qwen2_5_vl.jax, "shard_map", side_effect=fake_shard_map):
        qwen2_5_vl._build_merge_shard_map(Mesh(np.array(jax.devices()[:1]), ("data",)))

    assert captured["in_specs"] == (
        PartitionSpec("data"),
        PartitionSpec("data", None),
        PartitionSpec(None),
        PartitionSpec("data", None),
    )
    assert captured["out_specs"] == PartitionSpec("data", None)
    assert captured["check_vma"] is False


def test_qwen2_5_vl_initializes_visual_encoder():
    config = SimpleNamespace(
        vision_config=SimpleNamespace(marker="vision"),
        image_token_id=151655,
        vocab_size=8,
        hidden_size=2,
        num_hidden_layers=0,
        tie_word_embeddings=True,
    )

    with (
        patch.object(
            qwen2_5_vl.Qwen2_5_VL_Generation,
            "__init__",
            autospec=True,
        ) as base_init,
        patch.object(qwen2_5_vl, "Qwen2_5_VL_VisionModel") as vision_cls,
    ):
        base_init.return_value = None
        Qwen2_5_VLForConditionalGeneration(config=config, dtype=jnp.float32, mesh=None)

    base_init.assert_called_once()
    vision_cls.assert_called_once_with(
        config=config.vision_config,
        dtype=jnp.float32,
        rngs=None,
        mesh=None,
    )


def test_qwen2_5_vl_encode_builder_uses_data_axis_shard_map_specs():
    captured = {}

    def fake_shard_map(**kwargs):
        captured.update(kwargs)

        def decorate(fn):
            return fn

        return decorate

    fake_visual = SimpleNamespace(compute_hidden_states=lambda pixel_values, *args: pixel_values)
    model = SimpleNamespace(
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        visual=SimpleNamespace(visual=fake_visual),
        _encode_shard_map_mesh=lambda: Mesh(np.array(jax.devices()[:1]), ("data",)),
    )

    with (
        patch.object(qwen2_5_vl.nnx, "split", return_value=("def", "state")),
        patch.object(qwen2_5_vl.nnx, "merge", return_value=fake_visual),
        patch.object(qwen2_5_vl.jax, "shard_map", side_effect=fake_shard_map),
    ):
        Qwen2_5_VLForConditionalGeneration._build_encode_fn(
            model, per_dp_vision_size=1, slot_count=1
        )

    assert captured["in_specs"] == (
        PartitionSpec("data", None, None, None),
        PartitionSpec("data", None, None),
        PartitionSpec("data", None, None, None),
        PartitionSpec("data", None, None),
        PartitionSpec("data", None, None),
        PartitionSpec("data", None),
        PartitionSpec("data", None),
    )
    assert captured["out_specs"] == PartitionSpec("data", None)
    assert captured["check_vma"] is False


def test_qwen2_5_vl_encode_mm_uses_one_shard_map_call_for_mixed_size_slot():
    shard_calls = []

    def record(pixel_values, window_index, valid_patch_rows, valid_feature_rows):
        shard_calls.append(
            {
                "pixel_values": np.asarray(pixel_values).tolist(),
                "valid_patch_rows": np.asarray(valid_patch_rows).tolist(),
            }
        )

    model = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=2),
        dtype=jnp.float32,
        _data_hidden_sharding=lambda: None,
        mesh=None,
        visual=_fake_visual(),
        _get_encode_fn=lambda per_dp_vision_size, slot_count: _make_fake_encode_fn(
            per_dp_vision_size, slot_count, record=record
        ),
        _device_put_data_axis=lambda values, spec: values,
    )
    rank0_item = SimpleNamespace(
        feature=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
        offsets=[(0, 0)],
        model_specific_data={"image_grid_thw": (1, 1, 1)},
    )
    rank1_item = SimpleNamespace(
        feature=jnp.array([[3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32),
        offsets=[(0, 1)],
        model_specific_data={"image_grid_thw": ((1, 1, 1), (1, 1, 1))},
    )

    features = Qwen2_5_VLForConditionalGeneration.encode_mm(
        model,
        mm_items_by_rank=[[rank0_item], [rank1_item]],
        rank_vision_rows=[1, 2],
    )

    # ONE stacked encode call: pixel_values [dp=2, slot=1, P_max=2, patch_dim=2],
    # valid_patch_rows [dp=2, slot=1].
    assert shard_calls == [
        {
            "pixel_values": [
                [[[1.0, 2.0], [0.0, 0.0]]],
                [[[3.0, 4.0], [5.0, 6.0]]],
            ],
            "valid_patch_rows": [[1], [2]],
        }
    ]
    np.testing.assert_array_equal(
        np.asarray(features),
        np.array(
            [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        ),
    )


def test_qwen2_5_vl_encode_mm_rejects_rows_that_do_not_match_placeholders():
    model = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=2),
        dtype=jnp.float32,
        _data_hidden_sharding=lambda: None,
        mesh=None,
        visual=_fake_visual(),
        _get_encode_fn=lambda per_dp_vision_size, slot_count: _make_fake_encode_fn(
            per_dp_vision_size, slot_count
        ),
        _device_put_data_axis=lambda values, spec: values,
    )
    image_item = SimpleNamespace(
        feature=jnp.ones((2, 1), dtype=jnp.float32),
        offsets=[(4, 5)],
        model_specific_data={"image_grid_thw": (1, 1, 1)},
    )

    with pytest.raises(ValueError, match="actual feature rows 1.*placeholder rows 2"):
        Qwen2_5_VLForConditionalGeneration.encode_mm(
            model,
            mm_items_by_rank=[[image_item]],
            rank_vision_rows=[2],
        )


def test_qwen2_5_vl_encode_mm_rejects_dp_item_rows_hidden_by_slot_padding():
    model = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=1),
        dtype=jnp.float32,
        _data_hidden_sharding=lambda: None,
        mesh=None,
        visual=_fake_visual(),
        _get_encode_fn=lambda per_dp_vision_size, slot_count: _make_fake_encode_fn(
            per_dp_vision_size, slot_count
        ),
        _device_put_data_axis=lambda values, spec: values,
    )
    rank0_item = SimpleNamespace(
        feature=jnp.array([[10.0]], dtype=jnp.float32),
        offsets=[(0, 1)],
        model_specific_data={"image_grid_thw": (1, 1, 1)},
    )
    rank1_item = SimpleNamespace(
        feature=jnp.array([[20.0], [21.0], [22.0]], dtype=jnp.float32),
        offsets=[(0, 2)],
        model_specific_data={"image_grid_thw": ((1, 1, 1), (1, 1, 1), (1, 1, 1))},
    )

    with pytest.raises(ValueError, match="actual feature rows 1.*placeholder rows 2"):
        Qwen2_5_VLForConditionalGeneration.encode_mm(
            model,
            mm_items_by_rank=[[rank0_item], [rank1_item]],
            rank_vision_rows=[2, 3],
        )


def test_qwen2_5_vl_encode_mm_preserves_rank_order_and_padding():
    calls = []

    def record(pixel_values, window_index, valid_patch_rows, valid_feature_rows):
        calls.append(
            {
                "pixel_values": np.asarray(pixel_values).tolist(),
                "window_index": np.asarray(window_index).tolist(),
                "valid_patch_rows": np.asarray(valid_patch_rows).tolist(),
            }
        )

    model = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=1),
        dtype=jnp.float32,
        _data_hidden_sharding=lambda: None,
        mesh=None,
        visual=_fake_visual(),
        _get_encode_fn=lambda per_dp_vision_size, slot_count: _make_fake_encode_fn(
            per_dp_vision_size, slot_count, record=record
        ),
        _device_put_data_axis=lambda values, spec: values,
    )
    rank0_item0 = SimpleNamespace(
        feature=jnp.array([[10.0]], dtype=jnp.float32),
        offsets=[(0, 0)],
        model_specific_data={"image_grid_thw": (1, 1, 1)},
    )
    rank0_item1 = SimpleNamespace(
        feature=jnp.array([[11.0]], dtype=jnp.float32),
        offsets=[(1, 1)],
        model_specific_data={"image_grid_thw": (1, 1, 1)},
    )
    rank1_item0 = SimpleNamespace(
        feature=jnp.array([[20.0], [21.0]], dtype=jnp.float32),
        offsets=[(0, 1)],
        model_specific_data={"image_grid_thw": ((1, 1, 1), (1, 1, 1))},
    )

    features = Qwen2_5_VLForConditionalGeneration.encode_mm(
        model,
        mm_items_by_rank=[[rank0_item0, rank0_item1], [rank1_item0]],
        rank_vision_rows=[2, 2],
    )

    # ONE stacked encode call across all (rank, slot): leading [dp=2, slot=2].
    # rank0 has 2 real images, rank1 has 1 (slot 1 is a zero dummy lane).
    assert calls == [
        {
            "pixel_values": [
                [[[10.0], [0.0]], [[11.0], [0.0]]],
                [[[20.0], [21.0]], [[0.0], [0.0]]],
            ],
            "window_index": [
                [[0, 1], [0, 1]],
                [[0, 1], [0, 1]],
            ],
            "valid_patch_rows": [[1, 1], [2, 0]],
        }
    ]
    np.testing.assert_array_equal(
        np.asarray(features),
        np.array([[10.0], [11.0], [20.0], [21.0]], dtype=np.float32),
    )


def test_qwen2_5_vl_vision_attention_masks_padded_keys():
    q = jnp.ones((1, 2, 1, 1), dtype=jnp.float32)
    k = jnp.ones((1, 2, 1, 1), dtype=jnp.float32)
    v_with_large_pad = jnp.array([[[[2.0]], [[1000.0]]]], dtype=jnp.float32)
    v_without_large_pad = jnp.array([[[[2.0]], [[3.0]]]], dtype=jnp.float32)

    masked_large = qwen2_5_vit.vision_attention(
        q,
        k,
        v_with_large_pad,
        scale=1.0,
        valid_token_count=jnp.array(1, dtype=jnp.int32),
    )
    masked_small = qwen2_5_vit.vision_attention(
        q,
        k,
        v_without_large_pad,
        scale=1.0,
        valid_token_count=jnp.array(1, dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        np.asarray(masked_large[:, :1]),
        np.asarray(masked_small[:, :1]),
        rtol=1e-6,
    )


def test_qwen2_5_vl_encode_and_merge_compose_rank_local_rows():
    image_token_id = 151655
    model = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=1),
        dtype=jnp.float32,
        _data_hidden_sharding=lambda: None,
        _data_axis_size=lambda: 2,
        mesh=None,
        visual=_fake_visual(),
        _get_encode_fn=lambda per_dp_vision_size, slot_count: _make_fake_encode_fn(
            per_dp_vision_size, slot_count
        ),
        _device_put_data_axis=lambda values, spec: values,
        model=SimpleNamespace(
            embed_tokens=lambda ids: jnp.full((ids.shape[0], 1), -1, dtype=jnp.float32)
        ),
    )
    rank0_item = SimpleNamespace(
        feature=jnp.array([[10.0]], dtype=jnp.float32),
        offsets=[(0, 0)],
        model_specific_data={"image_grid_thw": (1, 1, 1)},
    )
    rank1_item = SimpleNamespace(
        feature=jnp.array([[20.0], [21.0]], dtype=jnp.float32),
        offsets=[(0, 1)],
        model_specific_data={"image_grid_thw": ((1, 1, 1), (1, 1, 1))},
    )

    vision_features = Qwen2_5_VLForConditionalGeneration.encode_mm(
        model,
        mm_items_by_rank=[[rank0_item], [rank1_item]],
        rank_vision_rows=[1, 2],
    )

    def fake_build_merge(_mesh):
        def fake_merge(input_ids, text_embeds, placeholders, vision_features):
            input_ids = input_ids.reshape(2, 2)
            text_embeds = text_embeds.reshape(2, 2, 1)
            vision_features = vision_features.reshape(2, 2, 1)
            merged = [
                qwen2_5_vl._merge_local_image_features(
                    input_ids[rank],
                    text_embeds[rank],
                    placeholders,
                    vision_features[rank],
                )
                for rank in range(2)
            ]
            return jnp.concatenate(merged, axis=0)

        return fake_merge

    with patch.object(qwen2_5_vl, "_build_merge_shard_map", side_effect=fake_build_merge):
        merged = Qwen2_5_VLForConditionalGeneration.merge_mm(
            model,
            input_ids=jnp.array(
                [image_token_id, 1, image_token_id, image_token_id], dtype=jnp.int32
            ),
            placeholder_values={image_token_id},
            vision_features=vision_features,
        )

    np.testing.assert_array_equal(
        np.asarray(merged).reshape(-1),
        np.array([10.0, -1.0, 20.0, 21.0], dtype=np.float32),
    )


def test_qwen2_5_vl_load_weights_reuses_visual_loader():
    config = SimpleNamespace(
        vision_config=SimpleNamespace(),
        image_token_id=151655,
        vocab_size=8,
        hidden_size=2,
        num_hidden_layers=0,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )
    visual = SimpleNamespace(load_weights=lambda model_config: None)
    with (
        patch.object(
            qwen2_5_vl.Qwen2_5_VL_Generation,
            "__init__",
            autospec=True,
            return_value=None,
        ),
        patch.object(qwen2_5_vl, "Qwen2_5_VL_VisionModel", return_value=visual),
    ):
        model = Qwen2_5_VLForConditionalGeneration(config=config, dtype=jnp.float32, mesh=None)

    with (
        patch.object(
            qwen2_5_vl.Qwen2_5_VL_Generation,
            "load_weights",
            autospec=True,
        ) as base_load,
        patch.object(model.visual, "load_weights", wraps=model.visual.load_weights) as visual_load,
    ):
        model_config = SimpleNamespace(model_path="/tmp/qwen2_5_vl")
        model.load_weights(model_config=model_config)

    base_load.assert_called_once_with(model, model_config)
    visual_load.assert_called_once()
    assert visual_load.call_args.args[0].model_path == "/tmp/qwen2_5_vl"


def test_qwen2_5_vl_load_weights_supplies_text_fields_to_visual_loader():
    config = SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=4),
        image_token_id=151655,
        vocab_size=17,
        hidden_size=5,
        num_hidden_layers=0,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )
    visual_load_configs = []

    def spy_visual_load(model_config):
        visual_load_configs.append(model_config)
        assert model_config.vocab_size == 17
        assert model_config.text_hidden_size == 5
        assert model_config.hidden_size == 4

    visual = SimpleNamespace(load_weights=spy_visual_load)
    with (
        patch.object(
            qwen2_5_vl.Qwen2_5_VL_Generation,
            "__init__",
            autospec=True,
            return_value=None,
        ),
        patch.object(qwen2_5_vl, "Qwen2_5_VL_VisionModel", return_value=visual),
    ):
        model = Qwen2_5_VLForConditionalGeneration(config=config, dtype=jnp.float32, mesh=None)
    model.text_config = SimpleNamespace(vocab_size=17, hidden_size=5)

    with (
        patch.object(
            qwen2_5_vl.Qwen2_5_VL_Generation,
            "load_weights",
            autospec=True,
        ),
    ):
        model.load_weights(model_config=SimpleNamespace(model_path="/tmp/qwen2_5_vl"))

    assert len(visual_load_configs) == 1
    assert not hasattr(config.vision_config, "vocab_size")
    assert not hasattr(config.vision_config, "text_hidden_size")


def test_mrope_positions_propagate_through_model_worker_batch():
    item = SimpleNamespace(modality="image", offsets=[(1, 1)])
    mrope_positions = np.array(
        [
            [0, 10, 2],
            [0, 11, 2],
            [0, 12, 2],
        ],
        dtype=np.int32,
    )
    batch = ScheduleBatch(
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[
                    SimpleNamespace(
                        mm_inputs={
                            "mm_items": [item],
                            "mrope_positions": mrope_positions,
                        },
                        lora_id="0",
                    )
                ],
                input_ids=np.array([1, 151655, 2], dtype=np.int32),
                seq_lens=np.array([3], dtype=np.int32),
                out_cache_loc=np.array([1, 2, 3], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([3], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
            )
        ],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
    )
    batch._merge_sampling_info = lambda per_dp_bs_size, total_bs: None
    batch._merge_cache_loc = lambda *args: np.array([1, 2, 3], dtype=np.int32)

    mwb = batch.get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )

    np.testing.assert_array_equal(mwb.mrope_positions[:, :3], mrope_positions)
