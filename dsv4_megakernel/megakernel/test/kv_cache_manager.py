"""
kv_cache_manager.py -- host-side KV cache management for the DSV4 decode chain
(kccache_design.png). P2 form: the KERNELS write every cache byte directly
(wq_b: indexer fused pages + SWA MODEL1 pages + idx state pool; mqa_logits
tail: Main-compressed MODEL1 pages + main state pool reads) -- this module only
owns ALLOCATION and ADDRESSING:

1. RTP state rings, one tensor per compressor:
     main_state fp32 [capacity, ring, 2048] = [kv(1024) | score(1024)]
     idx_state  fp32 [capacity, ring,  512] = [kv(256) | score(256)]
   Writers use block*ring + pos%ring rows; -1 skips a row.

2. Three paged pools with independent logical entries and physical strides:
     SWA/Window      MODEL1 584B/tok, 256 entries/block in RTP
     Main compressed MODEL1 584B/tok  append at ct = (pos+1)/4 - 1
     Indexer         FP8 132B/tok     append at ct (same compressed axis)

3. Consumption views: block tables (mqa/topk page transform), MODEL1 cache
   views + swa/cmp indices_in_kvcache for FlashMLA.

The production path writes all state/cache bytes in kernels. write_main_state
remains only as a test fallback.
"""
import torch

NEG_INF = float("-inf")

TOKENS_PER_BLOCK = 256
RATIO = 4
PAGE = TOKENS_PER_BLOCK // RATIO
WIN = 128                       # model.py window_size (SWA ring)
SROWS = 8
D_NOPE, D_ROPE, TILE, NTILES = 448, 64, 64, 7
M1_TOK_BODY = D_NOPE + 2 * D_ROPE                                   # 576
M1_TOK_BYTES = M1_TOK_BODY + NTILES + 1                             # 584
IDX_D = 128
IDX_ENTRIES_PER_BLOCK = TOKENS_PER_BLOCK // RATIO                    # 64
CMP_ENTRIES_PER_BLOCK = TOKENS_PER_BLOCK // RATIO                    # 64
SWA_ENTRIES_PER_BLOCK = TOKENS_PER_BLOCK                             # 256


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


IDX_FP8_ENTRY_BYTES = IDX_D + 4
IDX_FP4_ENTRY_BYTES = IDX_D // 2 + 4
IDX_PAGE_BYTES = IDX_ENTRIES_PER_BLOCK * IDX_FP8_ENTRY_BYTES
M1_PAGE_BYTES = _align_up(CMP_ENTRIES_PER_BLOCK * M1_TOK_BYTES, 576)
M1_TAIL_OFF = CMP_ENTRIES_PER_BLOCK * M1_TOK_BODY


class KVCacheManager:
    """Design-image pools + slot lifecycle + dst-index builders (no data)."""

    def __init__(self, capacity, pages_per_pool, max_pages_per_req, device="cuda",
                 state_ring_entries=SROWS,
                 idx_entries_per_block=IDX_ENTRIES_PER_BLOCK,
                 cmp_entries_per_block=CMP_ENTRIES_PER_BLOCK,
                 swa_entries_per_block=SWA_ENTRIES_PER_BLOCK,
                 indexer_fp8=True,
                 idx_block_stride_bytes=None,
                 cmp_block_stride_bytes=None,
                 swa_block_stride_bytes=None):
        f32 = dict(device=device, dtype=torch.float32)
        u8 = dict(device=device, dtype=torch.uint8)
        self.state_ring_entries = int(state_ring_entries)
        self.indexer_fp8 = bool(indexer_fp8)
        self.idx_entry_bytes = (IDX_FP8_ENTRY_BYTES if self.indexer_fp8
                                else IDX_FP4_ENTRY_BYTES)
        if self.state_ring_entries < SROWS:
            raise ValueError("state ring must cover the 8-entry overlap window")
        self.entries_per_block = {
            "idx": int(idx_entries_per_block),
            "cmp": int(cmp_entries_per_block),
            "swa": int(swa_entries_per_block),
        }
        payloads = {
            "idx": self.entries_per_block["idx"] * self.idx_entry_bytes,
            "cmp": self.entries_per_block["cmp"] * M1_TOK_BYTES,
            "swa": self.entries_per_block["swa"] * M1_TOK_BYTES,
        }
        requested_strides = {
            "idx": idx_block_stride_bytes,
            "cmp": cmp_block_stride_bytes,
            "swa": swa_block_stride_bytes,
        }
        self.block_stride_bytes = {
            name: int(requested_strides[name] or
                      (payload if name == "idx" else _align_up(payload, 576)))
            for name, payload in payloads.items()
        }
        for name, payload in payloads.items():
            if self.entries_per_block[name] <= 0:
                raise ValueError(f"{name} entries_per_block must be positive")
            if self.block_stride_bytes[name] < payload:
                raise ValueError(f"{name} block stride does not cover its payload")

        # 1. Framework state entries concatenate kv and score halves.
        self.main_state = torch.empty(capacity, self.state_ring_entries, 2048, **f32)
        self.idx_state = torch.empty(capacity, self.state_ring_entries, 512, **f32)
        self._reset_state(torch.arange(capacity, device=device))
        # 2. paged pools (kernel-written)
        self.swa_pool = torch.zeros(pages_per_pool, self.block_stride_bytes["swa"], **u8)
        self.cmp_pool = torch.zeros(pages_per_pool, self.block_stride_bytes["cmp"], **u8)
        self.idx_pool = torch.zeros(pages_per_pool, self.block_stride_bytes["idx"], **u8)
        self._free_pages = {p: list(range(pages_per_pool - 1, -1, -1))
                            for p in ("swa", "cmp", "idx")}
        self.capacity, self.max_pages = capacity, max_pages_per_req
        self.device = device
        self._free_slots = list(range(capacity - 1, -1, -1))
        self.reqs = {}   # slot -> dict(pos, swa/cmp/idx page lists)

    def _reset_state(self, slots):
        self.main_state[slots, :, :1024].zero_()
        self.main_state[slots, :, 1024:].fill_(NEG_INF)
        self.idx_state[slots, :, :256].zero_()
        self.idx_state[slots, :, 256:].fill_(NEG_INF)

    # ---- slot lifecycle (design: new/reused slot => KV=0, score=-inf) ----
    def alloc_request(self):
        slot = self._free_slots.pop()
        self._reset_state(slot)
        # The RTP SWA block has 256 entries and covers the 128-token ring.
        self.reqs[slot] = dict(
            pos=-1, cmp=[], idx=[],
            swa=[self._free_pages["swa"].pop()
                 for _ in range((WIN + self.entries_per_block["swa"] - 1) //
                                self.entries_per_block["swa"])])
        return slot

    def free_request(self, slot):
        r = self.reqs.pop(slot)
        for p in ("swa", "cmp", "idx"):
            self._free_pages[p].extend(reversed(r[p]))
        self._free_slots.append(slot)

    def _page_of(self, name, slot, tok):
        table = self.reqs[slot][name]
        epb = self.entries_per_block[name]
        while len(table) <= tok // epb:
            table.append(self._free_pages[name].pop())
        return table[tok // epb]

    # ---- per-step batch bundle: positions, state rows, and cache destinations ----
    def step_begin(self, slots):
        for s in slots:
            self.reqs[s]["pos"] += 1
        pos = [self.reqs[s]["pos"] for s in slots]
        idx_dst, cmp_dst, swa_dst = [], [], []
        for s, p in zip(slots, pos):
            # SWA ring slot (model.py kv_cache[:, pos % win]), every token.
            ring = p % WIN
            swa_epb = self.entries_per_block["swa"]
            swa_dst.append(self.reqs[s]["swa"][ring // swa_epb] * swa_epb
                           + ring % swa_epb)
            if (p + 1) % RATIO == 0:            # compress row: ct-th entry
                ct = (p + 1) // RATIO - 1
                idx_epb = self.entries_per_block["idx"]
                cmp_epb = self.entries_per_block["cmp"]
                idx_dst.append(self._page_of("idx", s, ct) * idx_epb + ct % idx_epb)
                cmp_dst.append(self._page_of("cmp", s, ct) * cmp_epb + ct % cmp_epb)
            else:
                idx_dst.append(-1)
                cmp_dst.append(-1)
        t = lambda v, dt: torch.tensor(v, dtype=dt, device=self.device)
        return dict(
            slots=list(slots),
            pos=t(pos, torch.int64), q_pos=t(pos, torch.int32),
            main_state_row=self.state_rows(slots, pos),
            idx_state_row=self.state_rows(slots, pos),
            idx_dst=t(idx_dst, torch.int32), cmp_dst=t(cmp_dst, torch.int32),
            swa_dst=t(swa_dst, torch.int32))

    def block_table(self, name, slots):
        bt = torch.zeros(len(slots), self.max_pages, dtype=torch.int32,
                         device=self.device)
        for i, s in enumerate(slots):
            tab = self.reqs[s][name]
            if tab:
                bt[i, :len(tab)] = torch.tensor(tab, dtype=torch.int32,
                                                device=self.device)
        return bt

    # ---- MAIN state fresh-row write. LEGACY host path (the production form
    # is FRONT-EMIT: front_mixed's epilogue scatters fp32 accum + ape into
    # the pool rows below directly; this stays for tests/fallback).
    def write_main_state(self, slots, pos, main_seg_f32):
        for i, s in enumerate(slots):
            p = int(pos[i])
            row = p % self.state_ring_entries
            self.main_state[s, row] = main_seg_f32[i]

    def main_state_rows(self, slots, pos):
        return self.state_rows(slots, pos)

    def state_rows(self, slots, pos):
        """RTP folded state rows: block * ring + pos % ring."""
        p = torch.as_tensor(pos, dtype=torch.long, device=self.device) \
            % self.state_ring_entries
        s = torch.tensor(slots, dtype=torch.long, device=self.device)
        return (s * self.state_ring_entries + p).int()

    def geometry(self, name):
        return dict(entries_per_block=self.entries_per_block[name],
                    block_stride_bytes=self.block_stride_bytes[name])

    # ---- FlashMLA-side views -------------------------------------------------
    def model1_cache_view(self, name):
        """[num_pages, PAGE, 1, 584] fp8 view (flash_mla k_cache layout)."""
        pool = self.swa_pool if name == "swa" else self.cmp_pool
        epb = self.entries_per_block[name]
        stride = self.block_stride_bytes[name]
        return pool.as_strided((pool.size(0), epb, 1, M1_TOK_BYTES),
                               (stride, M1_TOK_BYTES, M1_TOK_BYTES, 1)) \
            .view(torch.float8_e4m3fn)

    def indexer_cache_view(self):
        """DeepGEMM fused FP8 cache view with RTP's grouped K/scale layout.

        The logical last dimension is 132 bytes, while the underlying page is
        physically [all 128B K records | all 4B FP32 scales]. DeepGEMM derives
        those two views from the base pointer and block stride.
        """
        if not self.indexer_fp8:
            raise ValueError("FP8 Indexer cache view requested for an FP4 pool")
        epb = self.entries_per_block["idx"]
        stride = self.block_stride_bytes["idx"]
        return self.idx_pool.as_strided(
            (self.idx_pool.size(0), epb, 1, IDX_FP8_ENTRY_BYTES),
            (stride, IDX_FP8_ENTRY_BYTES, IDX_FP8_ENTRY_BYTES, 1))

    def swa_indices(self, slots, pos, swa_topk):
        """indices_in_kvcache [B,1,swa_topk] of the last min(swa_topk, pos+1)
        tokens through the RING (token t lives at ring slot t %% WIN); -1 pad."""
        B = len(slots)
        out = torch.full((B, 1, swa_topk), -1, dtype=torch.int32,
                         device=self.device)
        lens = torch.zeros(B, dtype=torch.int32, device=self.device)
        for i, s in enumerate(slots):
            p = int(pos[i])
            n = min(swa_topk, p + 1)
            table = self.reqs[s]["swa"]
            epb = self.entries_per_block["swa"]
            phys = [table[(t % WIN) // epb] * epb + (t % WIN) % epb
                    for t in range(p + 1 - n, p + 1)]
            out[i, 0, :n] = torch.tensor(phys, dtype=torch.int32,
                                         device=self.device)
            lens[i] = n
        return out, lens

    def n_compressed(self, slots, pos):
        return torch.tensor([(int(pos[i]) + 1) // RATIO
                             for i in range(len(slots))],
                            dtype=torch.int32, device=self.device)
