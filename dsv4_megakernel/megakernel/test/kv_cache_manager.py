"""
kv_cache_manager.py -- host-side KV cache management for the DSV4 decode chain
(kccache_design.png). P2 form: the KERNELS write every cache byte directly
(wq_b: indexer fused pages + SWA MODEL1 pages + idx state pool; mqa_logits
tail: Main-compressed MODEL1 pages + main state pool reads) -- this module only
owns ALLOCATION and ADDRESSING:

1. C4ARequestStateContext (persistent request slots, state_slot_mapping[M]):
     main_kv/main_sc fp32 [capacity, 8, 1024]   (mqa tail reads, ping-pong)
     idx_kv/idx_sc   fp32 [capacity, 8,  256]   (wq_b comp chain, INOUT)
   alloc_request(): KV = 0, score = -inf (design lifecycle contract).

2. Three paged pools + per-step dst indices (dst = page*64 + off, -1 = skip):
     SWA/Window      MODEL1 584B/tok  ring of window_size=128 (model.py
                     kv_cache[:, pos %% win]) -> 2 fixed pages per request
     Main compressed MODEL1 584B/tok  append at ct = (pos+1)/4 - 1
     Indexer         fused 68B/tok    append at ct (same compressed axis)

3. Consumption views: block tables (mqa/topk page transform), MODEL1 cache
   views + swa/cmp indices_in_kvcache for FlashMLA.

The ONLY host-side data write left is write_main_state (front's main_comp
segment -> the main state fresh row): its producer kernel is a pending op
(design: the segment producer owns it); the copy here is a plain relayout,
not a numeric simulation.
"""
import torch

NEG_INF = float("-inf")

PAGE = 64
WIN = 128                       # model.py window_size (SWA ring)
RATIO = 4
SROWS = 8
D_NOPE, D_ROPE, TILE, NTILES = 448, 64, 64, 7
M1_TOK_BODY = D_NOPE + 2 * D_ROPE                                   # 576
M1_PAGE_BYTES = (PAGE * (M1_TOK_BODY + 8) + 575) // 576 * 576       # 37440
M1_TAIL_OFF = PAGE * M1_TOK_BODY
M1_TOK_BYTES = M1_TOK_BODY + NTILES + 1                             # 584
IDX_D = 128
IDX_PAGE_BYTES = PAGE * (IDX_D // 2 + 4)                            # 4352


class KVCacheManager:
    """Design-image pools + slot lifecycle + dst-index builders (no data)."""

    def __init__(self, capacity, pages_per_pool, max_pages_per_req, device="cuda"):
        f32 = dict(device=device, dtype=torch.float32)
        u8 = dict(device=device, dtype=torch.uint8)
        # 1. C4ARequestStateContext (kernel-addressed via slot_map)
        self.main_kv = torch.zeros(capacity, SROWS, 1024, **f32)
        self.main_sc = torch.full((capacity, SROWS, 1024), NEG_INF, **f32)
        self.idx_kv = torch.zeros(capacity, SROWS, 256, **f32)
        self.idx_sc = torch.full((capacity, SROWS, 256), NEG_INF, **f32)
        # 2. paged pools (kernel-written)
        self.swa_pool = torch.zeros(pages_per_pool, M1_PAGE_BYTES, **u8)
        self.cmp_pool = torch.zeros(pages_per_pool, M1_PAGE_BYTES, **u8)
        self.idx_pool = torch.zeros(pages_per_pool, IDX_PAGE_BYTES, **u8)
        self._free_pages = {p: list(range(pages_per_pool - 1, -1, -1))
                            for p in ("swa", "cmp", "idx")}
        self.capacity, self.max_pages = capacity, max_pages_per_req
        self.device = device
        self._free_slots = list(range(capacity - 1, -1, -1))
        self.reqs = {}   # slot -> dict(pos, swa/cmp/idx page lists)

    # ---- slot lifecycle (design: new/reused slot => KV=0, score=-inf) ----
    def alloc_request(self):
        slot = self._free_slots.pop()
        self.main_kv[slot].zero_(); self.main_sc[slot].fill_(NEG_INF)
        self.idx_kv[slot].zero_();  self.idx_sc[slot].fill_(NEG_INF)
        # SWA ring: fixed WIN/PAGE pages for the request's lifetime.
        self.reqs[slot] = dict(
            pos=-1, cmp=[], idx=[],
            swa=[self._free_pages["swa"].pop() for _ in range(WIN // PAGE)])
        return slot

    def free_request(self, slot):
        r = self.reqs.pop(slot)
        for p in ("swa", "cmp", "idx"):
            self._free_pages[p].extend(reversed(r[p]))
        self._free_slots.append(slot)

    def _page_of(self, name, slot, tok):
        table = self.reqs[slot][name]
        while len(table) <= tok // PAGE:
            table.append(self._free_pages[name].pop())
        return table[tok // PAGE]

    # ---- per-step batch bundle: pos, slot_map and the three dst vectors ----
    def step_begin(self, slots):
        for s in slots:
            self.reqs[s]["pos"] += 1
        pos = [self.reqs[s]["pos"] for s in slots]
        idx_dst, cmp_dst, swa_dst = [], [], []
        for s, p in zip(slots, pos):
            # SWA ring slot (model.py kv_cache[:, pos % win]), every token.
            ring = p % WIN
            swa_dst.append(self.reqs[s]["swa"][ring // PAGE] * PAGE
                           + ring % PAGE)
            if (p + 1) % RATIO == 0:            # compress row: ct-th entry
                ct = (p + 1) // RATIO - 1
                idx_dst.append(self._page_of("idx", s, ct) * PAGE + ct % PAGE)
                cmp_dst.append(self._page_of("cmp", s, ct) * PAGE + ct % PAGE)
            else:
                idx_dst.append(-1)
                cmp_dst.append(-1)
        t = lambda v, dt: torch.tensor(v, dtype=dt, device=self.device)
        return dict(
            slots=list(slots),
            pos=t(pos, torch.int64), q_pos=t(pos, torch.int32),
            slot_map=t(list(slots), torch.int32),
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
    # Ping-pong mapping (mqa_logits_fp4.cuh [B1]): fresh logical row 4+p%4,
    # physical = (4*((p>>2)&1) + 4 + p%4) & 7; kv|sc = front main_comp segment.
    def write_main_state(self, slots, pos, main_seg_f32):
        for i, s in enumerate(slots):
            p = int(pos[i])
            phys = ((4 * ((p >> 2) & 1)) + 4 + (p & 3)) & 7
            self.main_kv[s, phys] = main_seg_f32[i, :1024]
            self.main_sc[s, phys] = main_seg_f32[i, 1024:]

    def main_state_rows(self, slots, pos):
        """[B] i32 flat pool row (slot*8 + ping-pong physical row) for the
        FRONT-EMIT direct state write."""
        p = pos.long()
        phys = ((4 * ((p >> 2) & 1)) + 4 + (p & 3)) & 7
        s = torch.tensor(slots, dtype=torch.long, device=self.device)
        return (s * SROWS + phys).int()

    # ---- FlashMLA-side views -------------------------------------------------
    def model1_cache_view(self, name):
        """[num_pages, PAGE, 1, 584] fp8 view (flash_mla k_cache layout)."""
        pool = self.swa_pool if name == "swa" else self.cmp_pool
        return pool[:, : PAGE * M1_TOK_BYTES] \
            .view(torch.float8_e4m3fn).view(-1, PAGE, 1, M1_TOK_BYTES)

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
            phys = [table[(t % WIN) // PAGE] * PAGE + (t % WIN) % PAGE
                    for t in range(p + 1 - n, p + 1)]
            out[i, 0, :n] = torch.tensor(phys, dtype=torch.int32,
                                         device=self.device)
            lens[i] = n
        return out, lens

    def n_compressed(self, slots, pos):
        return torch.tensor([(int(pos[i]) + 1) // RATIO
                             for i in range(len(slots))],
                            dtype=torch.int32, device=self.device)
