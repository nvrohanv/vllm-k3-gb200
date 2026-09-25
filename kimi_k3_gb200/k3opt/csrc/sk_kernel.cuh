// moebody stream-K kernel (M >= 5; included by moebody.cu inside namespace moebody).
// moe8's all-SM schedule (agents/moe8/moe8.cu), restructured for the persistent-kernel body:
//   * x -> MXFP8 before the routing is known; routing through the epoch flag (flag mode) or griddepcontrol;
//   * every post-routing code path runs once in a dry / warm-up mode before the routing arrives
//     (noinline per-item functions), so its first real execution hits a warm instruction cache;
//   * NG = 1..3 expert groups: FC1(g0) .. FC1(g_{NG-1}), FC2(g0) .. FC2(g_{NG-1}) per CTA: group g's h
//     hand-off (global memory, data-carried validity, no fences) is hidden behind the next groups' work.
// Work split (per group g, D_g experts): FC1 = 42 * D_g (tile, K-stage) units cut into G contiguous ranges
// (stream-K; a tile split across CTAs is reduced by the CTA holding its first K-stage, the others write
// fp32 partials, 0xFFFFFFFF = empty, re-armed by the owner); FC2 = 28 * D_g row tiles in G contiguous
// ranges. The owner writes SiTU(h) as MXFP8 straight into the FC2 B-operand image of a global buffer
// (0xFF = empty; double-buffered by call parity, the idle parity re-armed during the call); FC2 consumers
// poll the image itself.
#pragma once



struct SkGeo {
  int ng, G;
  int ub[kSkMaxNG + 1];  // expert-slot bounds of the groups
  int s1[kSkMaxNG];      // FC1 units per group
  int f1lo[kSkMaxNG], f1hi[kSkMaxNG], f2lo[kSkMaxNG], f2hi[kSkMaxNG];
};

__device__ __noinline__ void sk_geometry(SkGeo& g, int D, int G, int cta, int ng) {
  g.ng = ng;
  g.G = G;
  for (int q = 0; q <= ng; ++q) g.ub[q] = (D * q) / ng;
  for (int q = 0; q < ng; ++q) {
    const int De = g.ub[q + 1] - g.ub[q];
    g.s1[q] = kFc1Seq * De;
    g.f1lo[q] = split_lo(cta, g.s1[q], G);
    g.f1hi[q] = split_lo(cta + 1, g.s1[q], G);
    g.f2lo[q] = kFc2Tiles * g.ub[q] + split_lo(cta, kFc2Tiles * De, G);
    g.f2hi[q] = kFc2Tiles * g.ub[q] + split_lo(cta + 1, kFc2Tiles * De, G);
  }
}

// One epilogue item of the stream-K kernel (8-token accumulators). warm != 0: warm-up (global writes go to
// scratch, no partial polling, no TMEM release).
__device__ EPI_FN void sk_epi_item(uint8_t* sm, uint32_t tacc, uint32_t accempty, int mtype, int ma, int mb, int mc,
                                         int eg, int ew, int lane, int M, int cta, float* pbase, uint8_t* hset,
                                         uint8_t* scratch, int warm, EpiConst ec) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const SkGeo& geo = *reinterpret_cast<const SkGeo*>(ms.skgeo);
  float v[8];
  tmem_ldn<8>(tacc, v);
  tc::tc_fence_before();
  __syncwarp();
  if (lane == 0 && accempty != 0) mbar_arrive_u32(accempty);
  const int row = 32 * ew + lane;  // physical row within the 128-row tile
  if (mtype == kFC1) {
    const int T = ma;
    int g = 0;
    while (g + 1 < geo.ng && T >= kFc1Tiles * geo.ub[g + 1]) ++g;
    const int s0 = (T - kFc1Tiles * geo.ub[g]) * kFc1Stages;  // group-local first stage of the tile
    const bool owner = warm ? warm == 2 : s0 >= geo.f1lo[g];
    if (!owner) {  // not the owner: publish the partial
      float* dst = warm ? reinterpret_cast<float*>(scratch) + row
                        : pbase + (static_cast<long>(g) * kMaxG + cta) * 1024 + row;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        if (t < M) dst[t * 128] = v[t];
      return;
    }
    if (eg == 0 && ew == 0 && lane == 0 && kTraceOn && ec.trace != nullptr && !warm) ec.trace[cta * 128 + 12] = tc::globaltimer();
    // other pieces of this tile: CTAs cta+1.. whose ranges start inside the tile
    for (int c2 = cta + 1; !warm && c2 < geo.G && split_lo(c2, geo.s1[g], geo.G) < s0 + kFc1Stages; ++c2) {
      if (split_lo(c2 + 1, geo.s1[g], geo.G) == split_lo(c2, geo.s1[g], geo.G)) continue;  // empty range
      float* src = pbase + (static_cast<long>(g) * kMaxG + c2) * 1024 + row;
      float pv[8];
      const long long pc0 = PCLK();
      for (;;) {
        bool bad = false;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          pv[t] = 0.f;
          if (t < M) {
            const uint32_t w = ld_volatile_u32(src + t * 128);
            bad |= (w == 0xffffffffu);
            pv[t] = __uint_as_float(w);
          }
        }
        if (!__any_sync(0xffffffffu, bad)) break;
      }
      if (kTraceOn && lane == 0 && ew == 0) ms.epi_poll[eg] += PCLK() - pc0;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        if (t < M) {
          v[t] += pv[t];
          src[t * 128] = __uint_as_float(0xffffffffu);
        }
    }
    const int u = T / 3, r = T % 3;
    const int ebar = eg == 0 ? 1 : 4;
    const bool up_lane = (lane & 8) == 0;
    const int i_local = 2 * (lane % 8) + lane / 16;
    float h[8], am[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const float o = __shfl_xor_sync(0xffffffffu, v[t], 8);  // the gate row sits 8 lanes up
      const float hv = ec.beta_lb * tanh_fast(o * ec.inv_beta) * sigmoid_fast(o) * tanh_fast(v[t] * ec.inv_lb);
      h[t] = up_lane ? hv : 0.f;
      am[t] = fabsf(h[t]);
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], off));
    }
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < 8; ++t) ms.amax[eg][ew][t] = am[t];
    }
    asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");
    int sbt = 0;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int sb = mx_exp(fmaxf(am[t], ms.amax[eg][ew ^ 1][t]));
      if (lane == t) sbt = sb;
      if (up_lane)
        ms.hst[eg][ew][t][i_local] =
            static_cast<uint8_t>(__nv_cvt_float_to_fp8(h[t] * mx_rescale(sb), __NV_SATFINITE, __NV_E4M3));
    }
    __syncwarp();
    uint8_t* himg = warm ? scratch : hset + static_cast<long>(u) * kHImg;
    if (lane < 8) {
      const int t = lane, kc = 4 * r + ew;
      const uint4 val = *reinterpret_cast<const uint4*>(ms.hst[eg][ew][t]);
      *reinterpret_cast<uint4*>(himg + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16)) = val;
      if ((ew & 1) == 0) himg[2048 + t * 16 + 2 * r + ew / 2] = static_cast<uint8_t>(sbt);
    }
    asm volatile("bar.sync %0, 128;" ::"r"(ebar) : "memory");  // ms.amax / hst reuse
    if (eg == 0 && ew == 0 && lane == 0 && kTraceOn && ec.trace != nullptr && !warm) ec.trace[cta * 128 + 6] = tc::globaltimer();
  } else {
    const int u = ma, mt = mb;
    const uint2 ts = warm ? make_uint2(0xffffff00u, ~0u) : *reinterpret_cast<const uint2*>(ms.tokslot[u]);
    if (warm) {
      EpiConst wc = ec;
      wc.out = reinterpret_cast<__nv_bfloat16*>(scratch);
      wc.mode = 0;
      fc2_emit<8>(v, ts, mt, ew, lane, wc);
    } else {
      fc2_emit<8>(v, ts, mt, ew, lane, ec);
    }
  }
}

// h loader item: poll expert image `src` (global) until complete, copy it into h buffer b, build the three
// per-FC1-tile SFB images (row t: bytes sb[2r], sb[2r+1]) and arrive on hfull[b]. warm: no polling / arrive.
__device__ __noinline__ void sk_h_load(uint8_t* sm, const uint8_t* src, int b, int lane, int warm) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  uint4 q[4];
  uint32_t s4;
  for (;;) {
    bool bad = false;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int ci = lane + 32 * v;  // 16-B chunk of the 2 KB image
      q[v] = ld_volatile_v4(src + ci * 16);
      const int atom = ci >> 6, t = (ci >> 3) & 7, pos = ci & 7;
      if (atom == 0 || (pos ^ t) < 4)
        bad |= has_ff_byte(q[v].x) | has_ff_byte(q[v].y) | has_ff_byte(q[v].z) | has_ff_byte(q[v].w);
    }
    // compact scales: row t = lane/4 at bytes 16*t + 4*(lane%4); k-blocks 0..5 are written.
    s4 = ld_volatile_u32(src + 2048 + lane * 4);
    if ((lane & 3) < 2) bad |= has_ff_byte(s4 & ((lane & 3) == 0 ? 0xffffffffu : 0x0000ffffu));
    if (warm || !__any_sync(0xffffffffu, bad)) break;
    __nanosleep(20);
  }
  uint4* dq = reinterpret_cast<uint4*>(sm + kOffHq + b * 2048);
#pragma unroll
  for (int v = 0; v < 4; ++v) dq[lane + 32 * v] = q[v];
  {
    const int r = lane >> 3, t = lane & 7;  // lanes 0..23: image r, row t
    const int srcl = 4 * t + (r == 2 ? 1 : 0);
    const uint32_t w = __shfl_sync(0xffffffffu, s4, srcl);
    const uint32_t word = ((r == 1) ? (w >> 16) : w) & 0xffffu;
    if (lane < 24) *reinterpret_cast<uint32_t*>(sm + kOffHsf + b * 768 + r * 128 + t * 16) = word;
  }
  tc::fence_proxy_async_smem();
  __syncwarp();
  if (lane == 0 && !warm) tc::mbar_arrive(&ms.hfull[b]);
}


// ================================================================ stream-K TMA producer (one thread); dry: one fake
// FC1 and one FC2 stage, same instructions, nothing loaded or published.
__device__ __noinline__ void sk_producer(uint8_t* sm, const CUtensorMap* tmA1, const CUtensorMap* tmA2, const Params& p,
                                         int dry) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const SkGeo& geo = *reinterpret_cast<const SkGeo*>(ms.skgeo);
  const uint32_t live = dry ? 0u : 1u;
  int slot = 0, phase = 0, n = 0;
  const int ng = dry ? 1 : geo.ng;
#pragma unroll 1
  for (int g = 0; g < 2 * ng; ++g) {
    const bool fc1 = g < ng;
    const int q = fc1 ? g : g - ng;
    const int lo = dry ? 0 : (fc1 ? geo.f1lo[q] : geo.f2lo[q]);
    const int hi = dry ? 1 : (fc1 ? geo.f1hi[q] : geo.f2hi[q]);
#pragma unroll 1
    for (int s = lo; s < hi; ++s) {
      Meta m;
      const CUtensorMap* map;
      int row0, kc;
      const uint8_t* sf;
      if (fc1) {
        const int T = kFc1Tiles * (dry ? 0 : geo.ub[q]) + s / kFc1Stages, ks = s % kFc1Stages;
        const int e = dry ? 0 : ms.expert[T / 3], r = T % 3;
        const int flags = ((s == lo || ks == 0) ? fFirst : 0) | ((s == hi - 1 || ks == kFc1Stages - 1) ? fLast : 0);
        m = Meta{kFC1, T, ks, flags};
        map = tmA1;
        row0 = e * kW13Rows + r * 128;
        kc = 2 * ks;
        sf = p.w13s + static_cast<long>(e) * kW13ScaleBytes + r * 14336 + ks * 1024;
      } else {
        const int u = s / kFc2Tiles, mt = s % kFc2Tiles, e = dry ? 0 : ms.expert[u];
        m = Meta{kFC2, u, mt, 0};
        map = tmA2;
        row0 = e * kHidden + mt * 128;
        kc = 0;
        sf = p.w2s + static_cast<long>(e) * kW2ScaleBytes + mt * 1024;
      }
      if (n >= kSlots && live) tc::mbar_wait(&ms.empty[slot], phase ^ 1);
      if (n == 0 && live) trace_ev(p, 4);
      if (live) ms.meta[slot] = m;
      tma_stage(sm, slot, map, row0, kc, sf, live);
      ++n;
      if (++slot == kSlots) {
        slot = 0;
        phase ^= 1;
      }
    }
  }
  if (!live) return;
  trace_ev(p, 5);
  if (n >= kSlots) tc::mbar_wait(&ms.empty[slot], phase ^ 1);
  ms.meta[slot] = Meta{kEnd, 0, 0, 0};
  tc::mbar_arrive(&ms.full[slot]);
}

// ================================================================ stream-K MMA issuer (whole warp); dry as above.
__device__ __noinline__ void sk_mma(uint8_t* sm, const Params& p, int dry) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  const int lane = threadIdx.x % 32;
  const uint32_t live = dry ? 0u : 1u;
  tc::tc_fence_after();  // tmem_base was written by tcgen05.alloc (warp 3) before a CTA barrier
  const uint32_t tmem = ms.tmem_base;
  const uint32_t idesc = tc::idesc_mxf8f6f4(128, 8, 5, 0);
  int phase = 0, slot = 0;
  int acc = 0, accph = 0, accuses = 0;
  int cur_u = -1, hord = -1;
  uint32_t xmask = 0;
  long long w_full = 0, w_h = 0, w_acc = 0, n_st = 0;
  const uint64_t adesc0 = tc::desc_sw128(tc::su32(sm + kOffA));
  const uint64_t sfa_src0 = desc_sf(tc::su32(sm + kOffSFA));
  const uint64_t xdesc0 = tc::desc_sw128(tc::su32(sm + kOffXq));
  const uint64_t xsf_src0 = desc_sf(tc::su32(sm + kOffXsf));
  const uint64_t hdesc0 = tc::desc_sw128(tc::su32(sm + kOffHq));
  const uint64_t hsf_src0 = desc_sf(tc::su32(sm + kOffHsf));
  const uint32_t bar_empty0 = tc::su32(&ms.empty[0]), bar_acc0 = tc::su32(&ms.accfull[0]);
  const uint32_t bar_hempty0 = tc::su32(&ms.hempty[0]);
#pragma unroll 1
  for (;;) {
    long long c0 = PCLK();
    if (live) {
      tc::mbar_wait(&ms.full[slot], phase);
      tc::tc_fence_after();
    }
    w_full += PCLK() - c0;
    Meta m;
    if (live) m = ms.meta[slot];
    else m = n_st == 0 ? Meta{kFC1, 0, 0, fFirst | fLast} : n_st == 1 ? Meta{kFC2, 0, 0, 0} : Meta{kEnd, 0, 0, 0};
    ++n_st;
    if (m.type == kEnd) break;
    const bool fc1 = m.type == kFC1;
    const bool first = fc1 ? (m.c & fFirst) != 0 : true;
    const bool last = fc1 ? (m.c & fLast) != 0 : true;
    if (first && accuses >= kAcc && live) {
      c0 = PCLK();
      tc::mbar_wait(&ms.accempty[acc], accph ^ 1);
      tc::tc_fence_after();
      w_acc += PCLK() - c0;
    }
    if (fc1) {
      const int ks = m.b;
      const uint32_t copy_sfb = ((xmask >> ks) & 1) ? 0u : 1u;
      xmask |= 1u << ks;
      mma_fc1(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, xdesc0 + ks * 128, idesc, tmem + kColSFA + slot * 8,
              sfa_src0 + slot * 64, tmem + kColSFBx + 8 * ks, xsf_src0 + ks * 16, copy_sfb, first ? 0u : 1u,
              bar_empty0 + slot * 8, last ? bar_acc0 + acc * 8 : 0u, live);
    } else {
      uint32_t copy_sfb = 0, bar_hrel = 0;
      if (m.a != cur_u) {  // new expert: release the previous h buffer, wait for this one
        if (cur_u >= 0) bar_hrel = bar_hempty0 + (hord & 1) * 8;
        cur_u = m.a;
        ++hord;
        c0 = PCLK();
        if (live) {
          tc::mbar_wait(&ms.hfull[hord & 1], (hord >> 1) & 1);
          tc::tc_fence_after();
        }
        w_h += PCLK() - c0;
        if (hord == 0 && lane == 0 && live) trace_ev(p, 7);
        copy_sfb = 1;
      }
      const int hb = hord & 1;
      mma_fc2(tmem + kColAcc + acc * 8, adesc0 + slot * 2048, hdesc0 + hb * 128, idesc, tmem + kColSFA + slot * 8,
              sfa_src0 + slot * 64, tmem + kColSFBh + hb * 12, hsf_src0 + hb * 48, copy_sfb, bar_hrel,
              bar_empty0 + slot * 8, bar_acc0 + acc * 8, live);
    }
    __syncwarp();
    if (last) {
      if (lane == 0 && live) {
        ms.accmeta[acc] = m;
        tc::mbar_arrive(&ms.accfull[acc]);
      }
      ++accuses;
      if (++acc == kAcc) {
        acc = 0;
        accph ^= 1;
      }
    }
    if (++slot == kSlots) {
      slot = 0;
      phase ^= 1;
    }
  }
  if (!live) return;
  if (lane == 0) {
    trace_ev(p, 8);
    trace_ctr(p, 1, w_full);
    trace_ctr(p, 2, w_acc);
    trace_ctr(p, 3, w_h);
    if (cur_u >= 0) tc::mma_commit(&ms.hempty[hord & 1]);
  }
  const Meta mend{kEnd, 0, 0, 0};
  for (int q = 0; q < 2; ++q) {  // one END per epilogue group
    if (accuses >= kAcc) tc::mbar_wait(&ms.accempty[acc], accph ^ 1);
    if (lane == 0) {
      ms.accmeta[acc] = mend;
      tc::mbar_arrive(&ms.accfull[acc]);
      tc::mbar_arrive(&ms.accfull[acc]);
    }
    ++accuses;
    if (++acc == kAcc) {
      acc = 0;
      accph ^= 1;
    }
  }
}

// ================================================================ stream-K body (M >= 5 by default), device entry
// point. Requirements as group_body, except: no cluster dims; one CTA per SM on (up to) all SMs.
__device__ __forceinline__ void sk_body(uint8_t* sm, const CUtensorMap& tmA1, const CUtensorMap& tmA2, const Params& p) {
  Misc& ms = *reinterpret_cast<Misc*>(sm + kOffMisc);
  SkGeo& geo = *reinterpret_cast<SkGeo*>(ms.skgeo);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int M = p.M;
  const int G = gridDim.x, cta = blockIdx.x;
  uint8_t* ws = reinterpret_cast<uint8_t*>(p.ctr);
  uint8_t* scratch = reinterpret_cast<uint8_t*>(p.scratch);
  float* pbase = reinterpret_cast<float*>(ws + kWsSkP);

  if (threadIdx.x == 0) {
    if ((tc::su32(sm) & 1023) != 0) asm volatile("trap;");
    trace_ev(p, 0);
    for (int s = 0; s < kSlots; ++s) {
      tc::mbar_init(&ms.full[s], 1);
      tc::mbar_init(&ms.empty[s], 1);
    }
    for (int a = 0; a < kAcc; ++a) {
      tc::mbar_init(&ms.accfull[a], 2);
      tc::mbar_init(&ms.accempty[a], 4);
    }
    for (int b = 0; b < 2; ++b) {
      tc::mbar_init(&ms.hfull[b], 1);
      tc::mbar_init(&ms.hempty[b], 1);
    }
    tc::fence_mbar_init();
    tc::prefetch_tmap(&tmA1);
    tc::prefetch_tmap(&tmA2);
    ms.epi_poll[0] = ms.epi_poll[1] = 0;
  }
  if (warp == 3) {
    tc::tmem_alloc(&ms.tmem_base, kTmemCols);
    tc::tc_fence_before();
  }
  {
    int* table = reinterpret_cast<int*>(sm + kOffA);  // dedupe scratch (the ring is still unused)
    for (int e = threadIdx.x; e < p.E; e += kThreads) table[e] = 0x7fffffff;
  }
  for (int i = threadIdx.x; i < (kMaxM - M) * 224; i += kThreads) {
    const int t = M + i / 224, kc = i % 224;
    *reinterpret_cast<uint4*>(sm + kOffXq + (kc / 8) * 1024 + t * 128 + (((kc % 8) ^ t) * 16)) = make_uint4(0, 0, 0, 0);
  }
  for (int i = threadIdx.x; i < 28 * 8; i += kThreads)
    *reinterpret_cast<uint4*>(sm + kOffXsf + (i >> 3) * 128 + (i & 7) * 16) = make_uint4(0, 0, 0, 0);
  body_sync();
  {
#ifndef BODY_NO_WARM
  // ---- warm every post-routing code path (no inputs needed: runs before griddepcontrol.wait)
  const EpiConst ec = make_epi(p, 0);
  uint8_t* hset = scratch;
  if (warp == 0) {
    if (lane == 0) {
#ifndef BODY_NO_DRY
      sk_producer(sm, &tmA1, &tmA2, p, 1);
#endif
      sk_geometry(*reinterpret_cast<SkGeo*>(scratch + 2048), 0, G, cta, 2);
    }
  } else if (warp == 1) {
#ifndef BODY_NO_DRY
    tc::tc_fence_after();
    sk_mma(sm, p, 1);
#endif
  } else if (warp == 2) {
    sk_h_load(sm, scratch, 1, lane, 1);
  } else if (warp >= 4) {
    tc::tc_fence_after();
    const int ew = warp & 3, eg = (warp - 4) >> 2;
    const uint32_t tl = ms.tmem_base + (static_cast<uint32_t>(32 * ew) << 16);
    if (threadIdx.x == 128) sk_geometry(geo, 3, G, cta, 1);  // placeholder geometry for the warm-up
    asm volatile("bar.sync 5, 256;" ::: "memory");
    sk_epi_item(sm, tl, 0u, kFC1, 0, 0, 0, eg, ew, lane, M, cta, pbase, hset, scratch, 1, ec);  // non-owner
    sk_epi_item(sm, tl, 0u, kFC1, 0, 0, 0, eg, ew, lane, M, cta, pbase, hset, scratch, 2, ec);  // owner
    sk_epi_item(sm, tl, 0u, kFC2, 0, 0, 0, eg, ew, lane, M, cta, pbase, hset, scratch, 1, ec);
  }
  if (warp < 4) dedupe(sm, -1, 0);
#endif
  }

  const int npairs = M * kTopK;
  // Routing source: (a) p.ids in global memory, after griddepcontrol.wait; (b) flag mode: x is ready, the
  // routing arrives through the epoch counter (waiting for the predecessor grid would serialize with it);
  // (c) p.ids_smem: ids written into shared memory by the caller's own warps, signalled on p.ids_bar.
  const bool flagmode = p.flag != nullptr;
  const bool smemids = p.ids_smem != nullptr;
  if (!flagmode && !smemids && !p.no_griddep) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) trace_ev(p, 1);
  if (threadIdx.x == 0) ms.call = ~p.ctr[cta];  // after griddepcontrol.wait (see group_body)
  int myid = -1;
  if (!flagmode && !smemids && threadIdx.x < npairs) myid = p.ids[threadIdx.x];
  const bool xlate = flagmode && p.x_late;
  if (!xlate)
    for (int it = threadIdx.x; it < M * 112; it += kThreads) quant_x_block(sm, p.x, it / 112, it % 112);
  tc::fence_proxy_async_smem();
  body_sync();
  const int call = ms.call;
  const int P = call & 1;
  uint8_t* hset = ws + kWsSkH + static_cast<long>(P) * kMaxPairs * kHImg;

  const EpiConst ec = make_epi(p, call);
  if (flagmode) {
    body_sync();
    if (threadIdx.x == 0) {
      trace_ev(p, 16);
      const unsigned long long target = static_cast<unsigned long long>(call) + 1ull;
      while (ld_acquire_u64(p.flag) < target) {
      }
    }
    body_sync();
    if (threadIdx.x < npairs) myid = ld_relaxed_i32(p.ids + threadIdx.x);
    if (xlate) {  // x published with the ids (acquire by thread 0's ld.acquire + the CTA barrier above)
      for (int it = threadIdx.x; it < M * 112; it += kThreads) quant_x_block(sm, p.x, it / 112, it % 112);
      tc::fence_proxy_async_smem();
    }
  } else if (smemids) {
    if (threadIdx.x == 0) trace_ev(p, 16);
    if (threadIdx.x < npairs) {
      tc::mbar_wait(reinterpret_cast<uint64_t*>(__cvta_shared_to_generic(p.ids_bar)), p.ids_phase);
      myid = p.ids_smem[threadIdx.x];
    }
  }
  if (threadIdx.x == 0) trace_ev(p, 2);
  if (warp < 4) {
    dedupe(sm, myid, npairs);
    if (threadIdx.x == 0) {
      const int D = ms.D;
      const int ng = p.sk_groups > 0 ? p.sk_groups : (D > 40 ? 2 : 1);
      sk_geometry(geo, D, G, cta, ng);
    }
  }
  body_sync();
  asm volatile("griddepcontrol.launch_dependents;");
  if (threadIdx.x == 0) trace_ev(p, 3);

  if (warp == 0) {
    if (lane == 0) sk_producer(sm, &tmA1, &tmA2, p, 0);
  } else if (warp == 1) {
    sk_mma(sm, p, 0);
  } else if (warp == 2) {
    // ================================================================ h loader
    int k = -1;
    for (int g = 0; g < geo.ng; ++g) {
      if (geo.f2hi[g] <= geo.f2lo[g]) continue;
      const int u0 = geo.f2lo[g] / kFc2Tiles, u1 = (geo.f2hi[g] - 1) / kFc2Tiles;
      for (int u = u0; u <= u1; ++u) {
        ++k;
        const int b = k & 1;
        if (k >= 2) tc::mbar_wait(&ms.hempty[b], ((k >> 1) - 1) & 1);
        sk_h_load(sm, hset + static_cast<long>(u) * kHImg, b, lane, 0);
      }
    }
  } else if (warp == 3) {
    // ================================================================ re-arm the idle h parity
    uint4* other = reinterpret_cast<uint4*>(ws + kWsSkH + static_cast<long>(P ^ 1) * kMaxPairs * kHImg);
    const int n16 = kMaxPairs * kHImg / 16;
    const int lo = split_lo(cta, n16, G), hi = split_lo(cta + 1, n16, G);
    const uint4 ff = make_uint4(~0u, ~0u, ~0u, ~0u);
    for (int i = lo + lane; i < hi; i += 32) other[i] = ff;
  } else {
    // ================================================================ epilogue (two groups of 4 warps)
    tc::tc_fence_after();
    const uint32_t tmem = ms.tmem_base;
    const int ew = warp & 3, eg = (warp - 4) >> 2;
    const uint32_t tl = tmem + (static_cast<uint32_t>(32 * ew) << 16);
    int acc = eg, accph = 0;
    for (;;) {
      tc::mbar_wait(&ms.accfull[acc], accph);
      tc::tc_fence_after();
      const Meta m = ms.accmeta[acc];
      if (m.type == kEnd) break;
      sk_epi_item(sm, tl + kColAcc + acc * 8, tc::su32(&ms.accempty[acc]), m.type, m.a, m.b, m.c, eg, ew, lane, M, cta,
                  pbase, hset, scratch, 0, ec);
      if ((acc += 2) >= kAcc) {
        acc -= kAcc;
        accph ^= 1;
      }
    }
    if (eg == 0 && ew == 0 && lane == 0) {
      trace_ev(p, 9);
      trace_ctr(p, 7, ms.epi_poll[0]);
      trace_ctr(p, 8, ms.epi_poll[1]);
    }
  }
  tc::tc_fence_before();
  body_sync();
  if (warp == 3) {
    tc::tc_fence_after();
    tc::tmem_dealloc(ms.tmem_base, kTmemCols);
  }
  if (flagmode) asm volatile("griddepcontrol.wait;" ::: "memory");  // see group_body
  if (threadIdx.x == 0) {
    if (p.endlog != nullptr)
      atomicMax(p.endlog + (call % p.endlog_n), static_cast<unsigned long long>(tc::globaltimer()));
    bump_counters(p.ctr, cta, G, call);
    trace_ev(p, 10);
  }
}
