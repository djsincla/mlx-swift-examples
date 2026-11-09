// dAI: Next-Concept Modeling (NCM) inference. Not part of upstream mlx-swift-examples.
//
// NCM (Kumbong et al., arXiv 2610.14162, github.com/HazyResearch/ncm) generates text coarse to
// fine: a multiscale residual VQ-VAE represents a sequence as levels of discrete codes, and a
// transformer predicts every code of a level at once, one forward pass per level.
//
// This is a port of the generation path of the official PyTorch code (models/next_concept.py,
// models/quant.py, models/vae.py), through the MLX Python port in the NCM repository
// (local/mlx/ncm_mlx.py), which it follows function for function. Checkpoints are converted by
// local/mlx/convert.py into model.safetensors + config.json.
//
// Activations are [B, L, C] throughout; PyTorch's [B, C, L] conv layouts are handled inside the
// conv helpers.

import Foundation
import MLX
import MLXRandom

// MARK: - Resampling as fixed matrices

/// `F.interpolate(mode: "area")`, i.e. adaptive average pooling: output i averages input
/// positions [floor(i*n/m), ceil((i+1)*n/m)).
func areaMatrix(from nIn: Int, to nOut: Int) -> [Float] {
    var w = [Double](repeating: 0, count: nOut * nIn)
    for i in 0 ..< nOut {
        let s = (i * nIn) / nOut
        let e = ((i + 1) * nIn + nOut - 1) / nOut
        for j in s ..< e { w[i * nIn + j] = 1.0 / Double(e - s) }
    }
    return w.map { Float($0) }
}

/// `F.interpolate(mode: "linear", align_corners: false)` with an explicit output size.
func linearMatrix(from nIn: Int, to nOut: Int) -> [Float] {
    var w = [Double](repeating: 0, count: nOut * nIn)
    let scale = Double(nIn) / Double(nOut)
    for i in 0 ..< nOut {
        let src = max(scale * (Double(i) + 0.5) - 0.5, 0)
        let i0 = min(Int(src.rounded(.down)), nIn - 1)
        let i1 = min(i0 + 1, nIn - 1)
        let lambda = src - Double(i0)
        w[i * nIn + i0] += 1 - lambda
        w[i * nIn + i1] += lambda
    }
    return w.map { Float($0) }
}

// MARK: - Layers

private let negInf = -Float.infinity

private func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray? = nil) -> MLXArray {
    let y = matmul(x, w.T)
    return b.map { y + $0 } ?? y
}

/// Exact (erf) GELU, as `torch.nn.GELU()`.
private func gelu(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + erf(x / Float(2).squareRoot()))
}

/// PyTorch Conv1d(kernel k, padding k/2) on [B, L, C]; `w` is [C_out, C_in, k].
private func conv1dSame(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    conv1d(x, w.transposed(0, 2, 1), padding: w.dim(-1) / 2) + b
}

/// PyTorch ConvTranspose1d with kernel == stride on [B, L, C]; `w` is [C_in, C_out, s].
private func convTransposeStride(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))
    let (cIn, cOut, s) = (w.dim(0), w.dim(1), w.dim(2))
    let y = matmul(x, w.reshaped([cIn, cOut * s])).reshaped([B, L, cOut, s])
    return y.transposed(0, 1, 3, 2).reshaped([B, L * s, cOut]) + b
}

/// Affine-free GroupNorm on [B, L, C], statistics over (C / groups, L).
private func groupNorm(_ x: MLXArray, groups: Int, eps: Float) -> MLXArray {
    let (B, L, C) = (x.dim(0), x.dim(1), x.dim(2))
    let g = x.reshaped([B, L, groups, C / groups])
    let mean = g.mean(axes: [1, 3], keepDims: true)
    let variance = ((g - mean) * (g - mean)).mean(axes: [1, 3], keepDims: true)
    return ((g - mean) * rsqrt(variance + eps)).reshaped([B, L, C])
}

/// Rotate-half RoPE tables for explicit positions (each NCM level restarts at 0).
private func ropeTables(positions: [Int], headDim: Int, base: Double) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    var cos = [Float](), sin = [Float]()
    cos.reserveCapacity(positions.count * headDim)
    sin.reserveCapacity(positions.count * headDim)
    let inv = (0 ..< half).map { Float(1.0 / pow(base, Double(Float($0) / Float(half)))) }
    for p in positions {
        let f = inv.map { Float(p) * $0 }
        let c = f.map { Foundation.cos($0) }, s = f.map { Foundation.sin($0) }
        cos += c + c
        sin += s + s
    }
    return (MLXArray(cos, [positions.count, headDim]), MLXArray(sin, [positions.count, headDim]))
}

private func applyRope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    return x * cos + rotated * sin
}

// MARK: - Model

/// A converted NCM checkpoint, ready to generate.
public final class NCMModel: @unchecked Sendable {
    public let configuration: NCMConfiguration
    public var levels: [Int] { configuration.levels }
    public var prefixLength: Int { configuration.generator.prefixLen }

    let weights: [String: MLXArray]
    private let lvlFirst: Int
    private let ropeGen: [(MLXArray, MLXArray)]
    private let levelIds: MLXArray
    private let tied: [(e: MLXArray, eSq: MLXArray, dScale: MLXArray)]
    private var resampleCache: [String: MLXArray] = [:]
    private let lock = NSLock()

    /// Loads `config.json` and `model.safetensors` from a directory written by the converter.
    public init(directory: URL) throws {
        let config = try NCMConfiguration.load(from: directory)
        self.configuration = config
        self.weights = try loadArrays(url: directory.appending(path: "model.safetensors"))
        let g = config.generator
        lvlFirst = g.queryLevel0 ? 0 : 1
        let targetLevels = Array(config.levels[lvlFirst...])
        let headDim = g.nEmbd / g.nHead
        ropeGen = targetLevels.map {
            ropeTables(positions: Array(0 ..< g.prefixLen) + Array(0 ..< $0), headDim: headDim, base: g.ropeBase)
        }
        levelIds = MLXArray(targetLevels.enumerated().flatMap { i, l in [Int32](repeating: Int32(i), count: l) })
        var tied: [(MLXArray, MLXArray, MLXArray)] = []
        if g.head == "tied" {
            for i in 0 ..< targetLevels.count {
                guard let e = weights["tok.vq.embeddings.\(i + lvlFirst).weight"] else {
                    throw NCMError.missingWeight("tok.vq.embeddings.\(i + lvlFirst).weight")
                }
                let eSq = (e * e).sum(axis: -1)
                tied.append((e, eSq, maximum(eSq.mean(), MLXArray(Float(1e-12)))))
            }
        }
        self.tied = tied.map { (e: $0.0, eSq: $0.1, dScale: $0.2) }
        eval(Array(weights.values))
    }

    func w(_ key: String) -> MLXArray {
        guard let a = weights[key] else { fatalError(NCMError.missingWeight(key).description) }
        return a
    }

    private func resample(_ x: MLXArray, to nOut: Int, linear isLinear: Bool) -> MLXArray {
        let nIn = x.dim(1)
        if nIn == nOut { return x }
        let key = "\(isLinear ? "l" : "a")\(nIn)-\(nOut)"
        lock.lock()
        defer { lock.unlock() }
        let m: MLXArray
        if let cached = resampleCache[key] {
            m = cached
        } else {
            m = MLXArray(isLinear ? linearMatrix(from: nIn, to: nOut) : areaMatrix(from: nIn, to: nOut), [nOut, nIn])
            resampleCache[key] = m
        }
        return matmul(m, x)
    }

    // MARK: Transformer blocks

    private func attention(
        _ p: String, _ x: MLXArray, nHead: Int, qkNorm: Bool, mask: MLXArray?,
        rope: (MLXArray, MLXArray)?, pastKV: (MLXArray, MLXArray)? = nil,
        kvOut: UnsafeMutablePointer<[(MLXArray, MLXArray)]>? = nil
    ) -> MLXArray {
        let (B, T, C) = (x.dim(0), x.dim(1), x.dim(2))
        let hd = C / nHead
        let qkv = linear(x, w("\(p).c_attn.weight"), weights["\(p).c_attn.bias"])
        let parts = qkv.split(indices: [C, 2 * C], axis: -1)
        var q = parts[0].reshaped([B, T, nHead, hd]).transposed(0, 2, 1, 3)
        var k = parts[1].reshaped([B, T, nHead, hd]).transposed(0, 2, 1, 3)
        var v = parts[2].reshaped([B, T, nHead, hd]).transposed(0, 2, 1, 3)
        if qkNorm {
            q = MLXFast.rmsNorm(q, weight: w("\(p).q_norm.weight"), eps: 1e-6)
            k = MLXFast.rmsNorm(k, weight: w("\(p).k_norm.weight"), eps: 1e-6)
        }
        if let (cos, sin) = rope {
            let c = cos[..<T], s = sin[..<T]
            q = applyRope(q, c, s)
            k = applyRope(k, c, s)
        }
        kvOut?.pointee.append((k, v))
        if let (pk, pv) = pastKV {
            k = concatenated([pk, k], axis: 2)
            v = concatenated([pv, v], axis: 2)
        }
        let y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(hd).squareRoot(), mask: mask)
        return linear(y.transposed(0, 2, 1, 3).reshaped([B, T, C]), w("\(p).c_proj.weight"), weights["\(p).c_proj.bias"])
    }

    private func block(
        _ p: String, _ x: MLXArray, nHead: Int, qkNorm: Bool, mask: MLXArray?,
        rope: (MLXArray, MLXArray)?, pastKV: (MLXArray, MLXArray)? = nil,
        kvOut: UnsafeMutablePointer<[(MLXArray, MLXArray)]>? = nil
    ) -> MLXArray {
        let n1 = MLXFast.layerNorm(x, weight: w("\(p).ln_1.weight"), bias: weights["\(p).ln_1.bias"], eps: 1e-6)
        let x = x + attention("\(p).attn", n1, nHead: nHead, qkNorm: qkNorm, mask: mask, rope: rope,
                              pastKV: pastKV, kvOut: kvOut)
        let n2 = MLXFast.layerNorm(x, weight: w("\(p).ln_2.weight"), bias: weights["\(p).ln_2.bias"], eps: 1e-6)
        let h = linear(gelu(linear(n2, w("\(p).mlp.c_fc.weight"), weights["\(p).mlp.c_fc.bias"])),
                       w("\(p).mlp.c_proj.weight"), weights["\(p).mlp.c_proj.bias"])
        return x + h
    }

    // MARK: Quantizer

    private func upToFull(_ q: MLXArray, level: Int) -> MLXArray {
        let vq = configuration.vq
        if level == 0, let stages = vq.learnedUpL1, !stages.isEmpty {
            var q = q
            for st in stages {
                if st.groupnormBefore { q = groupNorm(q, groups: vq.groupnormGroups, eps: vq.groupnormEps) }
                q = convTransposeStride(q, w("\(st.prefix).weight"), w("\(st.prefix).bias"))
            }
            return q
        }
        return resample(q, to: levels.last!, linear: true)
    }

    private func quantResi(_ h: MLXArray, level: Int) -> MLXArray {
        guard let r = configuration.vq.quantResiRatio, let idx = configuration.vq.quantResiIndex else { return h }
        let p = "tok.vq.quant_resi.\(idx[level])"
        return h * (1 - r) + conv1dSame(h, w("\(p).weight"), w("\(p).bias")) * r
    }

    private func codeVectors(level: Int, _ idx: MLXArray) -> MLXArray {
        take(w("tok.vq.embeddings.\(level).weight"), idx, axis: 0)
    }

    /// The decoder input for a full set of codes: every level's code vectors, upsampled to
    /// full length and summed.
    public func zHat(codes: MLXArray) -> MLXArray {
        var z: MLXArray? = nil
        var off = 0
        for (i, l) in levels.enumerated() {
            var e = codeVectors(level: i, codes[0..., off ..< off + l])
            if i != levels.count - 1 { e = upToFull(e, level: i) }
            e = quantResi(e, level: i)
            z = z.map { $0 + e } ?? e
            off += l
        }
        return z!
    }

    /// Token logits [B, L, vocab] from a decoder input.
    public func decodeLogits(_ z: MLXArray) -> MLXArray {
        let d = configuration.decoder
        var z = z
        if d.positional { z = z + w("tok.decoder.wpe.weight")[..<z.dim(1)] }
        let names = ["tok.decoder.middle_block"] + (0 ..< max(0, d.nBlocks - 1)).map { "tok.decoder.extra_blocks.\($0)" }
        for p in names { z = block(p, z, nHead: d.nHead, qkNorm: d.useQKNorm, mask: nil, rope: nil) }
        z = MLXFast.layerNorm(z, weight: w("tok.decoder.ln_f.weight"), bias: weights["tok.decoder.ln_f.bias"], eps: 1e-6)
        return linear(z, w("tok.decoder.output_layer.weight"))
    }

    // MARK: Generator

    private func wte(_ z: MLXArray) -> MLXArray {
        linear(z, w("gen.transformer.wte.weight"), weights["gen.transformer.wte.bias"])
    }

    private func wle(_ lo: Int, _ hi: Int) -> MLXArray {
        take(w("gen.transformer.wle.weight"), levelIds[lo ..< hi], axis: 0).expandedDimensions(axis: 0)
    }

    private func postConv(_ x: MLXArray, _ k: Int) -> MLXArray {
        guard configuration.generator.postUpsampleConv else { return x }
        let p = "gen.post_upsample_convs.\(k)"
        return x + conv1dSame(x, w("\(p).weight"), w("\(p).bias"))
    }

    private func stack(
        _ h: MLXArray, mask: MLXArray?, rope: (MLXArray, MLXArray), pastKV: [(MLXArray, MLXArray)]? = nil,
        kvOut: UnsafeMutablePointer<[(MLXArray, MLXArray)]>? = nil
    ) -> MLXArray {
        let g = configuration.generator
        var h = h
        for i in 0 ..< g.nLayer {
            h = block("gen.transformer.blocks.\(i)", h, nHead: g.nHead, qkNorm: g.useQKNorm, mask: mask,
                      rope: rope, pastKV: pastKV?[i], kvOut: kvOut)
        }
        return MLXFast.layerNorm(h, weight: w("gen.transformer.ln_f.weight"), bias: weights["gen.transformer.ln_f.bias"], eps: 1e-6)
    }

    private func head(_ i: Int, _ x: MLXArray) -> MLXArray {
        let g = configuration.generator
        switch g.head {
        case "tied":
            let t = tied[i]
            let z = linear(x, w("gen.lm_heads.\(i).proj.weight"))
            var out = (2 * matmul(z, t.e.T) - t.eSq) / t.dScale
            if let s = weights["gen.lm_heads.\(i).log_inv_T"] { out = out * exp(s) }
            if g.tiedHeadBias { out = out + w("gen.lm_heads.\(i).bias") }
            return out
        case "shared":
            let out = linear(x, w("gen.lm_heads.head.weight"), weights["gen.lm_heads.head.bias"])
            return g.sharedHeadPerLevelBias ? out + w("gen.lm_heads.biases.\(i)") : out
        default:
            return linear(x, w("gen.lm_heads.\(i).weight"), weights["gen.lm_heads.\(i).bias"])
        }
    }

    /// Greedy decoding when temperature is 0, top-k is 1 or top-p is 0; otherwise sampling,
    /// with the same top-k / top-p rules as the PyTorch code.
    static func sample(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float) -> MLXArray {
        if temperature == 0 || topK == 1 || topP == 0 { return argMax(logits, axis: -1) }
        var lg = logits / temperature
        if topK > 0 {
            let kth = sorted(lg, axis: -1)[.ellipsis, (-topK) ..< (-topK + 1)]
            lg = which(lg .< kth, MLXArray(negInf), lg)
        }
        if topP < 1 {
            let order = argSort(-lg, axis: -1)
            var sl = takeAlong(lg, order, axis: -1)
            let p = softmax(sl, axis: -1)
            sl = which((cumsum(p, axis: -1) - p) .> topP, MLXArray(negInf), sl)
            lg = takeAlong(sl, argSort(order, axis: -1), axis: -1)
        }
        return categorical(lg, axis: -1)
    }

    /// Generates codes for every level.
    ///
    /// - Parameters:
    ///   - prefix: [B, P] token ids of the conditioning prefix (padded with the model's
    ///     prefix pad token). Unconditional models use a single end-of-text token, P = 1.
    ///   - level0: [B, levels[0]] level-0 codes, for models that do not predict level 0
    ///     themselves (sampled from the checkpoint's `inference.level0_unigram`).
    ///   - onLevel: called with the running decoder input after each level, coarse to fine.
    /// - Returns: [B, sum(levels)] codes, level by level.
    public func generate(
        prefix: MLXArray, level0: MLXArray? = nil, temperature: Float = 0, topK: Int = 0, topP: Float = 1,
        cfgScale: Float = 0, cfgNullToken: Int? = nil, onLevel: ((Int, MLXArray) -> Void)? = nil
    ) throws -> MLXArray {
        let g = configuration.generator
        let P = g.prefixLen, nlev = levels.count
        var prefix = prefix, level0 = level0
        var cfgB: Int? = nil
        if cfgScale > 0 {
            guard let pad = g.prefixPadTokenId else { throw NCMError.needsPadToken }
            cfgB = prefix.dim(0)
            let null = MLXArray.full(prefix.shape, values: MLXArray(Int32(pad)))
            null[0..., 0] = MLXArray(Int32(cfgNullToken ?? pad))
            prefix = concatenated([prefix, null], axis: 0)
            if let l0 = level0 { level0 = concatenated([l0, l0], axis: 0) }
        }
        func pick(_ logits: MLXArray, _ level: Int) -> MLXArray {
            guard let b = cfgB else { return Self.sample(logits, temperature: temperature, topK: topK, topP: topP) }
            let t = cfgScale * Float(level) / Float(max(nlev - 1, 1))
            let mixed = (1 + t) * logits[..<b] - t * logits[b...]
            let pred = Self.sample(mixed, temperature: temperature, topK: topK, topP: topP)
            return concatenated([pred, pred], axis: 0)
        }

        let B = prefix.dim(0)
        let prefixEmb = wte(take(w("tok.encoder.wte.weight"), prefix[0..., ..<P], axis: 0))

        var padBias: MLXArray? = nil
        var padMask: MLXArray? = nil
        if let pad = g.prefixPadTokenId {
            let m = prefix[0..., ..<P] .== MLXArray(Int32(pad))
            padMask = m
            padBias = which(m, MLXArray(negInf), MLXArray(Float(0)))
                .reshaped([B, 1, 1, P]).asType(.float32)
        }

        // The prefix attends only to itself at every level, so its per-layer keys and values
        // are computed once and reused: the same values a full pass would recompute.
        var kv: [(MLXArray, MLXArray)] = []
        var prefixMask = MLXArray.zeros([1, 1, P, P])
        if let pb = padBias { prefixMask = prefixMask + pb }
        let hPrefix = withUnsafeMutablePointer(to: &kv) { ptr in
            stack(prefixEmb, mask: prefixMask, rope: (ropeGen[0].0[..<P], ropeGen[0].1[..<P]), kvOut: ptr)
        }
        eval([hPrefix] + kv.flatMap { [$0.0, $0.1] })

        func levelHidden(_ x: MLXArray, _ hi: Int) -> MLXArray {
            let n = x.dim(1)
            var mask = MLXArray.zeros([1, 1, n, P + n])
            if let pb = padBias { mask = mask + padded(pb, widths: [0, 0, 0, .init((0, n))]) }
            let (cos, sin) = ropeGen[hi]
            return stack(x, mask: mask, rope: (cos[P ..< P + n], sin[P ..< P + n]), pastKV: kv)
        }

        var l0: MLXArray
        if let given = level0 {
            l0 = given
        } else if g.queryLevel0 {
            let n0 = levels[0]
            var q = broadcast(w("gen.l0_queries"), to: [B, n0, configuration.embedDim])
            q = wte(postConv(q, 0)) + wle(0, n0)
            l0 = pick(head(0, levelHidden(q, 0)), 0)
        } else if g.predictLevel0 {
            let n0 = levels[0]
            let hs: MLXArray
            if let pm = padMask {
                let nonPad = (1 - pm.asType(.int32)).sum(axis: 1) - 1
                let last = maximum(nonPad, MLXArray(Int32(n0 - 1)))
                let offs = MLXArray((0 ..< n0).reversed().map { Int32($0) })
                let pos = clip(last.expandedDimensions(axis: 1) - offs.expandedDimensions(axis: 0), min: 0, max: P - 1)
                hs = takeAlong(hPrefix, pos.expandedDimensions(axis: 2), axis: 1)
            } else {
                hs = hPrefix[0..., (P - n0) ..< P]
            }
            l0 = pick(linear(hs, w("gen.lm_head_l0.weight"), weights["gen.lm_head_l0.bias"]), 0)
        } else {
            throw NCMError.needsLevel0Codes
        }

        var out = [l0]
        var next = l0
        var z = MLXArray.zeros([B, levels.last!, configuration.embedDim])
        var acc = g.queryLevel0 ? levels[0] : 0
        for li in 0 ..< nlev - 1 {
            z = z + quantResi(upToFull(codeVectors(level: li, next), level: li), level: li)
            if let cb = onLevel { cb(li, cfgB.map { z[..<$0] } ?? z) }
            let n = levels[li + 1]
            let hi = li + (1 - lvlFirst)
            var x = resample(z, to: n, linear: false)
            x = wte(postConv(x, hi)) + wle(acc, acc + n)
            next = pick(head(hi, levelHidden(x, hi)), li + 1)
            eval(next)
            out.append(next)
            acc += n
        }
        if let cb = onLevel {
            let final = z + quantResi(codeVectors(level: nlev - 1, next), level: nlev - 1)
            cb(nlev - 1, cfgB.map { final[..<$0] } ?? final)
        }
        let codes = concatenated(out, axis: 1)
        return cfgB.map { codes[..<$0] } ?? codes
    }
}
