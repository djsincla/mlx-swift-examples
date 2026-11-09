// dAI: Next-Concept Modeling (NCM) inference. Not part of upstream mlx-swift-examples.
//
// Mirrors the config.json written by the converter in the NCM repository
// (local/mlx/convert.py), which flattens the PyTorch config and the values PyTorch derives
// while building its modules, so nothing here needs PyTorch to interpret.

import Foundation

public struct NCMConfiguration: Codable, Sendable {
    public struct Generator: Codable, Sendable {
        public var nLayer: Int
        public var nHead: Int
        public var nEmbd: Int
        public var prefixLen: Int
        public var ropeBase: Double
        public var useQKNorm: Bool
        public var attnPattern: String
        public var queryLevel0: Bool
        public var predictLevel0: Bool
        public var prefixPadTokenId: Int?
        public var postUpsampleConv: Bool
        /// "tied" (dot product with the predicted level's codebook), "shared" or "linear".
        public var head: String
        public var tiedHeadBias: Bool
        public var sharedHeadPerLevelBias: Bool

        enum CodingKeys: String, CodingKey {
            case nLayer = "n_layer", nHead = "n_head", nEmbd = "n_embd", prefixLen = "prefix_len"
            case ropeBase = "rope_base", useQKNorm = "use_qk_norm", attnPattern = "attn_pattern"
            case queryLevel0 = "query_level0", predictLevel0 = "predict_level0"
            case prefixPadTokenId = "prefix_pad_token_id", postUpsampleConv = "post_upsample_conv"
            case head, tiedHeadBias = "tied_head_bias"
            case sharedHeadPerLevelBias = "shared_head_per_level_bias"
        }
    }

    /// One stage of the learned level-0 upsampler: a ConvTranspose1d with kernel == stride,
    /// optionally preceded by an affine-free GroupNorm.
    public struct UpsampleStage: Codable, Sendable {
        public var prefix: String
        public var stride: Int
        public var groupnormBefore: Bool

        enum CodingKeys: String, CodingKey {
            case prefix, stride, groupnormBefore = "groupnorm_before"
        }
    }

    public struct Quantizer: Codable, Sendable {
        public var learnedUpL1: [UpsampleStage]?
        public var quantResiRatio: Float?
        /// Which quant_resi convolution each level uses.
        public var quantResiIndex: [Int]?
        public var groupnormGroups: Int
        public var groupnormEps: Float

        enum CodingKeys: String, CodingKey {
            case learnedUpL1 = "learned_up_l1", quantResiRatio = "quant_resi_ratio"
            case quantResiIndex = "quant_resi_index", groupnormGroups = "groupnorm_groups"
            case groupnormEps = "groupnorm_eps"
        }
    }

    public struct Decoder: Codable, Sendable {
        public var nHead: Int
        public var nBlocks: Int
        public var positional: Bool
        public var useQKNorm: Bool

        enum CodingKeys: String, CodingKey {
            case nHead = "n_head", nBlocks = "n_blocks", positional, useQKNorm = "use_qk_norm"
        }
    }

    public var format: String
    public var task: String
    /// Codes per level, coarse to fine. The last level has one code per output token.
    public var levels: [Int]
    public var vqVocabSizes: [Int]
    public var embedDim: Int
    public var inVocabSize: Int
    public var generator: Generator
    public var vq: Quantizer
    public var decoder: Decoder

    enum CodingKeys: String, CodingKey {
        case format, task, levels, vqVocabSizes = "vq_vocab_sizes", embedDim = "embed_dim"
        case inVocabSize = "in_vocab_size", generator, vq, decoder
    }

    public static func load(from directory: URL) throws -> NCMConfiguration {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        let config = try JSONDecoder().decode(NCMConfiguration.self, from: data)
        guard config.format == "ncm-mlx/1" else {
            throw NCMError.unsupportedFormat(config.format)
        }
        return config
    }
}

public enum NCMError: Error, CustomStringConvertible {
    case unsupportedFormat(String)
    case missingWeight(String)
    case needsLevel0Codes
    case needsPadToken

    public var description: String {
        switch self {
        case .unsupportedFormat(let f): "unsupported NCM checkpoint format \(f); expected ncm-mlx/1"
        case .missingWeight(let k): "checkpoint has no tensor named \(k)"
        case .needsLevel0Codes: "this model predicts no level 0 itself; pass level0 codes"
        case .needsPadToken: "classifier-free guidance needs prefix_pad_token_id"
        }
    }
}
