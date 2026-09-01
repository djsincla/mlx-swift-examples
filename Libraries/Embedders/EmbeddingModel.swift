// Copyright © 2024 Apple Inc.
//
// Upstream: ml-explore/mlx-swift-examples, MIT (c) 2024 ml-explore.
// Modified for dAI in 2026: `EmbeddingModelOutput.hiddenStates` and
// `pooledOutput` are public rather than internal, so pooling can be done
// outside this library. The change is commented where it is made.

import Foundation
@preconcurrency import Hub
import MLX
import MLXNN
import Tokenizers

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer:
///
/// ```swift
/// let promptTokens = await modelContainer.perform { _, tokenizer in
///     tokenizer.encode(text: prompt)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let result = await modelContainer.perform { model, tokenizer in
///     LLM.generate(
///         promptTokens: promptTokens, parameters: generateParameters, model: model,
///         tokenizer: tokenizer, extraEOSTokens: modelConfiguration.extraEOSTokens
///     ) { tokens in
///     ...
///     }
/// }
/// ```
public actor ModelContainer {
    let model: EmbeddingModel
    let tokenizer: Tokenizer
    let pooler: Pooling

    public init(
        model: EmbeddingModel, tokenizer: Tokenizer, pooler: Pooling = Pooling(strategy: .none)
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.pooler = pooler
    }

    /// build the model and tokenizer without passing non-sendable data over isolation barriers
    public init(
        hub: HubApi, modelDirectory: URL, configuration: ModelConfiguration
    ) async throws {
        self.model = try loadSynchronous(modelDirectory: modelDirectory)

        let (tokenizerConfig, tokenizerData) = try await loadTokenizerConfig(
            configuration: configuration, hub: hub)
        self.tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        self.pooler = loadPooling(modelDirectory: modelDirectory)  //?? Pooling(strategy: .none)
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<R>(_ action: @Sendable (EmbeddingModel, Tokenizer, Pooling) throws -> R)
        rethrows
        -> R
    {
        try action(model, tokenizer, pooler)
    }
}

extension Module {

    /// Compute the number of parameters in a possibly quantized model
    public func numParameters() -> Int {
        return leafModules().flattenedValues().map {
            mod -> Int in
            if let qlin = mod as? QuantizedLinear {
                return qlin.scales.size * qlin.groupSize
            } else if let qemb = mod as? QuantizedEmbedding {
                return qemb.scales.size * qemb.groupSize
            } else {
                return mod.parameters().flattenedValues().reduce(
                    0,
                    {
                        $0 + $1.size
                    })
            }
        }.reduce(0, +)
    }
}

public struct EmbeddingModelOutput {
    // Public rather than internal, which is a deliberate delta from upstream.
    //
    // Pooling has to happen outside this library for us. Its loadPooling reads
    // 1_Pooling/config.json and silently falls back to Strategy.none when the
    // file is absent, and the mlx-community conversions we stage do not ship
    // that directory: the result is unpooled hidden states presented as an
    // embedding. Its .last strategy is also padding-unaware, taking the final
    // position of a padded sequence, so a short input batched with a long one
    // would pool from padding.
    //
    // DaiWorker/EmbedRuntime pools per row from each row's real token count,
    // which needs these. See docs/EMBEDDINGS_PLAN.md.
    public let hiddenStates: MLXArray?
    public let pooledOutput: MLXArray?
}

public protocol EmbeddingModel: Module {
    var vocabularySize: Int { get }
    func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?,
        attentionMask: MLXArray?
    ) -> EmbeddingModelOutput
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

extension EmbeddingModel {
    func callAsFunction(
        _ inputs: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> EmbeddingModelOutput {
        return callAsFunction(
            inputs, positionIds: positionIds, tokenTypeIds: tokenTypeIds,
            attentionMask: attentionMask)
    }
}
