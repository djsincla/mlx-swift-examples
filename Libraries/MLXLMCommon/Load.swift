// Copyright © 2024 Apple Inc.
//
// Upstream: ml-explore/mlx-swift-examples, MIT (c) 2024 ml-explore.
// Modified for dAI in 2026: `loadWeights(..., keepingLayers:)` drops the layers
// this machine does not own and renumbers the rest. Changes are marked `dAI:`.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id, let revision):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json"]
            return try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: progressHandler
            )
        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    // dAI: the slice of layers this machine holds, when a model is split across
    // several. Nil means the whole model, which is every ordinary case.
    keepingLayers: Range<Int>? = nil
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // dAI: drop the layers this machine does not own, and renumber the rest.
    //
    // This is where the memory saving actually happens. The weights are only
    // materialised when something references them, so a layer removed here is
    // never read off disk: that is how a 41GB model becomes 21GB on each of two
    // machines rather than needing 41GB somewhere first.
    //
    // Renumbering is necessary because the layer array was shortened rather
    // than padded, so what was layer 40 is now layer 0. Loading unrenumbered
    // weights would either fail verification or, worse, quietly put layer 40's
    // weights into layer 0's slot.
    if let keep = keepingLayers, keep.lowerBound > 0 || keep.count > 0 {
        weights = renumberLayers(weights, keeping: keep)
    }

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}

/// dAI: keep only the weights for a range of layers, renumbered from zero.
///
/// Keys look like `model.layers.42.self_attn.q_proj.weight`. A machine holding
/// layers 40 to 79 has them as its own layers 0 to 39, so the index has to move
/// with them. Anything that is not a layer - embeddings, the final norm, the
/// output head - is kept as it is, because every machine loads a small number of
/// those and only the last one uses the head.
func renumberLayers(_ weights: [String: MLXArray], keeping: Range<Int>) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)

    for (key, value) in weights {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false)
        // Find "layers" followed by a number, wherever it sits in the path.
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count,
              let index = Int(parts[at + 1])
        else {
            out[key] = value
            continue
        }
        guard keeping.contains(index) else { continue }

        var renumbered = parts
        renumbered[at + 1] = Substring(String(index - keeping.lowerBound))
        out[renumbered.joined(separator: ".")] = value
    }
    return out
}
