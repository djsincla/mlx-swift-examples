// Added for dAI. Not part of upstream mlx-swift-examples.
//
// Splitting one model across several machines, so that a model too large for
// any single machine can be served by a group of them.
//
// The Python mlx-lm has this and applies it only to certain mixture-of-experts
// architectures; no released version pipelines Qwen. The mechanism is small and
// the same either way: divide the layers, and pass the hidden state between the
// machines that hold each slice.
//
// The transport is deliberately not defined here. This library should not know
// about certificates, fleets or sockets, and the code that does should not have
// to fork a model library to change how bytes move. A caller supplies something
// that can send and receive a tensor and nothing else.

import Foundation
import MLX

/// Moves a hidden state between the machines holding adjacent slices of a model.
///
/// Synchronous on purpose. A model's forward pass is synchronous, and a pipeline
/// step genuinely has nothing else to do until the tensor arrives, so an
/// asynchronous interface here would buy nothing and would force every model in
/// the library to become async.
public protocol PipelineTransport: AnyObject, Sendable {
    /// Hand a hidden state to the machine holding the next slice.
    func send(_ x: MLXArray, to rank: Int) throws

    /// Take the hidden state from the machine holding the previous slice.
    /// `like` carries the shape and element type expected, so the
    /// implementation can validate what arrived rather than trusting it.
    func receive(like: MLXArray, from rank: Int) throws -> MLXArray
}

/// Which slice of a model's layers this machine is responsible for.
///
/// Assigned in reverse, so rank 0 holds the *last* layers along with the final
/// norm and the output head. Rank 0 is therefore the machine that produces a
/// token, which is the one a scheduler addresses and the one that answers.
public struct PipelineSplit: Sendable, Equatable {
    public let rank: Int
    public let size: Int
    public let startIndex: Int
    public let endIndex: Int

    public init(rank: Int, size: Int, layerCount: Int) {
        precondition(size > 0 && rank >= 0 && rank < size, "rank \(rank) is not in 0..<\(size)")
        self.rank = rank
        self.size = size

        // Boundaries accumulated rather than multiplied.
        //
        // The obvious formula, and the one upstream uses, is to give this rank
        // `perRank` layers and start it at `(size - rank - 1) * perRank`. That
        // is only correct when every rank holds the same number: with 80 layers
        // over 3 machines the ranks hold 27, 27 and 26, and multiplying by this
        // rank's own count leaves layer 26 owned by nobody. A skipped layer does
        // not fail; the model simply computes without it and answers fluently
        // from the wrong network.
        let base = layerCount / size
        let extra = layerCount - base * size
        func count(of r: Int) -> Int { base + (r < extra ? 1 : 0) }

        // Ranks are numbered in reverse, so everything after this one in rank
        // order sits before it in layer order.
        var start = 0
        for higher in stride(from: size - 1, to: rank, by: -1) { start += count(of: higher) }

        self.startIndex = start
        self.endIndex = start + count(of: rank)
    }

    /// Whether this machine holds the final layers, the norm and the head.
    public var isLast: Bool { rank == 0 }

    /// Whether this machine holds the very first layers and the embeddings.
    public var isFirst: Bool { rank == size - 1 }

    /// Nothing is split. The ordinary single-machine case.
    public static func whole(layerCount: Int) -> PipelineSplit {
        PipelineSplit(rank: 0, size: 1, layerCount: layerCount)
    }
}

/// A model that can be divided across machines.
public protocol Pipelineable: AnyObject {
    /// Keep only the layers this machine owns, and route the hidden state
    /// through the supplied transport.
    func pipeline(_ split: PipelineSplit, transport: PipelineTransport)
}
