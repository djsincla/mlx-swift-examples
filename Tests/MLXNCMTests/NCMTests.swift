// dAI: tests for the Next-Concept Modeling port.
//
// The resampling tests need nothing. The parity tests need converted checkpoints, which are
// too large to commit: set NCM_MODEL_DIRS to a colon-separated list of directories written by
// the NCM repository's local/mlx/convert.py, each holding a parity.safetensors written by
// local/mlx/make_fixture.py (inputs plus the official PyTorch model's outputs). Under
// xcodebuild, pass it as TEST_RUNNER_NCM_MODEL_DIRS. Without it those tests are skipped.

import Foundation
import MLX
import XCTest

@testable import MLXNCM

final class NCMResampleTests: XCTestCase {
    func testAreaMatrixAveragesAdaptivePoolWindows() {
        // 5 -> 2: windows [0, 3) and [2, 5), as adaptive_avg_pool1d
        let m = areaMatrix(from: 5, to: 2)
        let third: Float = 1.0 / 3.0
        XCTAssertEqual(m, [third, third, third, 0, 0, 0, 0, third, third, third])
        // rows always sum to one, including non-divisible sizes such as 89 -> 37
        let big = areaMatrix(from: 89, to: 37)
        for i in 0 ..< 37 {
            XCTAssertEqual(big[(i * 89) ..< ((i + 1) * 89)].reduce(0, +), 1, accuracy: 1e-5)
        }
    }

    func testLinearMatrixMatchesAlignCornersFalse() {
        // 2 -> 4 with align_corners=False: sources -0.25(clamped 0), 0.25, 0.75, 1.25(clamped)
        let m = linearMatrix(from: 2, to: 4)
        XCTAssertEqual(m, [1, 0, 0.75, 0.25, 0.25, 0.75, 0, 1])
    }
}

final class NCMParityTests: XCTestCase {
    private func modelDirectories() throws -> [URL] {
        guard let dirs = ProcessInfo.processInfo.environment["NCM_MODEL_DIRS"], !dirs.isEmpty else {
            throw XCTSkip("set NCM_MODEL_DIRS to converted NCM checkpoints to run the parity tests")
        }
        return dirs.split(separator: ":").map { URL(fileURLWithPath: String($0)) }
    }

    func testGreedyCodesAndDecodedTokensMatchPyTorch() throws {
        for dir in try modelDirectories() {
            let model = try NCMModel(directory: dir)
            let fixture = try loadArrays(url: dir.appending(path: "parity.safetensors"))
            let prefix = try XCTUnwrap(fixture["prefix"])
            let expectedCodes = try XCTUnwrap(fixture["codes"])
            let expectedTokens = try XCTUnwrap(fixture["tokens"])

            let codes = try model.generate(prefix: prefix, temperature: 0)
            XCTAssertEqual(codes.shape, expectedCodes.shape, "\(model.configuration.task)")

            // Report agreement level by level, so a mismatch says where it starts.
            var offset = 0
            for (i, l) in model.levels.enumerated() {
                let same = (codes[0..., offset ..< offset + l] .== expectedCodes[0..., offset ..< offset + l])
                    .all().item(Bool.self)
                XCTAssertTrue(same, "\(model.configuration.task): level \(i) codes differ from PyTorch")
                offset += l
            }

            // The decoder, on PyTorch's own codes, must pick the same tokens.
            let tokens = argMax(model.decodeLogits(model.zHat(codes: expectedCodes)), axis: -1)
            let agree = (tokens .== expectedTokens).asType(.float32).mean().item(Float.self)
            XCTAssertEqual(agree, 1, "\(model.configuration.task): decoded tokens differ from PyTorch")
        }
    }

    func testClassifierFreeGuidanceMatchesPyTorch() throws {
        var checked = 0
        for dir in try modelDirectories() {
            let fixture = try loadArrays(url: dir.appending(path: "parity.safetensors"))
            guard let expected = fixture["codes_cfg"], let eos = fixture["eos"] else { continue }
            let model = try NCMModel(directory: dir)
            let codes = try model.generate(
                prefix: try XCTUnwrap(fixture["prefix"]), temperature: 0, cfgScale: 1,
                cfgNullToken: eos[0].item(Int.self))
            XCTAssertTrue((codes .== expected).all().item(Bool.self),
                          "\(model.configuration.task): guided codes differ from PyTorch")
            checked += 1
        }
        if checked == 0 { throw XCTSkip("no fixture has guided reference codes") }
    }

    func testOnLevelReportsEveryLevelCoarseToFine() throws {
        for dir in try modelDirectories() {
            let model = try NCMModel(directory: dir)
            let fixture = try loadArrays(url: dir.appending(path: "parity.safetensors"))
            let prefix = try XCTUnwrap(fixture["prefix"])[..<2]
            var seen: [Int] = []
            var last: MLXArray? = nil
            let codes = try model.generate(prefix: prefix, temperature: 0) { level, z in
                seen.append(level)
                last = z
            }
            XCTAssertEqual(seen, Array(0 ..< model.levels.count))
            // The last report is exactly the decoder input for the finished codes.
            let z = model.zHat(codes: codes)
            XCTAssertLessThan(abs(try XCTUnwrap(last) - z).max().item(Float.self), 1e-4)
        }
    }
}
