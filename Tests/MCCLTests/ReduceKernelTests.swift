import XCTest
@testable import MCCL

// `Kernels.reduce` / `Kernels.scale` are the only arithmetic on the critical
// path of every collective: the ring folds a received chunk into the local one
// `n-1` times per all-reduce, and `.avg` scales the result once at the end.
//
// They were written as scalar loops on the assumption that the cable, not the
// CPU, is the bottleneck. That assumption is the same one the wire codecs were
// written under, and it was wrong there for a specific and repeatable reason:
// the loop body switched on a *runtime* value, so the vectoriser had nothing
// constant to work with. `reduce` had exactly that shape — `combine(_:_:_:)`
// switched on `op` once per element.
//
// These tests pin the semantics element-for-element against an independently
// written reference (including the rounding, which is what a dtype-specialised
// rewrite is most likely to change) and report the achieved rate.

final class ReduceKernelTests: XCTestCase {

    // MARK: - References
    //
    // Deliberately restated here rather than shared with the library: a
    // reference that calls the code under test proves nothing.

    private func referenceFloat(_ a: Float, _ b: Float, _ op: ReduceOp) -> Float {
        switch op {
        case .sum, .avg: return a + b
        case .prod: return a * b
        case .min: return Swift.min(a, b)
        case .max: return Swift.max(a, b)
        }
    }

    private func referenceInt32(_ a: Int32, _ b: Int32, _ op: ReduceOp) -> Int32 {
        switch op {
        case .sum, .avg: return a &+ b
        case .prod: return a &* b
        case .min: return Swift.min(a, b)
        case .max: return Swift.max(a, b)
        }
    }

    private func referenceInt8(_ a: Int8, _ b: Int8, _ op: ReduceOp) -> Int8 {
        switch op {
        case .sum, .avg: return a &+ b
        case .prod: return a &* b
        case .min: return Swift.min(a, b)
        case .max: return Swift.max(a, b)
        }
    }

    /// Values chosen to exercise the tails as well as the vector body: the
    /// counts used below are deliberately not multiples of the lane width.
    private func sample(_ i: Int, _ salt: Int) -> Float {
        let unit = Float((i &* 2654435761 &+ salt &* 40503) % 2000) / 1000.0 - 1.0
        return unit * Float(1 + (i % 7))
    }

    // MARK: - Correctness, every dtype x every op

    func testReduceMatchesScalarReferenceForEveryDataTypeAndOp() {
        let ops: [ReduceOp] = [.sum, .prod, .min, .max, .avg]
        // 1..17 covers empty-ish, sub-lane, exactly-one-lane and lane+tail.
        for count in [1, 3, 8, 9, 17, 1000, 1003] {
            for op in ops {
                for dataType in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
                    assertReduceMatches(count: count, dataType: dataType, op: op)
                }
            }
        }
    }

    private func assertReduceMatches(count: Int, dataType: DataType, op: ReduceOp) {
        let width = dataType.byteWidth
        let dst = UnsafeMutableRawBufferPointer.allocate(byteCount: count * width, alignment: 64)
        let src = UnsafeMutableRawBufferPointer.allocate(byteCount: count * width, alignment: 64)
        defer { dst.deallocate(); src.deallocate() }

        // Small integers for the integer dtypes so `prod` stays interesting
        // rather than saturating everything to a fixed point.
        for i in 0..<count {
            let a = dataType.isFloatingPoint ? sample(i, 1) : Float((i % 9) - 4)
            let b = dataType.isFloatingPoint ? sample(i, 2) : Float((i % 5) - 2)
            store(a, into: dst.baseAddress!, byteOffset: i * width, dataType)
            store(b, into: src.baseAddress!, byteOffset: i * width, dataType)
        }

        var expected: [Float] = []
        expected.reserveCapacity(count)
        for i in 0..<count {
            let a = load(dst.baseAddress!, byteOffset: i * width, dataType)
            let b = load(src.baseAddress!, byteOffset: i * width, dataType)
            switch dataType {
            case .float32:
                expected.append(referenceFloat(a, b, op))
            case .float16:
                expected.append(Float(Float16(referenceFloat(a, b, op))))
            case .bfloat16:
                expected.append(BFloat16.toFloat(BFloat16.fromFloat(referenceFloat(a, b, op))))
            case .int32:
                expected.append(Float(referenceInt32(Int32(a), Int32(b), op)))
            case .int8:
                expected.append(Float(referenceInt8(Int8(a), Int8(b), op)))
            }
        }

        Kernels.reduce(into: dst.baseAddress!, from: src.baseAddress!,
                       count: count, dataType: dataType, op: op)

        for i in 0..<count {
            let got = load(dst.baseAddress!, byteOffset: i * width, dataType)
            XCTAssertEqual(got, expected[i], accuracy: 0,
                           "\(dataType) \(op) count=\(count) element \(i)")
        }
    }

    /// NaN handling is the part a rewrite is most likely to get subtly wrong:
    /// `Swift.min(x, y)` is `y < x ? y : x`, which is *not* symmetric in NaN,
    /// and a vector `minimum` instruction usually is.
    func testReduceMinMaxNaNAsymmetryIsPreserved() {
        for op in [ReduceOp.min, .max] {
            var dst: [Float] = [.nan, 1.0, .nan, 2.0]
            let src: [Float] = [1.0, .nan, .nan, 3.0]
            let expected = zip(dst, src).map { referenceFloat($0, $1, op) }
            dst.withUnsafeMutableBytes { d in
                src.withUnsafeBytes { s in
                    Kernels.reduce(into: d.baseAddress!, from: s.baseAddress!,
                                   count: 4, dataType: .float32, op: op)
                }
            }
            for i in 0..<4 {
                if expected[i].isNaN {
                    XCTAssertTrue(dst[i].isNaN, "\(op) element \(i) should be NaN")
                } else {
                    XCTAssertEqual(dst[i], expected[i], "\(op) element \(i)")
                }
            }
        }
    }

    func testScaleMatchesScalarReferenceForEveryDataType() {
        for count in [1, 7, 8, 9, 1003] {
            for factor in [0.5, 1.0 / 3.0, 2.0] {
                for dataType in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
                    let width = dataType.byteWidth
                    let buffer = UnsafeMutableRawBufferPointer.allocate(
                        byteCount: count * width, alignment: 64)
                    defer { buffer.deallocate() }
                    for i in 0..<count {
                        let v = dataType.isFloatingPoint ? sample(i, 3) : Float((i % 13) - 6)
                        store(v, into: buffer.baseAddress!, byteOffset: i * width, dataType)
                    }
                    let f = Float(factor)
                    var expected: [Float] = []
                    for i in 0..<count {
                        let v = load(buffer.baseAddress!, byteOffset: i * width, dataType)
                        switch dataType {
                        case .float32: expected.append(v * f)
                        case .float16: expected.append(Float(Float16(Float(Float16(v)) * f)))
                        case .bfloat16: expected.append(BFloat16.toFloat(BFloat16.fromFloat(v * f)))
                        case .int32: expected.append(Float(ElementIO.clampToInt32(v * f)))
                        case .int8: expected.append(Float(ElementIO.clampToInt8(v * f)))
                        }
                    }
                    Kernels.scale(buffer.baseAddress!, count: count, dataType: dataType, by: factor)
                    for i in 0..<count {
                        XCTAssertEqual(load(buffer.baseAddress!, byteOffset: i * width, dataType),
                                       expected[i], accuracy: 0,
                                       "\(dataType) x\(factor) count=\(count) element \(i)")
                    }
                }
            }
        }
    }

    /// int32/int8 scaling saturates rather than wrapping — the accumulator
    /// dtypes must not silently produce a negative from a positive.
    func testScaleSaturatesIntegerDataTypes() {
        var i32: [Int32] = [Int32.max, Int32.min, 100]
        i32.withUnsafeMutableBytes {
            Kernels.scale($0.baseAddress!, count: 3, dataType: .int32, by: 4.0)
        }
        XCTAssertEqual(i32[0], Int32.max)
        XCTAssertEqual(i32[1], Int32.min)
        XCTAssertEqual(i32[2], 400)

        var i8: [Int8] = [120, -120, 10]
        i8.withUnsafeMutableBytes {
            Kernels.scale($0.baseAddress!, count: 3, dataType: .int8, by: 3.0)
        }
        XCTAssertEqual(i8[0], 127)
        XCTAssertEqual(i8[1], -128)
        XCTAssertEqual(i8[2], 30)
    }

    // MARK: - Throughput

    /// Reports the achieved rate rather than asserting a specific one — the
    /// number that matters is in ARCHITECTURE.md. The assertion here is only a
    /// floor low enough that it can never be flaky but high enough to catch a
    /// regression back to a per-element switch.
    ///
    /// Release only. An unoptimised build runs these loops around a thousand
    /// times slower (fp32 sum: 1.15 GB/s debug against 75 GB/s release), so a
    /// debug run measures the absence of the optimiser, not the kernel.
    func testReduceThroughput() throws {
        #if DEBUG
        throw XCTSkip("throughput is only meaningful in a release build — "
                      + "swift test -c release --filter testReduceThroughput")
        #else
        let count = 1 << 22   // 16 MiB of fp32 per side
        let dst = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
        let src = UnsafeMutableRawBufferPointer.allocate(byteCount: count * 4, alignment: 64)
        defer { dst.deallocate(); src.deallocate() }
        let d = dst.baseAddress!.bindMemory(to: Float.self, capacity: count)
        let s = src.baseAddress!.bindMemory(to: Float.self, capacity: count)

        // Re-seeded before every timed pass. `prod` in particular would
        // otherwise drive the whole buffer to 0 or infinity within two
        // iterations and measure denormal handling instead of the kernel.
        func seed() {
            for i in 0..<count {
                d[i] = 1.0 + Float(i % 1000) * 0.0001
                s[i] = 1.0 - Float(i % 977) * 0.0001
            }
        }

        for op in [ReduceOp.sum, .prod, .min, .max] {
            var best = Double.infinity
            // One untimed pass per op so the first measurement is not the one
            // that pulls 32 MiB through a cold cache.
            seed()
            Kernels.reduce(into: dst.baseAddress!, from: src.baseAddress!,
                           count: count, dataType: .float32, op: op)
            for _ in 0..<7 {
                seed()
                let start = DispatchTime.now().uptimeNanoseconds
                Kernels.reduce(into: dst.baseAddress!, from: src.baseAddress!,
                               count: count, dataType: .float32, op: op)
                best = Swift.min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            }
            // Two streams read, one written.
            let gbps = Double(count * 4 * 3) / best / 1e9
            print(String(format: "reduce fp32 %@: %.2f GB/s (%.3f ms for %d elements)",
                         String(describing: op), gbps, best * 1e3, count))
            XCTAssertGreaterThan(gbps, 0.5, "reduce fp32 \(op) collapsed to \(gbps) GB/s")
        }

        for dataType in [DataType.float16, .bfloat16, .int32, .int8] {
            let width = dataType.byteWidth
            let a = UnsafeMutableRawBufferPointer.allocate(byteCount: count * width, alignment: 64)
            let b = UnsafeMutableRawBufferPointer.allocate(byteCount: count * width, alignment: 64)
            defer { a.deallocate(); b.deallocate() }
            a.initializeMemory(as: UInt8.self, repeating: 1)
            b.initializeMemory(as: UInt8.self, repeating: 2)
            var best = Double.infinity
            Kernels.reduce(into: a.baseAddress!, from: b.baseAddress!,
                           count: count, dataType: dataType, op: .sum)
            for _ in 0..<5 {
                let start = DispatchTime.now().uptimeNanoseconds
                Kernels.reduce(into: a.baseAddress!, from: b.baseAddress!,
                               count: count, dataType: dataType, op: .sum)
                best = Swift.min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
            }
            let gbps = Double(count * width * 3) / best / 1e9
            print(String(format: "reduce %@ sum: %.2f GB/s", String(describing: dataType), gbps))
            XCTAssertGreaterThan(gbps, 0.2, "reduce \(dataType) sum collapsed to \(gbps) GB/s")
        }

        seed()
        Kernels.scale(dst.baseAddress!, count: count, dataType: .float32, by: 0.5)
        var best = Double.infinity
        for _ in 0..<7 {
            seed()
            let start = DispatchTime.now().uptimeNanoseconds
            Kernels.scale(dst.baseAddress!, count: count, dataType: .float32, by: 0.5)
            best = Swift.min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
        }
        let gbps = Double(count * 4 * 2) / best / 1e9
        print(String(format: "scale fp32: %.2f GB/s (%.3f ms for %d elements)", gbps, best * 1e3, count))
        XCTAssertGreaterThan(gbps, 0.5, "scale fp32 collapsed to \(gbps) GB/s")
        #endif
    }

    // MARK: - Helpers

    private func store(_ v: Float, into p: UnsafeMutableRawPointer, byteOffset: Int, _ dt: DataType) {
        ElementIO.storeFloat(v, into: p, byteOffset: byteOffset, dt)
    }

    private func load(_ p: UnsafeRawPointer, byteOffset: Int, _ dt: DataType) -> Float {
        ElementIO.loadFloat(p, byteOffset: byteOffset, dt)
    }
}
