import Foundation
import Accelerate

// Vector kernels for the wire codecs.
//
// These produce byte-for-byte the same frames the scalar loops produced — the
// wire format is a compatibility contract, and a vectorised sender has to
// interoperate with a scalar receiver still running last week's build.
// `CompressionInteropTests` pins that equality down, including the rounding.
//
// Why the scalar versions were slow is worth recording, because it was not the
// arithmetic:
//
//  * `WireCodec.encode` walked its buffer with `i * width`, where `width` is a
//    *runtime* value read from the dtype, and `ElementIO.loadFloat` switched on
//    that dtype once per element. Neither the stride nor the element type was a
//    compile-time constant inside the loop, so the vectoriser had nothing to
//    work with. For fp32<->fp16 that is the entire story: a plain loop with a
//    constant stride is auto-vectorised into `fcvtn`/`fcvtl` pairs and runs at
//    50-80 GB/s, and hand-written SIMD is *slower* than letting the compiler do
//    it (`SIMD8<Float>(SIMD8<Float16>)` lowers to eight scalar converts).
//  * Where explicit SIMD does earn its place — the int8 blockwise scan and
//    top-k's residual fold — two standard-library conveniences have to be
//    avoided, because neither lowers to vector instructions today:
//    `clamped(lowerBound:upperBound:)`/`pointwiseMin`/`pointwiseMax` (~2 GB/s,
//    against ~19 GB/s for the same clamp written with `replace(with:where:)`)
//    and every float-to-integer SIMD conversion (~0.2 GB/s). The quantiser
//    therefore prepares floats with SIMD and hands the conversion to
//    `vDSP_vfix8`, which is one vectorised pass at ~78 GB/s.
//
// Every kernel here is a fast path for `.float32`, which is the dtype the
// codecs exist for: `downcast` is defined only for fp32, and an fp16 or bf16
// buffer is already half-width. Other dtypes keep the scalar path.

/// Lane count for the float kernels: NEON is 128-bit, so eight floats is two
/// registers per iteration.
private let lanes = 8
private typealias F8 = SIMD8<Float>

@inline(__always)
private func magnitude(_ v: F8) -> F8 {
    // `abs` is not defined on `SIMD8<Float>` in the standard library and the
    // simd module's overload is gated behind a newer SDK, so negate the lanes
    // that need it. NaN compares false and keeps its sign, which is fine: every
    // caller masks non-finite lanes out separately.
    v.replacing(with: -v, where: v .< F8())
}

enum CodecKernels {

    /// Elements per scratch chunk in the quantiser. Small enough that
    /// `withUnsafeTemporaryAllocation` keeps it on the stack, large enough that
    /// the per-call cost of `vDSP_vfix8` disappears.
    static let quantizeChunk = 4096

    // MARK: - fp32 <-> fp16

    /// fp32 -> fp16, round-to-nearest-even, exactly as `Float16(_:)` rounds.
    ///
    /// Deliberately not hand-vectorised: with the stride known at compile time
    /// the optimiser emits the two-register `fcvtn` sequence itself, and beats
    /// every explicit SIMD spelling tried.
    static func encodeFP16(from src: UnsafeRawPointer, count: Int, into dst: UnsafeMutableRawPointer) {
        let source = src.bindMemory(to: Float.self, capacity: count)
        let target = dst.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { target[i] = Float16(source[i]) }
    }

    /// fp16 -> fp32. Widening is exact, so there is no rounding to preserve.
    static func decodeFP16(from src: UnsafeRawPointer, count: Int, into dst: UnsafeMutableRawPointer) {
        let source = src.bindMemory(to: Float16.self, capacity: count)
        let target = dst.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { target[i] = Float(source[i]) }
    }

    // MARK: - int8 blockwise

    /// Largest finite magnitude in `src[0..<count]`, or 0 for an all-NaN or
    /// all-infinite block. Non-finite lanes are excluded rather than clamped:
    /// one stray infinity would otherwise drive the block's scale to infinity
    /// and quantise every real value in it to zero.
    ///
    /// `vDSP_maxmgv` is not used here precisely because it propagates NaN and
    /// infinity instead of skipping them.
    static func absmaxFP32(_ src: UnsafeRawPointer, count: Int) -> Float {
        var accumulator = F8()
        var i = 0
        while i + lanes <= count {
            let a = magnitude(src.loadUnaligned(fromByteOffset: i * 4, as: F8.self))
            // `a .< .infinity` is false for both NaN and +inf: one mask, both
            // exclusions.
            let candidate = a.replacing(with: F8(), where: .!(a .< F8(repeating: .infinity)))
            accumulator = accumulator.replacing(with: candidate, where: candidate .> accumulator)
            i += lanes
        }
        var absmax = accumulator.max()
        while i < count {
            let v = abs(src.loadUnaligned(fromByteOffset: i * 4, as: Float.self))
            if v.isFinite, v > absmax { absmax = v }
            i += 1
        }
        return absmax
    }

    /// Quantises `count` fp32 elements at `1/scale` into int8 codes: round half
    /// away from zero, NaN to 0, clamped to ±127 so the codes stay symmetric —
    /// the scalar encoder's semantics, element for element.
    ///
    /// Two passes over a stack chunk: SIMD does the arithmetic in float, and
    /// `vDSP_vfix8` does the narrowing, which truncates toward zero and is
    /// therefore exact on the already-rounded values.
    static func quantizeFP32(
        _ src: UnsafeRawPointer, count: Int, inverseScale: Float, into dst: UnsafeMutableRawPointer
    ) {
        let codes = dst.assumingMemoryBound(to: Int8.self)
        let chunkCapacity = Swift.min(Swift.max(count, 1), quantizeChunk)
        withUnsafeTemporaryAllocation(of: Float.self, capacity: chunkCapacity) { scratch in
            let limit = F8(repeating: 127)
            let inv = F8(repeating: inverseScale)
            var base = 0
            while base < count {
                let span = Swift.min(chunkCapacity, count - base)
                var i = 0
                while i + lanes <= span {
                    var t = src.loadUnaligned(fromByteOffset: (base + i) * 4, as: F8.self) * inv
                    // NaN never compares equal to itself. The scalar path stored
                    // 0 for it, and a NaN reaching a float->int conversion is
                    // undefined, so it has to go before the narrowing.
                    t.replace(with: F8(), where: t .!= t)
                    t = t.rounded(.toNearestOrAwayFromZero)
                    // Written as two masked replaces rather than `clamped`,
                    // which does not vectorise. ±infinity lands on ±127 here,
                    // exactly as the scalar comparison chain did.
                    t.replace(with: limit, where: t .> limit)
                    t.replace(with: -limit, where: t .< -limit)
                    UnsafeMutableRawPointer(scratch.baseAddress! + i).storeBytes(of: t, as: F8.self)
                    i += lanes
                }
                while i < span {
                    var q = (src.loadUnaligned(fromByteOffset: (base + i) * 4, as: Float.self)
                             * inverseScale).rounded()
                    if q > 127 { q = 127 } else if q < -127 { q = -127 } else if q.isNaN { q = 0 }
                    scratch[i] = q
                    i += 1
                }
                vDSP_vfix8(scratch.baseAddress!, 1, codes + base, 1, vDSP_Length(span))
                base += span
            }
        }
    }

    /// Dequantises `count` int8 codes into fp32 at `scale`.
    static func dequantizeFP32(
        _ src: UnsafeRawPointer, count: Int, scale: Float, into dst: UnsafeMutableRawPointer
    ) {
        let s = F8(repeating: scale)
        var i = 0
        while i + lanes <= count {
            let codes = src.loadUnaligned(fromByteOffset: i, as: SIMD8<Int8>.self)
            dst.storeBytes(of: F8(codes) * s, toByteOffset: i * 4, as: F8.self)
            i += lanes
        }
        while i < count {
            let code = src.loadUnaligned(fromByteOffset: i, as: Int8.self)
            dst.storeBytes(of: Float(code) * scale, toByteOffset: i * 4, as: Float.self)
            i += 1
        }
    }

    // MARK: - top-k

    /// How many of `count` scores are strictly greater than `threshold`.
    ///
    /// Top-k's collecting pass needs this before it starts, so that a run of
    /// ties early in the buffer cannot fill the block and displace a larger
    /// element further down. It is a pure reduction, so it vectorises: compare,
    /// turn the mask into ones, add.
    static func countAboveFP32(_ src: UnsafePointer<Float>, count: Int, threshold: Float) -> Int {
        let bar = F8(repeating: threshold)
        let one = SIMD8<Int32>(repeating: 1)
        var tally = SIMD8<Int32>()
        var total = 0
        var i = 0
        // Int32 lanes cannot overflow inside a chunk this size, and draining
        // the accumulator keeps the inner loop to compare-select-add.
        let chunk = 1 << 20
        while i + lanes <= count {
            let limit = Swift.min(count - (count % lanes), i + chunk)
            while i + lanes <= limit {
                let v = UnsafeRawPointer(src + i).loadUnaligned(as: F8.self)
                tally &+= SIMD8<Int32>().replacing(with: one, where: v .> bar)
                i += lanes
            }
            total += Int(tally.wrappedSum())
            tally = SIMD8<Int32>()
        }
        while i < count {
            if src[i] > threshold { total += 1 }
            i += 1
        }
        return total
    }

    /// `work += src` and `magnitudes = |work|`, with non-finite elements scored
    /// zero so selection can never pick one.
    static func foldResidualFP32(
        source: UnsafeRawPointer, count: Int,
        work: UnsafeMutablePointer<Float>, magnitudes: UnsafeMutablePointer<Float>
    ) {
        var i = 0
        while i + lanes <= count {
            let value = source.loadUnaligned(fromByteOffset: i * 4, as: F8.self)
                + UnsafeRawPointer(work + i).loadUnaligned(as: F8.self)
            UnsafeMutableRawPointer(work + i).storeBytes(of: value, as: F8.self)
            let a = magnitude(value)
            let scored = a.replacing(with: F8(), where: .!(a .< F8(repeating: .infinity)))
            UnsafeMutableRawPointer(magnitudes + i).storeBytes(of: scored, as: F8.self)
            i += lanes
        }
        while i < count {
            let value = source.loadUnaligned(fromByteOffset: i * 4, as: Float.self) + work[i]
            work[i] = value
            magnitudes[i] = value.isFinite ? abs(value) : 0
            i += 1
        }
    }
}
