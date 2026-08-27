import Foundation

// MARK: - DataType layout

extension DataType {
    /// Size of one element in bytes.
    public var byteWidth: Int {
        switch self {
        case .float32, .int32: return 4
        case .float16, .bfloat16: return 2
        case .int8: return 1
        }
    }

    public var isFloatingPoint: Bool {
        switch self {
        case .float32, .float16, .bfloat16: return true
        case .int32, .int8: return false
        }
    }

    /// Stable on-the-wire discriminator. Must never be renumbered — the C shim
    /// will expose these as `mcclDataType_t`.
    var wireCode: UInt8 {
        switch self {
        case .float32: return 0
        case .float16: return 1
        case .bfloat16: return 2
        case .int32: return 3
        case .int8: return 4
        }
    }

    static func from(wireCode: UInt8) throws -> DataType {
        switch wireCode {
        case 0: return .float32
        case 1: return .float16
        case 2: return .bfloat16
        case 3: return .int32
        case 4: return .int8
        default: throw MCCLError.protocolViolation("unknown dtype code \(wireCode)")
        }
    }
}

extension DataType: CustomStringConvertible {
    public var description: String {
        switch self {
        case .float32: return "fp32"
        case .float16: return "fp16"
        case .bfloat16: return "bf16"
        case .int32: return "i32"
        case .int8: return "i8"
        }
    }
}

extension ReduceOp {
    /// `.avg` is executed as a sum across the collective and scaled once at the
    /// end; every wire-level reduction therefore uses this op.
    var wireReduction: ReduceOp { self == .avg ? .sum : self }
}

// MARK: - bfloat16

/// bfloat16 has no native Swift type; we carry it as `UInt16` holding the top
/// 16 bits of the fp32 bit pattern.
enum BFloat16 {
    @inline(__always)
    static func toFloat(_ x: UInt16) -> Float {
        Float(bitPattern: UInt32(x) << 16)
    }

    /// Round-to-nearest-even fp32 -> bf16.
    @inline(__always)
    static func fromFloat(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        if f.isNaN { return UInt16(truncatingIfNeeded: (bits >> 16)) | 0x0040 }
        let rounding = UInt32(0x7FFF) &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ rounding) >> 16)
    }
}

// MARK: - Element access helpers (unaligned-safe)

/// Reads/writes a single element as `Float`, regardless of the underlying
/// dtype. Used by the compression codecs, which operate on wire buffers whose
/// alignment we do not control.
enum ElementIO {
    @inline(__always)
    static func loadFloat(_ p: UnsafeRawPointer, byteOffset: Int, _ dt: DataType) -> Float {
        switch dt {
        case .float32: return p.loadUnaligned(fromByteOffset: byteOffset, as: Float.self)
        case .float16: return Float(p.loadUnaligned(fromByteOffset: byteOffset, as: Float16.self))
        case .bfloat16: return BFloat16.toFloat(p.loadUnaligned(fromByteOffset: byteOffset, as: UInt16.self))
        case .int32: return Float(p.loadUnaligned(fromByteOffset: byteOffset, as: Int32.self))
        case .int8: return Float(p.loadUnaligned(fromByteOffset: byteOffset, as: Int8.self))
        }
    }

    @inline(__always)
    static func storeFloat(_ v: Float, into p: UnsafeMutableRawPointer, byteOffset: Int, _ dt: DataType) {
        switch dt {
        case .float32: p.storeBytes(of: v, toByteOffset: byteOffset, as: Float.self)
        case .float16: p.storeBytes(of: Float16(v), toByteOffset: byteOffset, as: Float16.self)
        case .bfloat16: p.storeBytes(of: BFloat16.fromFloat(v), toByteOffset: byteOffset, as: UInt16.self)
        case .int32: p.storeBytes(of: Int32(clampToInt32(v)), toByteOffset: byteOffset, as: Int32.self)
        case .int8: p.storeBytes(of: Int8(clampToInt8(v)), toByteOffset: byteOffset, as: Int8.self)
        }
    }

    @inline(__always)
    static func clampToInt32(_ v: Float) -> Int32 {
        if v.isNaN { return 0 }
        let r = v.rounded()
        if r >= 2147483520.0 { return Int32.max }
        if r <= -2147483648.0 { return Int32.min }
        return Int32(r)
    }

    @inline(__always)
    static func clampToInt8(_ v: Float) -> Int8 {
        if v.isNaN { return 0 }
        let r = v.rounded()
        if r >= 127 { return 127 }
        if r <= -128 { return -128 }
        return Int8(r)
    }
}

// MARK: - Reduction kernels

/// Lane count for the float reduction kernels — two 128-bit NEON registers per
/// iteration, matching `CodecKernels`.
private let reduceLanes = 8
private typealias R8 = SIMD8<Float>

/// Elementwise reductions: `dst = op(dst, src)`, and the `.avg` scale.
///
/// The ring folds one received chunk into the local buffer on every hop, so
/// `reduce` runs `n-1` times per all-reduce over the whole payload. That is not
/// free even when the cable is the bottleneck — at 1 GB/s over Thunderbolt a
/// kernel running at 16 GB/s still spends ~6% of the wall clock, and over a
/// fast fabric or in-process it is the whole cost.
///
/// The original loops were dtype-specialised on the *outside* but switched on
/// `op` on the *inside*, once per element, via a shared `combine(_:_:_:)`. That
/// is the same shape that cost the wire codecs an order of magnitude (see the
/// note at the top of `CodecKernels.swift`): with a runtime value in the loop
/// body the vectoriser has nothing to work with. Hoisting the `op` switch out
/// of the loop — so each loop body is a single compile-time-known operation —
/// is the entire fix.
///
/// Semantics are unchanged element for element, including the parts a vector
/// rewrite most easily breaks:
///
///  * `Swift.min(x, y)` is `y < x ? y : x`, which is **not** symmetric in NaN.
///    The masked selects below spell that out rather than using a hardware
///    `fmin`, which is. `ReduceKernelTests` pins it.
///  * fp16 and bf16 reduce through an fp32 intermediate and round once on
///    store, as they always did — not in half precision.
///  * the integer dtypes wrap (`&+`, `&*`); only `scale` saturates.
enum Kernels {
    static func reduce(
        into dst: UnsafeMutableRawPointer,
        from src: UnsafeRawPointer,
        count: Int,
        dataType: DataType,
        op: ReduceOp
    ) {
        let op = op.wireReduction
        guard count > 0 else { return }
        switch dataType {
        case .float32:
            let d = dst.bindMemory(to: Float.self, capacity: count)
            let s = src.bindMemory(to: Float.self, capacity: count)
            switch op {
            case .sum, .avg: mapF32(d, s, count, { $0 + $1 }, { $0 + $1 })
            case .prod: mapF32(d, s, count, { $0 * $1 }, { $0 * $1 })
            // `b .< a ? b : a` — Swift.min's own definition, lane by lane.
            case .min: mapF32(d, s, count, { $0.replacing(with: $1, where: $1 .< $0) },
                              { Swift.min($0, $1) })
            case .max: mapF32(d, s, count, { $0.replacing(with: $1, where: $1 .>= $0) },
                              { Swift.max($0, $1) })
            }
        case .float16:
            let d = dst.bindMemory(to: Float16.self, capacity: count)
            let s = src.bindMemory(to: Float16.self, capacity: count)
            switch op {
            case .sum, .avg: mapNarrow(d, s, count) { $0 + $1 }
            case .prod: mapNarrow(d, s, count) { $0 * $1 }
            case .min: mapNarrow(d, s, count) { Swift.min($0, $1) }
            case .max: mapNarrow(d, s, count) { Swift.max($0, $1) }
            }
        case .bfloat16:
            let d = dst.bindMemory(to: UInt16.self, capacity: count)
            let s = src.bindMemory(to: UInt16.self, capacity: count)
            switch op {
            case .sum, .avg: mapBF16(d, s, count) { $0 + $1 }
            case .prod: mapBF16(d, s, count) { $0 * $1 }
            case .min: mapBF16(d, s, count) { Swift.min($0, $1) }
            case .max: mapBF16(d, s, count) { Swift.max($0, $1) }
            }
        case .int32:
            let d = dst.bindMemory(to: Int32.self, capacity: count)
            let s = src.bindMemory(to: Int32.self, capacity: count)
            switch op {
            case .sum, .avg: mapInteger(d, s, count) { $0 &+ $1 }
            case .prod: mapInteger(d, s, count) { $0 &* $1 }
            case .min: mapInteger(d, s, count) { Swift.min($0, $1) }
            case .max: mapInteger(d, s, count) { Swift.max($0, $1) }
            }
        case .int8:
            let d = dst.bindMemory(to: Int8.self, capacity: count)
            let s = src.bindMemory(to: Int8.self, capacity: count)
            switch op {
            case .sum, .avg: mapInteger(d, s, count) { $0 &+ $1 }
            case .prod: mapInteger(d, s, count) { $0 &* $1 }
            case .min: mapInteger(d, s, count) { Swift.min($0, $1) }
            case .max: mapInteger(d, s, count) { Swift.max($0, $1) }
            }
        }
    }

    /// fp32: explicit eight-lane body plus a scalar tail. Written out rather
    /// than left to the auto-vectoriser because `min`/`max` need the exact
    /// NaN-asymmetric select, which no `fmin` instruction gives.
    @inline(__always)
    private static func mapF32(
        _ d: UnsafeMutablePointer<Float>, _ s: UnsafePointer<Float>, _ count: Int,
        _ vector: (R8, R8) -> R8, _ scalar: (Float, Float) -> Float
    ) {
        var i = 0
        while i + reduceLanes <= count {
            let a = UnsafeRawPointer(d + i).loadUnaligned(as: R8.self)
            let b = UnsafeRawPointer(s + i).loadUnaligned(as: R8.self)
            UnsafeMutableRawPointer(d + i).storeBytes(of: vector(a, b), as: R8.self)
            i += reduceLanes
        }
        while i < count { d[i] = scalar(d[i], s[i]); i += 1 }
    }

    /// fp16: widen, combine in fp32, round once on store. A constant stride and
    /// a constant operation are all the optimiser needs to emit the
    /// `fcvtl`/`fcvtn` pair — hand-written SIMD is slower here, for the reason
    /// recorded in `CodecKernels.swift`.
    @inline(__always)
    private static func mapNarrow(
        _ d: UnsafeMutablePointer<Float16>, _ s: UnsafePointer<Float16>, _ count: Int,
        _ combine: (Float, Float) -> Float
    ) {
        for i in 0..<count { d[i] = Float16(combine(Float(d[i]), Float(s[i]))) }
    }

    /// bf16 is carried as `UInt16`; the round-to-nearest-even store has a NaN
    /// branch in it, so this stays a scalar loop — but a scalar loop over one
    /// known operation, not a switch per element.
    @inline(__always)
    private static func mapBF16(
        _ d: UnsafeMutablePointer<UInt16>, _ s: UnsafePointer<UInt16>, _ count: Int,
        _ combine: (Float, Float) -> Float
    ) {
        for i in 0..<count {
            d[i] = BFloat16.fromFloat(combine(BFloat16.toFloat(d[i]), BFloat16.toFloat(s[i])))
        }
    }

    /// int32/int8. `SIMDScalar` integers auto-vectorise from a plain loop once
    /// the operation is fixed; `Int8.min`/`max` and the wrapping operators all
    /// have direct NEON forms.
    @inline(__always)
    private static func mapInteger<T: FixedWidthInteger>(
        _ d: UnsafeMutablePointer<T>, _ s: UnsafePointer<T>, _ count: Int,
        _ combine: (T, T) -> T
    ) {
        for i in 0..<count { d[i] = combine(d[i], s[i]) }
    }

    /// Multiply every element by `factor`. Used to turn a summed all-reduce
    /// into `.avg` exactly once, at the end.
    ///
    /// The integer dtypes saturate rather than wrap: an averaged accumulator
    /// that silently flipped sign would be worse than a clamped one.
    ///
    /// Left as plain loops on purpose. `scale` never had the per-element switch
    /// — `factor` is hoisted to a local before the loop and each dtype gets its
    /// own body — so it was already auto-vectorising. Measured, the fp32 path
    /// runs at ~98 GB/s both before and after this rewrite, and replacing it
    /// with hand-written SIMD changed nothing. It is documented here so the
    /// next person does not "optimise" it again.
    static func scale(
        _ buffer: UnsafeMutableRawPointer,
        count: Int,
        dataType: DataType,
        by factor: Double
    ) {
        guard count > 0, factor != 1.0 else { return }
        let f = Float(factor)
        switch dataType {
        case .float32:
            let d = buffer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { d[i] *= f }
        case .float16:
            let d = buffer.bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count { d[i] = Float16(Float(d[i]) * f) }
        case .bfloat16:
            let d = buffer.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count { d[i] = BFloat16.fromFloat(BFloat16.toFloat(d[i]) * f) }
        case .int32:
            let d = buffer.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { d[i] = ElementIO.clampToInt32(Float(d[i]) * f) }
        case .int8:
            let d = buffer.bindMemory(to: Int8.self, capacity: count)
            for i in 0..<count { d[i] = ElementIO.clampToInt8(Float(d[i]) * f) }
        }
    }
}

// MARK: - Chunking

/// Splits `count` elements into `parts` near-equal contiguous chunks — the
/// partition the ring algorithms stream around the loop.
struct ChunkLayout {
    let counts: [Int]
    let offsets: [Int]   // in elements

    init(count: Int, parts: Int) {
        precondition(parts > 0)
        let base = count / parts
        let remainder = count % parts
        var counts: [Int] = []
        var offsets: [Int] = []
        counts.reserveCapacity(parts)
        offsets.reserveCapacity(parts)
        var cursor = 0
        for i in 0..<parts {
            let c = base + (i < remainder ? 1 : 0)
            counts.append(c)
            offsets.append(cursor)
            cursor += c
        }
        self.counts = counts
        self.offsets = offsets
    }

    var maxCount: Int { counts.max() ?? 0 }
}
