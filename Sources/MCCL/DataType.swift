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

/// Elementwise reductions. Scalar loops; the interconnect is the bottleneck on
/// a Mac cluster, not this arithmetic, so vectorising is deferred.
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
            for i in 0..<count { d[i] = combine(d[i], s[i], op) }
        case .float16:
            let d = dst.bindMemory(to: Float16.self, capacity: count)
            let s = src.bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count { d[i] = Float16(combine(Float(d[i]), Float(s[i]), op)) }
        case .bfloat16:
            let d = dst.bindMemory(to: UInt16.self, capacity: count)
            let s = src.bindMemory(to: UInt16.self, capacity: count)
            for i in 0..<count {
                d[i] = BFloat16.fromFloat(combine(BFloat16.toFloat(d[i]), BFloat16.toFloat(s[i]), op))
            }
        case .int32:
            let d = dst.bindMemory(to: Int32.self, capacity: count)
            let s = src.bindMemory(to: Int32.self, capacity: count)
            for i in 0..<count { d[i] = combine(d[i], s[i], op) }
        case .int8:
            let d = dst.bindMemory(to: Int8.self, capacity: count)
            let s = src.bindMemory(to: Int8.self, capacity: count)
            for i in 0..<count { d[i] = combine(d[i], s[i], op) }
        }
    }

    @inline(__always)
    private static func combine(_ a: Float, _ b: Float, _ op: ReduceOp) -> Float {
        switch op {
        case .sum, .avg: return a + b
        case .prod: return a * b
        case .min: return Swift.min(a, b)
        case .max: return Swift.max(a, b)
        }
    }

    @inline(__always)
    private static func combine(_ a: Int32, _ b: Int32, _ op: ReduceOp) -> Int32 {
        switch op {
        case .sum, .avg: return a &+ b
        case .prod: return a &* b
        case .min: return Swift.min(a, b)
        case .max: return Swift.max(a, b)
        }
    }

    @inline(__always)
    private static func combine(_ a: Int8, _ b: Int8, _ op: ReduceOp) -> Int8 {
        switch op {
        case .sum, .avg: return a &+ b
        case .prod: return a &* b
        case .min: return Swift.min(a, b)
        case .max: return Swift.max(a, b)
        }
    }

    /// Multiply every element by `factor`. Used to turn a summed all-reduce
    /// into `.avg` exactly once, at the end.
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
