import MCCL
import MLX
import XCTest

@testable import MCCLMLX

/// The claims `MLXBridge.swift` makes about copying and about in-place
/// mutation, checked rather than asserted in a comment.
final class MLXBridgeTests: XCTestCase {

    // MARK: - Data type mapping

    func testEverySharedDataTypeRoundTrips() throws {
        for dtype in MLXDataType.supported {
            let mccl = try MLXDataType.mccl(dtype)
            XCTAssertEqual(MLXDataType.mlx(mccl), dtype)
        }
        // And the mapping is total in the other direction.
        for dataType in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
            XCTAssertEqual(try MLXDataType.mccl(MLXDataType.mlx(dataType)), dataType)
        }
    }

    /// A dtype mccl does not carry has to be an error. The alternative — a
    /// silent narrowing to the nearest supported type — produces a plausible
    /// wrong answer, which is the worst kind.
    func testUnsupportedDataTypeIsRejectedRatherThanCast() {
        for dtype in [DType.float64, .int64, .uint32, .bool, .complex64] {
            XCTAssertThrowsError(try MLXDataType.mccl(dtype)) { error in
                guard case MCCLError.invalidArgument(let message) = error else {
                    return XCTFail("expected invalidArgument for \(dtype), got \(error)")
                }
                XCTAssertTrue(message.contains("\(dtype)"), message)
            }
        }
    }

    // MARK: - Copy accounting

    /// The read path's zero-copy claim, measured by address.
    ///
    /// If `asData(access: .noCopyIfContiguous)` were copying, two calls would
    /// hand back two different buffers. They do not, and they both point at the
    /// array's own backing — which is what lets `allGather`'s send side and
    /// `MLXWorkingBuffer.copying`'s read side avoid a copy.
    func testReadPathDoesNotCopyAContiguousArray() throws {
        try MLXRuntime.requireMetallib()
        let array = MLXArray(Array(0..<1024).map { Float($0) }, [32, 32])
        array.eval()

        let first = withContiguousBytes(of: array) { UInt(bitPattern: $0.baseAddress) }
        let second = withContiguousBytes(of: array) { UInt(bitPattern: $0.baseAddress) }
        XCTAssertEqual(first, second, "contiguous read path handed back two different buffers")
        XCTAssertNotEqual(first, 0)
    }

    /// What `eval()` does to a strided *expression*, which turned out to be the
    /// more useful half of this question.
    ///
    /// The adapter is written to tolerate a non-contiguous backing, because
    /// `asData(access: .noCopyIfContiguous)` promises to materialise one. In
    /// practice that fallback almost never fires: MLX's own `eval()` evaluates a
    /// strided slice into a fresh contiguous array, so by the time the adapter
    /// looks at the bytes there is nothing left to materialise, and even a
    /// "strided" input takes the zero-copy read path. Measured, not assumed —
    /// the address is stable across calls, which a per-call copy would not be.
    ///
    /// The fallback stays in the code because it is MLX's contract that is being
    /// relied on here, not a documented guarantee.
    func testStridedSlicesAreMaterialisedByEvalAndThenReadWithoutCopying() throws {
        try MLXRuntime.requireMetallib()
        let base = MLXArray(Array(0..<1024).map { Float($0) }, [32, 32])
        // Every other column: strided as an expression.
        let strided = base[.ellipsis, .stride(by: 2)]
        strided.eval()
        XCTAssertEqual(strided.shape, [32, 16])

        let first = withContiguousBytes(of: strided) { UInt(bitPattern: $0.baseAddress) }
        let second = withContiguousBytes(of: strided) { UInt(bitPattern: $0.baseAddress) }
        XCTAssertEqual(first, second, "the read path copied an already-evaluated slice")
        // It is also not aliasing the *source*, which would be the wrong kind of
        // zero copy — the bytes have to be the slice's, not the base array's.
        let baseAddress = withContiguousBytes(of: base) { UInt(bitPattern: $0.baseAddress) }
        XCTAssertNotEqual(first, baseAddress)

        // And the contents are the slice's, whichever path produced them.
        let values = withContiguousBytes(of: strided) { raw -> [Float] in
            Array(raw.bindMemory(to: Float.self).prefix(4))
        }
        XCTAssertEqual(values, [0, 2, 4, 6])
    }

    /// **The finding that shaped the adapter.**
    ///
    /// `Data(bytesNoCopy:count:deallocator:)` is documented as not copying, and
    /// for buffers of 15 bytes or more it does not. Below that, Foundation
    /// stores the payload in inline storage — copying it — and `withUnsafeBytes`
    /// then hands back the address of a temporary. Writing through that address
    /// is silently lost.
    ///
    /// mlx-swift's only public route to an `MLXArray`'s bytes is exactly this
    /// wrapper (`Cmlx` is not a public product of the package), so this boundary
    /// is what `MLXWorkingBuffer.adoptionFloor` exists to stay clear of. If
    /// Foundation ever moves it, this test says so.
    func testFoundationInlinesSmallNoCopyData() {
        var aliasing: [Int: Bool] = [:]
        for byteCount in [1, 4, 8, 12, 14, 15, 16, 20, 64] {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
            defer { raw.deallocate() }
            raw.initializeMemory(as: UInt8.self, repeating: 0xAA, count: byteCount)
            let data = Data(bytesNoCopy: raw, count: byteCount, deallocator: .none)
            let seen = data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress) }
            aliasing[byteCount] = seen == UInt(bitPattern: raw)
        }
        XCTAssertEqual(aliasing[14], false, "Foundation used to inline 14-byte Data")
        XCTAssertEqual(aliasing[15], true, "Foundation used to alias from 15 bytes up")
        XCTAssertEqual(aliasing[64], true)
        XCTAssertGreaterThanOrEqual(
            MLXWorkingBuffer.adoptionFloor, 15,
            "the adoption floor must sit above Foundation's inline limit")
    }

    /// The load-bearing assumption of the adopted path: bytes written through
    /// the pointer taken from a freshly-created array are the array's contents.
    ///
    /// If this stopped holding, every collective would silently return its input
    /// unchanged — and every all-reduce test where the ranks contribute the same
    /// values would still pass. This one would not.
    func testInPlaceMutationThroughAnAdoptedBufferIsVisible() throws {
        try MLXRuntime.requireMetallib()
        let source = MLXArray((0..<32).map { Float($0 + 1) }, [4, 8])
        let work = try MLXWorkingBuffer.copying(source)
        XCTAssertTrue(work.isAdopted, "128 bytes should be above the adoption floor")
        XCTAssertEqual(work.count, 32)
        XCTAssertEqual(work.byteCount, 128)

        work.base.bindMemory(to: Float.self, capacity: 32)[2] = 99
        let result = work.finish()
        XCTAssertEqual(result.floats[2], 99,
                       "writes through the working buffer are not reaching the array")
        XCTAssertEqual(result.floats[0], 1)
        XCTAssertEqual(result.shape, [4, 8])
        // And the caller's array is untouched, which is why the copy exists.
        XCTAssertEqual(source.floats[2], 3)
    }

    /// The same for the scratch path, which is what a payload below the floor
    /// gets. `finish()` is where its bytes become an array.
    func testInPlaceMutationThroughAScratchBufferIsVisible() throws {
        try MLXRuntime.requireMetallib()
        let source = MLXArray([1, 2, 3, 4] as [Float], [2, 2])
        let work = try MLXWorkingBuffer.copying(source)
        XCTAssertFalse(work.isAdopted, "16 bytes should be below the adoption floor")
        XCTAssertEqual(work.count, 4)
        XCTAssertEqual(work.byteCount, 16)

        work.base.bindMemory(to: Float.self, capacity: 4)[2] = 99
        let result = work.finish()
        XCTAssertEqual(result.floats, [1, 2, 99, 4])
        XCTAssertEqual(result.shape, [2, 2])
        XCTAssertEqual(source.floats, [1, 2, 3, 4])
    }

    /// Both paths carry the contents across, at every size, in every dtype —
    /// which is what makes the threshold an implementation detail rather than a
    /// behavioural one.
    func testWorkingBufferRoundTripsEverySizeAndDataType() throws {
        try MLXRuntime.requireMetallib()
        for dtype in MLXDataType.supported {
            for count in [1, 2, 4, 8, 15, 16, 17, 32, 64, 129] {
                let values = (0..<count).map { Float($0 % 20 + 1) }
                let source = MLXArray(values, [count]).asType(dtype)
                let work = try MLXWorkingBuffer.copying(source)
                XCTAssertEqual(work.count, count, "\(dtype) x\(count)")
                XCTAssertEqual(work.byteCount, count * dtype.size, "\(dtype) x\(count)")
                XCTAssertEqual(work.isAdopted,
                               count * dtype.size >= MLXWorkingBuffer.adoptionFloor,
                               "\(dtype) x\(count) took the wrong path")
                let result = work.finish()
                XCTAssertEqual(result.dtype, dtype, "\(dtype) x\(count)")
                XCTAssertEqual(result.shape, [count], "\(dtype) x\(count)")
                XCTAssertEqual(result.floats, values, "\(dtype) x\(count)")
            }
        }
    }

    /// `zeros` is the all-gather receive path: it must produce a buffer of the
    /// right size that is genuinely zeroed, since the collective only writes
    /// the slots the peers own.
    func testZerosWorkingBufferIsSizedAndCleared() throws {
        try MLXRuntime.requireMetallib()
        for shape in [[3, 5], [2], [1]] {
            let work = try MLXWorkingBuffer.zeros(shape: shape, dtype: .float32)
            let count = shape.reduce(1, *)
            XCTAssertEqual(work.count, count, "\(shape)")
            XCTAssertEqual(work.byteCount, count * 4, "\(shape)")
            let result = work.finish()
            XCTAssertEqual(result.shape, shape)
            XCTAssertEqual(result.floats, [Float](repeating: 0, count: count), "\(shape)")
        }
    }

    /// A strided *expression* still produces a correct result, whether MLX
    /// materialises it on eval or the adapter's fallback does.
    func testWorkingBufferCopiesAStridedInputCorrectly() throws {
        try MLXRuntime.requireMetallib()
        let base = MLXArray(Array(0..<8).map { Float($0) }, [4, 2])
        let column = base[.ellipsis, 1]
        let work = try MLXWorkingBuffer.copying(column)
        XCTAssertEqual(work.count, 4)
        XCTAssertEqual(work.finish().floats, [1, 3, 5, 7])
    }
}
