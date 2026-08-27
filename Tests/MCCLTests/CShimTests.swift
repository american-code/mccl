import XCTest
@testable import MCCL
@testable import MCCLShim

/// The C ABI.
///
/// The load-bearing test here is `testCProgramLinksAndPasses`: it compiles
/// `CProgram/mccl_smoke.c` with `cc` against the built `libmccl.dylib` and runs
/// it. That is the only check that proves three separate things at once — that
/// `mccl.h` is valid C, that its declarations match the symbols the dylib
/// actually exports, and that a non-Swift process can link and run the library.
/// A Swift test calling `@_cdecl` functions directly would prove none of them.
///
/// The rest of this file covers the mapping tables and the argument validation
/// that the C program can only sample.
final class CShimTests: XCTestCase {

    // MARK: - The real thing

    func testCProgramLinksAndPasses() throws {
        let compiler = URL(fileURLWithPath: "/usr/bin/cc")
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: compiler.path),
                          "no /usr/bin/cc on this machine")

        let buildDirectory = Self.buildDirectory
        let library = buildDirectory.appendingPathComponent("libmccl.dylib")
        XCTAssertTrue(FileManager.default.fileExists(atPath: library.path),
                      "expected the dynamic product at \(library.path) — is 'mccl' still a "
                      + ".dynamic library product in Package.swift?")

        let source = Self.packageRoot
            .appendingPathComponent("Tests/MCCLTests/CProgram/mccl_smoke.c")
        let headers = Self.packageRoot.appendingPathComponent("Sources/MCCLShim/include")
        let binary = FileManager.default.temporaryDirectory
            .appendingPathComponent("mccl_smoke_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: binary) }

        let compile = try Self.run(compiler, [
            "-std=c11", "-Wall", "-Werror", "-O1",
            "-I", headers.path,
            source.path,
            "-L", buildDirectory.path, "-lmccl",
            "-Wl,-rpath,\(buildDirectory.path)",
            "-o", binary.path,
        ])
        XCTAssertEqual(compile.status, 0,
                       "cc failed to build the C client against mccl.h:\n\(compile.output)")

        let execution = try Self.run(binary, [])
        XCTAssertEqual(execution.status, 0, "the C client failed:\n\(execution.output)")
        XCTAssertTrue(execution.output.contains("smoke test: OK"),
                      "unexpected output:\n\(execution.output)")
    }

    /// The header is the published contract; a symbol it declares but the dylib
    /// does not export would only fail at a customer's link step.
    func testHeaderDeclaresOnlySymbolsTheLibraryExports() throws {
        let header = try String(contentsOf: Self.packageRoot
            .appendingPathComponent("Sources/MCCLShim/include/mccl.h"), encoding: .utf8)
        let exported = try Self.exportedSymbols()

        // Everything of the form `mcclX(` that is not a `static inline`.
        var declared: Set<String> = []
        for line in header.split(separator: "\n") {
            guard !line.contains("static inline"), line.contains("mccl"), line.contains("(") else { continue }
            guard line.hasPrefix("mcclResult_t ") || line.hasPrefix("const char* mccl") else { continue }
            guard let open = line.firstIndex(of: "("),
                  let start = line[..<open].lastIndex(of: " ") else { continue }
            let name = String(line[line.index(after: start)..<open])
            if name.hasPrefix("mccl") { declared.insert(name) }
        }

        XCTAssertFalse(declared.isEmpty, "failed to parse any declarations out of mccl.h")
        XCTAssertTrue(declared.contains("mcclAllReduce"))
        XCTAssertTrue(declared.contains("mcclCommInitRankFromId"))
        for name in declared.sorted() {
            XCTAssertTrue(exported.contains(name),
                          "mccl.h declares \(name) but libmccl.dylib does not export it")
        }
    }

    // MARK: - Result mapping

    func testEveryErrorMapsToItsOwnResultCode() {
        let errors: [MCCLError] = [
            .notImplemented("x"), .invalidArgument("x"), .rankOutOfRange(rank: 9, worldSize: 2),
            .noFabric, .bufferTooSmall(need: 2, have: 1), .connectionClosed,
            .protocolViolation("x"), .socketFailure("x", errno: 1), .timedOut("x"),
            .unsupportedCompression("x"), .topologyInvalid("x"),
        ]
        let codes = errors.map { ResultCode($0).rawValue }
        XCTAssertEqual(Set(codes).count, errors.count, "two errors share a result code")
        XCTAssertFalse(codes.contains(ResultCode.success.rawValue), "a failure must never map to success")
        // The values cross the ABI boundary: pin them.
        XCTAssertEqual(ResultCode(.invalidArgument("x")).rawValue, 2)
        XCTAssertEqual(ResultCode(.noFabric).rawValue, 4)
        XCTAssertEqual(ResultCode(.unsupportedCompression("x")).rawValue, 10)
    }

    func testErrorStringsAreStableAndSurviveTheCall() {
        for code in 0..<13 {
            let pointer = MCCLShim.mcclGetErrorString(Int32(code))
            let name = String(cString: pointer)
            XCTAssertTrue(name.hasPrefix("mccl"), "result \(code) is named '\(name)'")
            // Interned: the same code must hand back the same pointer, because
            // the C contract is that it stays valid indefinitely.
            XCTAssertEqual(pointer, MCCLShim.mcclGetErrorString(Int32(code)))
        }
        XCTAssertEqual(String(cString: MCCLShim.mcclGetErrorString(0)), "mcclSuccess")
        XCTAssertTrue(String(cString: MCCLShim.mcclGetErrorString(9999)).contains("9999"))
    }

    func testDataTypeCodesMatchTheWireDiscriminator() throws {
        // mcclDataType_t is DataType.wireCode. If these ever drift, a C caller
        // and a Swift caller would disagree about what is in a frame.
        for type in [DataType.float32, .float16, .bfloat16, .int32, .int8] {
            XCTAssertEqual(try MCCLShim.dataType(Int32(type.wireCode)), type)
        }
        XCTAssertThrowsError(try MCCLShim.dataType(99))
    }

    func testReduceOpAndCompressionMapping() throws {
        XCTAssertEqual(try MCCLShim.reduceOp(0), .sum)
        XCTAssertEqual(try MCCLShim.reduceOp(4), .avg)
        XCTAssertThrowsError(try MCCLShim.reduceOp(-1))

        XCTAssertEqual(try MCCLShim.compression(0, blockSize: 0, fraction: 0), .none)
        XCTAssertEqual(try MCCLShim.compression(1, blockSize: 0, fraction: 0), .downcast)
        XCTAssertEqual(try MCCLShim.compression(2, blockSize: 0, fraction: 0),
                       .int8Blockwise(blockSize: 256), "blockSize 0 selects the default")
        XCTAssertEqual(try MCCLShim.compression(2, blockSize: 64, fraction: 0),
                       .int8Blockwise(blockSize: 64))
        XCTAssertEqual(try MCCLShim.compression(3, blockSize: 0, fraction: 0.05),
                       .topK(fraction: 0.05))
        XCTAssertThrowsError(try MCCLShim.compression(7, blockSize: 0, fraction: 0))
    }

    func testStreamHandlesCarryTheirIdentity() {
        XCTAssertEqual(MCCLShim.streamID(nil), .default)
        XCTAssertEqual(MCCLShim.streamID(UnsafeMutableRawPointer(bitPattern: 7)), StreamID(7))
        XCTAssertNotEqual(MCCLShim.streamID(UnsafeMutableRawPointer(bitPattern: 7)), StreamID(8))
    }

    // MARK: - Buffer contract

    func testInPlaceAndOutOfPlaceBuffers() throws {
        let count = 8
        var source = [Float](repeating: 3, count: count)
        var target = [Float](repeating: 0, count: count)

        try source.withUnsafeMutableBytes { send throws in
            try target.withUnsafeMutableBytes { recv throws in
                // Different pointers: input is copied, not aliased.
                let buffer = try MCCLShim.destination(
                    send: UnsafeRawPointer(send.baseAddress!), recv: recv.baseAddress!,
                    byteCount: count * 4, label: "t")
                XCTAssertEqual(buffer.baseAddress, recv.baseAddress)
                XCTAssertEqual(buffer.count, count * 4)

                // Same pointer: in place, no copy attempted.
                let inPlace = try MCCLShim.destination(
                    send: UnsafeRawPointer(recv.baseAddress!), recv: recv.baseAddress!,
                    byteCount: count * 4, label: "t")
                XCTAssertEqual(inPlace.baseAddress, recv.baseAddress)
            }
        }
        XCTAssertEqual(target, [Float](repeating: 3, count: count),
                       "the caller's sendbuff must land in recvbuff untouched")

        XCTAssertThrowsError(try MCCLShim.destination(
            send: nil, recv: nil, byteCount: 16, label: "t"))
        XCTAssertThrowsError(try MCCLShim.destination(
            send: nil, recv: UnsafeMutableRawPointer(bitPattern: 0x1000), byteCount: 16, label: "t"))
    }

    func testCStringCopyRefusesToOverflow() {
        let out = UnsafeMutablePointer<CChar>.allocate(capacity: 8)
        out.initialize(repeating: 0, count: 8)
        defer { out.deallocate() }

        XCTAssertNoThrow(try MCCLShim.copyCString("abc", into: out, size: 8))
        XCTAssertEqual(String(cString: out), "abc")
        // 8 characters need 9 bytes with the terminator.
        XCTAssertThrowsError(try MCCLShim.copyCString("12345678", into: out, size: 8)) {
            guard case MCCLError.bufferTooSmall(let need, let have) = $0 else {
                return XCTFail("expected .bufferTooSmall, got \($0)")
            }
            XCTAssertEqual(need, 9)
            XCTAssertEqual(have, 8)
        }
        XCTAssertThrowsError(try MCCLShim.copyCString("abc", into: nil, size: 8))
    }

    func testGuardedTurnsThrowsIntoCodesAndRecordsDetail() {
        let slot = ErrorSlot()
        XCTAssertEqual(guarded(slot) {}, 0)
        XCTAssertEqual(String(cString: slot.pointer), "", "success must not invent an error")

        let code = guarded(slot) { throw MCCLError.rankOutOfRange(rank: 9, worldSize: 2) }
        XCTAssertEqual(code, ResultCode.rankOutOfRange.rawValue)
        XCTAssertTrue(String(cString: slot.pointer).contains("rank 9"),
                      "the detail string should carry the offending value")

        struct Other: Error {}
        XCTAssertEqual(guarded(slot) { throw Other() }, ResultCode.internalError.rawValue)
    }

    func testRunBlockingPropagatesValuesAndErrors() throws {
        let counter = Collector<Int>()
        try runBlocking { counter.set(0, 42) }
        XCTAssertEqual(counter.get(0), 42, "the closure must have finished before the call returned")

        XCTAssertThrowsError(try runBlocking { throw MCCLError.timedOut("x") }) {
            guard case MCCLError.timedOut = $0 else { return XCTFail("wrong error: \($0)") }
        }
    }

    // MARK: - Exported entry points, called directly

    func testExportedEntryPointsRejectNullArguments() {
        XCTAssertEqual(MCCLShim.mcclGetVersion(nil), ResultCode.invalidArgument.rawValue)
        XCTAssertEqual(MCCLShim.mcclCommCount(nil, nil), ResultCode.invalidArgument.rawValue)
        XCTAssertEqual(MCCLShim.mcclCommUserRank(nil, nil), ResultCode.invalidArgument.rawValue)
        XCTAssertEqual(MCCLShim.mcclCommDestroy(nil), ResultCode.success.rawValue,
                       "destroying NULL is a documented no-op")
        XCTAssertEqual(MCCLShim.mcclGetUniqueId(nil), ResultCode.invalidArgument.rawValue)
        XCTAssertEqual(MCCLShim.mcclUniqueIdFromString(nil, nil), ResultCode.invalidArgument.rawValue)
        XCTAssertEqual(MCCLShim.mcclAllReduce(nil, nil, 4, 0, 0, nil, nil),
                       ResultCode.invalidArgument.rawValue)
    }

    func testVersionMatchesTheHeader() throws {
        var version: Int32 = 0
        XCTAssertEqual(MCCLShim.mcclGetVersion(&version), 0)
        XCTAssertEqual(Int(version), MCCLShimVersion.code)

        let header = try String(contentsOf: Self.packageRoot
            .appendingPathComponent("Sources/MCCLShim/include/mccl.h"), encoding: .utf8)
        XCTAssertTrue(header.contains("#define MCCL_MAJOR \(MCCLShimVersion.major)"))
        XCTAssertTrue(header.contains("#define MCCL_MINOR \(MCCLShimVersion.minor)"))
        XCTAssertTrue(header.contains("#define MCCL_PATCH \(MCCLShimVersion.patch)"))
    }

    // MARK: - Helpers

    static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)          // Tests/MCCLTests/CShimTests.swift
            .deletingLastPathComponent()         // Tests/MCCLTests
            .deletingLastPathComponent()         // Tests
            .deletingLastPathComponent()         // package root
    }

    /// The directory SwiftPM put this test bundle in — the same one holding
    /// libmccl.dylib, whatever configuration or triple is in play.
    static var buildDirectory: URL {
        Bundle(for: CShimTests.self).bundleURL.deletingLastPathComponent()
    }

    static func exportedSymbols() throws -> Set<String> {
        let library = buildDirectory.appendingPathComponent("libmccl.dylib")
        let result = try run(URL(fileURLWithPath: "/usr/bin/nm"), ["-gU", library.path])
        var names: Set<String> = []
        for line in result.output.split(separator: "\n") {
            let fields = line.split(separator: " ")
            guard let last = fields.last, last.hasPrefix("_mccl") else { continue }
            names.insert(String(last.dropFirst()))
        }
        return names
    }

    static func run(_ executable: URL, _ arguments: [String]) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        // Drain before waiting: a pipe that fills would deadlock the child.
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }
}
