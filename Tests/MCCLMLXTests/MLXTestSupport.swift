import Foundation
import MCCL
import MLX
import XCTest

// Support for the MLX adapter's tests.
//
// Two things are different here from `MCCLTests`, and both are consequences of
// the adapter rather than of the tests.
//
// **Ranks run on real threads, not on a task group.** The MLX-facing API is
// synchronous: it parks the calling thread on a semaphore while the collective
// runs (`runBlocking`). Two ranks of a loopback world block on each other, so
// driving them from Swift concurrency would occupy two cooperative threads with
// a mutual wait. `Thread` is the honest tool for a blocking API.
//
// **Every test that touches MLX needs the Metal shader library.** mlx-swift does
// not build `mlx.metallib` under plain SwiftPM, and MLX's failure to find one is
// fatal rather than throwable — it cannot be caught and turned into a test
// failure. So the presence of the file is checked *before* any MLX call, and a
// missing one is a skip with the command that fixes it.

enum MLXRuntime {

    /// Directories MLX will search for `mlx.metallib`, in the order it searches.
    ///
    /// MLX resolves the library relative to the binary that contains the MLX
    /// code (`dladdr` on one of its own symbols). With SwiftPM's static linking
    /// that is the test bundle's executable, so its containing directory is the
    /// one that matters — `.build/<config>/mcclPackageTests.xctest/Contents/MacOS`.
    /// The `.build/<config>` directory is checked too, because that is where the
    /// executables (`mccltrain`) look.
    static var searchDirectories: [URL] {
        var directories: [URL] = []
        func add(_ url: URL?) {
            guard let url else { return }
            if !directories.contains(url) { directories.append(url) }
        }
        if let override = ProcessInfo.processInfo.environment["MCCL_MLX_METALLIB_DIR"] {
            add(URL(fileURLWithPath: override))
        }
        add(Bundle.main.executableURL?.deletingLastPathComponent())
        for bundle in Bundle.allBundles {
            add(bundle.executableURL?.deletingLastPathComponent())
            add(bundle.resourceURL)
        }
        return directories
    }

    static var metallibPath: URL? {
        searchDirectories
            .map { $0.appendingPathComponent("mlx.metallib") }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Call at the top of every test that touches MLX.
    static func requireMetallib() throws {
        guard metallibPath == nil else { return }
        throw XCTSkip("""
            mlx.metallib is not installed, so MLX cannot run a single GPU op and \
            this test would abort the process rather than fail.

            Install it with:    Tools/fetch-metallib.sh

            mlx-swift compiles the MLX C++ core but does not build the Metal \
            shader library — that is Xcode's job, and these tests are meant to \
            run under plain `swift test`. The script fetches the version-matched \
            library from the mlx-metal pip wheel. Searched:
            \(searchDirectories.map { "  " + $0.path }.joined(separator: "\n"))
            """)
    }
}

/// Runs `body` on every rank of an N-rank in-process world, each on its own
/// thread, and waits for all of them.
enum MLXRanks {
    static func run(
        worldSize: Int,
        file: StaticString = #filePath,
        line: UInt = #line,
        body: @escaping @Sendable (Communicator) throws -> Void
    ) throws {
        let comms = try Communicator.loopbackGroup(worldSize: worldSize)
        defer { comms.forEach { $0.shutdown() } }

        let failures = FailureBox()
        let group = DispatchGroup()
        for comm in comms {
            group.enter()
            let thread = Thread {
                defer { group.leave() }
                do { try body(comm) } catch { failures.record(comm.rank, error) }
            }
            // The default 512 KiB is enough, but the ring's scratch buffers are
            // sized from the message, so be explicit rather than lucky.
            thread.stackSize = 4 << 20
            thread.start()
        }
        guard group.wait(timeout: .now() + 120) == .success else {
            XCTFail("world of \(worldSize) did not finish within 120s", file: file, line: line)
            return
        }
        for (rank, error) in failures.all {
            XCTFail("rank \(rank): \(error)", file: file, line: line)
        }
    }
}

final class FailureBox: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [(Int, Error)] = []

    func record(_ rank: Int, _ error: Error) {
        lock.lock(); storage.append((rank, error)); lock.unlock()
    }
    var all: [(Int, Error)] { lock.lock(); defer { lock.unlock() }; return storage }
}

/// Thread-safe per-rank result collection.
final class RankResults<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int: T] = [:]

    func set(_ rank: Int, _ value: T) { lock.lock(); values[rank] = value; lock.unlock() }
    func get(_ rank: Int) -> T? { lock.lock(); defer { lock.unlock() }; return values[rank] }
}

extension MLXArray {
    /// Contents as `[Float]` whatever the dtype — the comparison currency for
    /// these tests, since bfloat16 has no Swift type to compare against.
    var floats: [Float] { asType(.float32).asArray(Float.self) }
}
