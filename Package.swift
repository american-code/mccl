// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mccl",
    platforms: [.macOS(.v14)],
    products: [
        // The collectives library itself. Infra, not a framework: MLX, llama.cpp,
        // or raw Metal apps link against this independently.
        .library(name: "MCCL", targets: ["MCCL"]),
        // The same library behind a C ABI, as libmccl.dylib. This is the artifact
        // a non-Swift runtime links against; the header is hand-written and lives
        // at Sources/MCCLShim/include/mccl.h.
        .library(name: "mccl", type: .dynamic, targets: ["MCCLShim"]),
        // Topology/bandwidth probe CLI: measures actual TB5/USB4/Ethernet link
        // bandwidth between nodes and prints the tuned topology plan.
        .executable(name: "mcclprobe", targets: ["mcclprobe"]),
        // Algorithm/codec benchmark harness. The tool that will later measure
        // mccl against MLX's built-in ring on the real cluster.
        .executable(name: "mcclbench", targets: ["mcclbench"]),
    ],
    targets: [
        .target(name: "MCCL"),
        // `include` holds the hand-written C header, which is documentation and
        // a compile-time contract for C callers — not a Swift source or resource.
        .target(name: "MCCLShim", dependencies: ["MCCL"], exclude: ["include"]),
        .target(name: "MCCLBenchmarks", dependencies: ["MCCL"]),
        .executableTarget(name: "mcclprobe", dependencies: ["MCCL"]),
        .executableTarget(name: "mcclbench", dependencies: ["MCCLBenchmarks", "MCCL"]),
        .testTarget(
            name: "MCCLTests",
            dependencies: ["MCCL", "MCCLShim", "MCCLBenchmarks"],
            // The C program that exercises the ABI is compiled by the test that
            // runs it, with `cc` against the built dylib — not by SwiftPM.
            exclude: ["CProgram"]
        ),
    ]
)
