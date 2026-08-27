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
        // MLX adapter. Separate product, separate target, because it is the
        // only thing in the package with an external dependency — linking
        // MCCL must never drag mlx-swift in behind it.
        .library(name: "MCCLMLX", targets: ["MCCLMLX"]),
        // Data-parallel training demo driven by the adapter. Buildable without
        // XCTest, because the lab nodes do not have it.
        .executable(name: "mccltrain", targets: ["mccltrain"]),
    ],
    dependencies: [
        // The MLX adapter's only dependency, and the package's only dependency
        // full stop. Reached exclusively from `MCCLMLX` and `mccltrain`.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
    ],
    targets: [
        .target(name: "MCCL"),
        // `include` holds the hand-written C header, which is documentation and
        // a compile-time contract for C callers — not a Swift source or resource.
        .target(name: "MCCLShim", dependencies: ["MCCL"], exclude: ["include"]),
        .target(name: "MCCLBenchmarks", dependencies: ["MCCL"]),
        .executableTarget(name: "mcclprobe", dependencies: ["MCCL"]),
        .executableTarget(name: "mcclbench", dependencies: ["MCCLBenchmarks", "MCCL"]),
        // The MLX seam. `MCCL` is not allowed to depend on this; the arrow only
        // points one way.
        .target(
            name: "MCCLMLX",
            dependencies: [
                "MCCL",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "mccltrain",
            dependencies: [
                "MCCLMLX",
                "MCCL",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MCCLTests",
            dependencies: ["MCCL", "MCCLShim", "MCCLBenchmarks"],
            // The C program that exercises the ABI is compiled by the test that
            // runs it, with `cc` against the built dylib — not by SwiftPM.
            exclude: ["CProgram"]
        ),
        // The adapter's tests live apart from `MCCLTests` on purpose: the 237
        // tests of the core library must keep running on a machine with no
        // mlx.metallib, and must not acquire an MLX dependency by association.
        .testTarget(
            name: "MCCLMLXTests",
            dependencies: [
                "MCCLMLX",
                "MCCL",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
    ]
)
