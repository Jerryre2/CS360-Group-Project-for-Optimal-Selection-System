// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "OptimalSelectionApp",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "OptimalSelectionApp",
            targets: ["OptimalSelectionApp"]
        )
    ],
    targets: [
        .executableTarget(
            name: "OptimalSelectionApp"
        )
    ]
)
