// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DS4MacApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "DS4MacApp", targets: ["DS4MacApp"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-markdown.git", revision: "f339d8afd712d1e2e207719b163d2003f7a91a36")
    ],
    targets: [
        .executableTarget(
            name: "DS4MacApp",
            dependencies: [
                .product(name: "Markdown", package: "swift-markdown")
            ],
            path: "Sources/DS4MacApp"
        )
    ]
)
