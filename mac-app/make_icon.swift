#!/usr/bin/env swift
// Generates DS4MacApp.iconset with all required PNG sizes.
// Usage: swift mac-app/make_icon.swift <iconset_dir>
import AppKit

_ = NSApplication.shared  // initialize Cocoa before drawing

func renderIcon(pixels: CGFloat) -> NSImage {
    NSImage(size: NSSize(width: pixels, height: pixels), flipped: false) { rect in
        guard let ctx = NSGraphicsContext.current?.cgContext else { return false }

        let corner = pixels * 0.225
        ctx.addPath(CGPath(roundedRect: rect, cornerWidth: corner,
                           cornerHeight: corner, transform: nil))
        ctx.clip()

        let colors = [CGColor(red: 0.25, green: 0.55, blue: 1.00, alpha: 1.0),
                      CGColor(red: 0.08, green: 0.33, blue: 0.88, alpha: 1.0)] as CFArray
        let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(),
                                  colors: colors, locations: [0.0, 1.0])!
        ctx.drawLinearGradient(gradient,
                               start: CGPoint(x: 0, y: pixels),
                               end: CGPoint(x: pixels, y: 0),
                               options: [])

        let palette = NSImage.SymbolConfiguration(paletteColors: [.white])
        let size = NSImage.SymbolConfiguration(pointSize: pixels * 0.50, weight: .semibold)
        if let sym = NSImage(systemSymbolName: "sparkles", accessibilityDescription: nil)?
            .withSymbolConfiguration(palette.applying(size)) {
            let s = sym.size
            sym.draw(in: NSRect(x: (pixels - s.width) / 2, y: (pixels - s.height) / 2,
                                width: s.width, height: s.height),
                     from: .zero, operation: .sourceOver, fraction: 1.0)
        }
        return true
    }
}

func writePNG(_ image: NSImage, to path: String) {
    guard let tiff = image.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let png = rep.representation(using: .png, properties: [:]) else {
        fputs("error: failed to render \(path)\n", stderr); exit(1)
    }
    do { try png.write(to: URL(fileURLWithPath: path)) }
    catch { fputs("error: \(error)\n", stderr); exit(1) }
}

let args = CommandLine.arguments
let dir = args.count > 1 ? args[1] : "DS4MacApp.iconset"
try! FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

let sizes: [(name: String, pixels: CGFloat)] = [
    ("icon_16x16",      16),
    ("icon_16x16@2x",   32),
    ("icon_32x32",      32),
    ("icon_32x32@2x",   64),
    ("icon_128x128",    128),
    ("icon_128x128@2x", 256),
    ("icon_256x256",    256),
    ("icon_256x256@2x", 512),
    ("icon_512x512",    512),
    ("icon_512x512@2x", 1024),
]

for entry in sizes {
    writePNG(renderIcon(pixels: entry.pixels), to: "\(dir)/\(entry.name).png")
}
print("iconset written to \(dir)")
