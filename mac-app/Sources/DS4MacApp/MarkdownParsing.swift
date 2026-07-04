import Foundation
import Markdown
import SwiftUI

/// Parses assistant message text into `MarkdownBlock`s using Apple's swift-markdown
/// (a CommonMark/GFM parser built on cmark-gfm), then hands the resulting tree to a
/// hand-written converter. swift-markdown only parses -- it has no notion of SwiftUI --
/// so all rendering decisions (which font a heading gets, how a table looks) still live
/// in DS4MacApp.swift's MarkdownContentView/MarkdownTableView/MarkdownCodeBlockView.
enum MarkdownParser {
    static func parse(_ text: String) -> [MarkdownBlock] {
        let document = Document(parsing: text)
        var converter = BlockConverter()
        return converter.visit(document)
    }
}

/// Walks top-level block nodes and converts each into a `MarkdownBlock`. Container
/// blocks that aren't handled explicitly (documents, unrecognized wrappers) fall back
/// to `defaultVisit`, which just concatenates whatever its children produce.
private struct BlockConverter: MarkupVisitor {
    typealias Result = [MarkdownBlock]

    mutating func defaultVisit(_ markup: Markup) -> [MarkdownBlock] {
        markup.children.flatMap { visit($0) }
    }

    mutating func visitParagraph(_ paragraph: Paragraph) -> [MarkdownBlock] {
        [.paragraph(id: UUID(), text: InlineConverter.text(from: paragraph))]
    }

    mutating func visitHeading(_ heading: Heading) -> [MarkdownBlock] {
        [.heading(id: UUID(), level: heading.level, text: InlineConverter.text(from: heading))]
    }

    mutating func visitThematicBreak(_ thematicBreak: ThematicBreak) -> [MarkdownBlock] {
        [.rule(id: UUID())]
    }

    mutating func visitCodeBlock(_ codeBlock: CodeBlock) -> [MarkdownBlock] {
        let language = codeBlock.language?.trimmingCharacters(in: .whitespaces)
        let code = codeBlock.code.trimmingCharacters(in: .newlines)
        return [.codeBlock(id: UUID(), language: (language?.isEmpty ?? true) ? nil : language, code: code)]
    }

    mutating func visitBlockQuote(_ blockQuote: BlockQuote) -> [MarkdownBlock] {
        let paragraphs = blockQuote.children.compactMap { $0 as? Paragraph }
        var text = AttributedString()
        for (index, paragraph) in paragraphs.enumerated() {
            if index > 0 { text += AttributedString(" ") }
            text += InlineConverter.text(from: paragraph)
        }
        return [.blockquote(id: UUID(), text: text)]
    }

    mutating func visitUnorderedList(_ unorderedList: UnorderedList) -> [MarkdownBlock] {
        let items = unorderedList.listItems.map { Self.listItemText($0) }
        return [.bulletList(id: UUID(), items: Array(items))]
    }

    mutating func visitOrderedList(_ orderedList: OrderedList) -> [MarkdownBlock] {
        var number = orderedList.startIndex
        var items: [(number: Int, text: AttributedString)] = []
        for item in orderedList.listItems {
            items.append((number: Int(number), text: Self.listItemText(item)))
            number += 1
        }
        return [.numberedList(id: UUID(), items: items)]
    }

    mutating func visitTable(_ table: Markdown.Table) -> [MarkdownBlock] {
        let header = Array(table.head.cells.map { InlineConverter.text(from: $0) })
        let alignments = table.columnAlignments.map { alignment -> HorizontalAlignment in
            switch alignment {
            case .center: return .center
            case .right: return .trailing
            case .left, .none: return .leading
            }
        }
        let rows = table.body.rows.map { row in
            Array(row.cells.map { InlineConverter.text(from: $0) })
        }
        return [.table(id: UUID(), header: header, alignments: alignments, rows: Array(rows))]
    }

    /// A list item's content is normally a single paragraph, but it can contain
    /// nested blocks (sub-lists, nested quotes). MarkdownBlock's list cases are
    /// flat, so -- matching the app's prior hand-rolled parser -- only the
    /// item's leading paragraph is rendered; nested block content is dropped.
    private static func listItemText(_ item: ListItem) -> AttributedString {
        guard let paragraph = item.children.first(where: { $0 is Paragraph }) as? Paragraph else {
            return AttributedString()
        }
        return InlineConverter.text(from: paragraph)
    }
}

/// Flattens the inline children of a block (paragraph, heading, table cell, ...)
/// into a single AttributedString, translating emphasis/strong/code/strikethrough
/// into `inlinePresentationIntent` and links into `.link` -- the same attributes
/// SwiftUI's `Text` already knows how to style, so no manual font/color mapping
/// is needed here.
private struct InlineConverter: MarkupVisitor {
    typealias Result = AttributedString

    static func text(from markup: Markup) -> AttributedString {
        var converter = InlineConverter()
        return markup.children.reduce(into: AttributedString()) { result, child in
            result += converter.visit(child)
        }
    }

    mutating func defaultVisit(_ markup: Markup) -> AttributedString {
        markup.children.reduce(into: AttributedString()) { result, child in
            result += visit(child)
        }
    }

    mutating func visitText(_ text: Markdown.Text) -> AttributedString {
        AttributedString(text.string)
    }

    mutating func visitInlineCode(_ inlineCode: InlineCode) -> AttributedString {
        var result = AttributedString(inlineCode.code)
        result.inlinePresentationIntent = .code
        return result
    }

    mutating func visitEmphasis(_ emphasis: Emphasis) -> AttributedString {
        applying(.emphasized, to: defaultVisit(emphasis))
    }

    mutating func visitStrong(_ strong: Strong) -> AttributedString {
        applying(.stronglyEmphasized, to: defaultVisit(strong))
    }

    mutating func visitStrikethrough(_ strikethrough: Strikethrough) -> AttributedString {
        applying(.strikethrough, to: defaultVisit(strikethrough))
    }

    mutating func visitLink(_ link: Markdown.Link) -> AttributedString {
        var result = defaultVisit(link)
        guard let destination = link.destination, let url = URL(string: destination) else { return result }
        for run in result.runs {
            result[run.range].link = url
        }
        return result
    }

    mutating func visitSoftBreak(_ softBreak: SoftBreak) -> AttributedString {
        AttributedString(" ")
    }

    mutating func visitLineBreak(_ lineBreak: LineBreak) -> AttributedString {
        AttributedString("\n")
    }

    /// Merges a new inline intent into every run of `input` without clobbering
    /// intents already applied by an enclosing node (e.g. **_bold italic_**).
    private func applying(_ intent: InlinePresentationIntent, to input: AttributedString) -> AttributedString {
        var result = input
        for run in input.runs {
            let existing = run.inlinePresentationIntent ?? []
            result[run.range].inlinePresentationIntent = existing.union(intent)
        }
        return result
    }
}
