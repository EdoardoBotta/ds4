import AppKit
import Foundation
import Metal
import SwiftUI
import UniformTypeIdentifiers

@main
struct DS4MacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 900, minHeight: 640)
        }
        .windowStyle(.titleBar)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}

private enum Defaults {
    static let agentPath = findExistingPath([
        FileManager.default.currentDirectoryPath + "/ds4-agent",
        FileManager.default.currentDirectoryPath + "/../ds4-agent"
    ]) ?? FileManager.default.currentDirectoryPath + "/../ds4-agent"

    static let modelPath = findExistingPath([
        FileManager.default.currentDirectoryPath + "/ds4flash.gguf",
        FileManager.default.currentDirectoryPath + "/../ds4flash.gguf"
    ]) ?? ""

    /// The Metal-recommended working set, in whole GiB. The slider allows the
    /// full 100%, but ds4_streaming_manual_cache_safe_bytes() in ds4.c warns
    /// (no longer silently clamps) once a request exceeds 70% of this, since
    /// that only leaves the routed expert cache room to breathe alongside
    /// non-routed weights, KV cache, scratch buffers, and macOS's own
    /// wired-memory overhead. safeCacheGiB marks that danger threshold so the
    /// slider can flag it instead of enforcing it.
    static let maxCacheGiB: Int = {
        guard let device = MTLCreateSystemDefaultDevice() else { return 128 }
        let gib: UInt64 = 1024 * 1024 * 1024
        return max(1, Int(device.recommendedMaxWorkingSetSize / gib))
    }()

    static let safeCacheGiB: Int = max(1, Int((Double(maxCacheGiB) * 0.7).rounded(.down)))

    private static func findExistingPath(_ candidates: [String]) -> String? {
        for candidate in candidates {
            let expanded = URL(fileURLWithPath: candidate).standardized.path
            if FileManager.default.fileExists(atPath: expanded) {
                return expanded
            }
        }
        return nil
    }
}

enum MessageRole {
    case assistant
    case user
}

struct ChatMessage: Identifiable {
    let id: UUID
    let role: MessageRole
    var text: String

    init(id: UUID = UUID(), role: MessageRole, text: String) {
        self.id = id
        self.role = role
        self.text = text
    }

    var isPlaceholder: Bool {
        role == .assistant && text.isEmpty
    }

    var displayText: String {
        isPlaceholder ? "Thinking..." : text
    }
}

struct ContextStatus {
    var used: Int = 0
    var total: Int = 0
    var generationTokensPerSecond: Double = 0

    var usageFraction: Double? {
        guard total > 0 else { return nil }
        return min(max(Double(used) / Double(total), 0), 1)
    }

    var usagePercentText: String {
        guard let usageFraction else { return "Context --" }
        let percent = usageFraction * 100
        return "Context \(percent.formatted(.number.precision(.fractionLength(1))))%"
    }

    var usageDetailText: String {
        guard total > 0 else { return "Context unavailable" }
        return "\(used.formatted()) / \(total.formatted()) tokens"
    }

    var generationSpeedText: String {
        if generationTokensPerSecond > 0 {
            return "\(generationTokensPerSecond.formatted(.number.precision(.fractionLength(1)))) tok/s"
        }
        return "-- tok/s"
    }
}

@MainActor
final class DS4Runner: ObservableObject {
    @Published var isGenerating = false
    @Published var isEngineRunning = false
    @Published var messages: [ChatMessage] = []
    @Published var startupDetails: [String] = []
    @Published var webApprovalMessage: String?
    @Published var status = "Idle"
    @Published var prefillProgress: PrefillProgress?
    @Published var contextStatus = ContextStatus()
    @Published var latestUserMessageID: UUID?
    @Published var latestAssistantMessageID: UUID?
    @Published var latestCompletedAssistantMessageID: UUID?

    private var agentProcess: Process?
    private var inputPipe: Pipe?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?
    private var startupOutputBuffer = ""
    private var errorBuffer = ""
    private var pendingPromptAfterInterrupt: String?
    private var currentAssistantMessageID: UUID?

    func startEngine(agentPath: String, modelPath: String, cacheGiB: Int, contextTokens: Int) {
        guard !isEngineRunning, agentProcess == nil else { return }

        let agentURL = URL(fileURLWithPath: agentPath).standardized
        guard FileManager.default.isExecutableFile(atPath: agentURL.path) else {
            status = "Invalid ds4-agent executable"
            appendSystemMessage("Build ds4-agent or choose a repository checkout that contains it.")
            return
        }

        let modelURL = URL(fileURLWithPath: modelPath).standardized
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            status = "Invalid model"
            appendSystemMessage("Choose a valid GGUF model.")
            return
        }

        let process = Process()
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()

        process.executableURL = agentURL
        process.arguments = [
            "-m", modelURL.path,
            "--non-interactive",
            "--ssd-streaming",
            "--ssd-streaming-cache-experts", "\(max(1, cacheGiB))GB",
            "--ctx", "\(max(1, contextTokens))",
            "--temp", "0"
        ]
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        process.currentDirectoryURL = agentURL.deletingLastPathComponent()

        startupDetails.removeAll()
        webApprovalMessage = nil
        prefillProgress = nil
        contextStatus = ContextStatus()
        startupOutputBuffer = ""
        errorBuffer = ""
        pendingPromptAfterInterrupt = nil
        status = "Starting agent"
        self.agentProcess = process
        self.inputPipe = inputPipe
        self.outputPipe = outputPipe
        self.errorPipe = errorPipe

        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async {
                self?.appendAssistantOutput(text)
            }
        }

        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async {
                self?.handleAgentErrorOutput(text)
            }
        }

        process.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                self?.outputPipe?.fileHandleForReading.readabilityHandler = nil
                self?.errorPipe?.fileHandleForReading.readabilityHandler = nil
                self?.agentProcess = nil
                self?.inputPipe = nil
                self?.outputPipe = nil
                self?.errorPipe = nil
                self?.isEngineRunning = false
                if self?.isGenerating == true {
                    self?.isGenerating = false
                }
                self?.prefillProgress = nil
                self?.status = process.terminationStatus == 0 ? "Agent stopped" : "Agent exited with code \(process.terminationStatus)"
            }
        }

        do {
            try process.run()
        } catch {
            outputPipe.fileHandleForReading.readabilityHandler = nil
            errorPipe.fileHandleForReading.readabilityHandler = nil
            self.agentProcess = nil
            self.inputPipe = nil
            self.outputPipe = nil
            self.errorPipe = nil
            status = "Failed to start agent"
            appendSystemMessage(error.localizedDescription)
        }
    }

    func restartEngine(agentPath: String, modelPath: String, cacheGiB: Int, contextTokens: Int) {
        stopEngine()
        Task { [weak self] in
            for _ in 0..<40 {
                if self?.agentProcess == nil {
                    self?.startEngine(agentPath: agentPath, modelPath: modelPath, cacheGiB: cacheGiB, contextTokens: contextTokens)
                    return
                }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            self?.status = "Agent restart timed out"
        }
    }

    func generate(prompt: String) -> Bool {
        guard isEngineRunning, !isGenerating else { return false }

        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            status = "Prompt required"
            appendSystemMessage("Enter a prompt before running.")
            return false
        }

        guard let inputPipe else {
            status = "Agent input unavailable"
            return false
        }

        let text = prompt.hasSuffix("\n") ? prompt : prompt + "\n"
        guard let data = text.data(using: .utf8) else {
            status = "Prompt encoding failed"
            return false
        }

        let userMessage = ChatMessage(role: .user, text: trimmed)
        let assistantMessage = ChatMessage(role: .assistant, text: "")
        messages.append(userMessage)
        messages.append(assistantMessage)
        latestUserMessageID = userMessage.id
        latestAssistantMessageID = assistantMessage.id
        currentAssistantMessageID = assistantMessage.id
        prefillProgress = nil
        status = "Agent working"
        isGenerating = true
        inputPipe.fileHandleForWriting.write(data)
        return true
    }

    func interruptGeneration(nextPrompt: String? = nil) {
        guard isGenerating, let inputPipe else { return }

        let trimmedNext = nextPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        pendingPromptAfterInterrupt = (trimmedNext?.isEmpty == false) ? nextPrompt : nil

        guard let data = "+DWARFSTAR_INTERRUPT\n".data(using: .utf8) else { return }
        inputPipe.fileHandleForWriting.write(data)
        status = "Stopping"
    }

    func stopEngine() {
        prefillProgress = nil
        inputPipe?.fileHandleForWriting.closeFile()
        agentProcess?.terminate()
    }

    func answerWebApproval(allow: Bool) {
        guard let inputPipe else {
            status = "Agent input unavailable"
            return
        }
        let command = "+DWARFSTAR_WEB_APPROVAL \(allow ? "yes" : "no")\n"
        guard let data = command.data(using: .utf8) else { return }
        inputPipe.fileHandleForWriting.write(data)
        webApprovalMessage = nil
        status = allow ? "Starting browser" : "Browser denied"
    }

    private func handleAgentErrorOutput(_ text: String) {
        errorBuffer += text
        while let newline = errorBuffer.firstIndex(of: "\n") {
            let line = String(errorBuffer[..<newline])
            errorBuffer.removeSubrange(...newline)
            handleAgentErrorLine(line)
        }
    }

    private func handleAgentErrorLine(_ value: String) {
        let webApprovalPrefix = "+DWARFSTAR_WEB_APPROVAL_REQUIRED "
        if value.hasPrefix(webApprovalPrefix) {
            let message = String(value.dropFirst(webApprovalPrefix.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            webApprovalMessage = message.isEmpty ? "Visible browser approval is required." : message
            status = "Approval required"
            return
        }

        let contextPrefix = "+DWARFSTAR_CONTEXT "
        if value.hasPrefix(contextPrefix) {
            let fields = value.split(separator: " ")
            if fields.count >= 4,
               let used = Int(fields[1]),
               let total = Int(fields[2]),
               let generationTokensPerSecond = Double(fields[3]) {
                contextStatus = ContextStatus(
                    used: used,
                    total: total,
                    generationTokensPerSecond: generationTokensPerSecond
                )
            }
            return
        }

        switch value {
        case "+DWARFSTAR_WAITING":
            isEngineRunning = true
            isGenerating = false
            prefillProgress = nil
            status = "Ready"
            latestCompletedAssistantMessageID = currentAssistantMessageID
            currentAssistantMessageID = nil
            if let pending = pendingPromptAfterInterrupt {
                pendingPromptAfterInterrupt = nil
                _ = generate(prompt: pending)
            }
        case "+DWARFSTAR_QUEUED":
            status = "Prompt queued"
        case "+DWARFSTAR_PREFILL_DONE":
            prefillProgress = nil
            if isGenerating {
                status = "Generating"
            }
        default:
            if value.hasPrefix("+DWARFSTAR_PREFILL ") {
                let fields = value.split(separator: " ")
                if fields.count >= 4,
                   let done = Int(fields[1]),
                   let total = Int(fields[2]),
                   let tokensPerSecond = Double(fields[3]) {
                    let kind = fields.count >= 5 ? String(fields[4]) : "user_prompt"
                    prefillProgress = PrefillProgress(
                        done: done,
                        total: total,
                        tokensPerSecond: tokensPerSecond,
                        kind: kind == "tool_results" ? .toolResults : .userPrompt
                    )
                    status = prefillProgress?.label ?? "Prefilling"
                }
            } else if !value.isEmpty {
                if isStartupDetail(value) {
                    appendStartupDetail(value)
                } else {
                    appendSystemMessage(value)
                }
                if !isEngineRunning {
                    status = "Starting agent"
                }
            }
        }
    }

    func clearChat() {
        messages.removeAll()
        latestUserMessageID = nil
        latestAssistantMessageID = nil
        latestCompletedAssistantMessageID = nil
        currentAssistantMessageID = nil
        prefillProgress = nil
    }

    private func appendAssistantOutput(_ text: String) {
        guard !text.isEmpty else { return }
        if currentAssistantMessageID == nil && captureStartupOutput(text) {
            return
        }

        if let id = currentAssistantMessageID,
           let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index].text += text
            return
        }

        let message = ChatMessage(role: .assistant, text: text)
        messages.append(message)
        latestAssistantMessageID = message.id
        currentAssistantMessageID = message.id
    }

    private func appendSystemMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let message = ChatMessage(role: .assistant, text: trimmed)
        messages.append(message)
        latestAssistantMessageID = message.id
        latestCompletedAssistantMessageID = message.id
    }

    private func isStartupDetail(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return !isEngineRunning && (trimmed.hasPrefix("ds4:") || trimmed.hasPrefix("ds4-agent:"))
    }

    private func appendStartupDetail(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if startupDetails.last != trimmed {
            startupDetails.append(trimmed)
        }
    }

    private func captureStartupOutput(_ text: String) -> Bool {
        startupOutputBuffer += text

        while let newline = startupOutputBuffer.firstIndex(of: "\n") {
            let line = String(startupOutputBuffer[..<newline])
            startupOutputBuffer.removeSubrange(...newline)
            if isStartupDetail(line) {
                appendStartupDetail(line)
            } else {
                appendSystemMessage(line)
            }
        }

        let trimmed = startupOutputBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if isStartupDetail(trimmed) {
            appendStartupDetail(trimmed)
            startupOutputBuffer = ""
        }

        return true
    }

}

enum PrefillKind {
    case userPrompt
    case toolResults
}

struct PrefillProgress {
    let done: Int
    let total: Int
    let tokensPerSecond: Double
    let kind: PrefillKind

    var label: String {
        switch kind {
        case .userPrompt:
            return "Reading prompt"
        case .toolResults:
            return "Reading tool results"
        }
    }

    var fraction: Double? {
        guard total > 0 else { return nil }
        return min(max(Double(done) / Double(total), 0), 1)
    }

    var detail: String {
        if total > 0 {
            let percent = 100 * (fraction ?? 0)
            if tokensPerSecond > 0 {
                return "\(done)/\(total) tokens | \(percent.formatted(.number.precision(.fractionLength(1))))% | \(tokensPerSecond.formatted(.number.precision(.fractionLength(0)))) tok/s"
            }
            return "\(done)/\(total) tokens | \(percent.formatted(.number.precision(.fractionLength(1))))%"
        }
        return "Preparing prompt"
    }
}

struct ContentView: View {
    @StateObject private var runner = DS4Runner()

    @AppStorage("agentPath") private var agentPath = Defaults.agentPath
    @AppStorage("modelPath") private var modelPath = Defaults.modelPath
    @AppStorage("prompt") private var prompt = "Write a concise explanation of cache locality in mixture-of-experts inference."
    @AppStorage("modelCacheGiB") private var modelCacheGiB: Double = 32
    @AppStorage("contextTokens") private var contextTokens: Double = 100000

    @State private var pickingModel = false
    @State private var pickingAgent = false
    @State private var showingSettings = false
    @State private var appliedCacheGiB = 0
    @State private var appliedContextTokens = 0
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            conversation
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .fileImporter(isPresented: $pickingModel, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first {
                modelPath = url.path
                restartEngineWithCurrentSettings()
            }
        }
        .fileImporter(isPresented: $pickingAgent, allowedContentTypes: [.item], allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first {
                agentPath = url.path
                restartEngineWithCurrentSettings()
            }
        }
        .onAppear {
            modelCacheGiB = min(modelCacheGiB, Double(Defaults.maxCacheGiB))
            appliedCacheGiB = Int(modelCacheGiB)
            appliedContextTokens = Int(contextTokens)
            runner.startEngine(agentPath: agentPath, modelPath: modelPath, cacheGiB: appliedCacheGiB, contextTokens: appliedContextTokens)
            promptFocused = true
        }
        .onDisappear {
            runner.stopEngine()
        }
    }

    private var header: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(
                            LinearGradient(
                                colors: [Color.accentColor, Color.accentColor.opacity(0.78)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                    Image(systemName: "sparkles")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .frame(width: 32, height: 32)

                VStack(alignment: .leading, spacing: 3) {
                    Text("DS4")
                        .font(.headline)
                    Text("Local assistant")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                StatusPill(text: runner.status, isActive: runner.isEngineRunning, isBusy: runner.isGenerating)
                MetricPill(
                    icon: "chart.pie",
                    text: runner.contextStatus.usagePercentText,
                    detail: runner.contextStatus.usageDetailText
                )
                MetricPill(
                    icon: "speedometer",
                    text: runner.contextStatus.generationSpeedText,
                    detail: "Generation speed"
                )

                Spacer()

                Button {
                    let wasOpen = showingSettings
                    withAnimation(.easeInOut(duration: 0.18)) {
                        showingSettings.toggle()
                    }
                    if wasOpen {
                        applySettingsIfNeeded()
                    }
                } label: {
                    Label("Settings", systemImage: "slider.horizontal.3")
                }
                .labelStyle(.iconOnly)
                .help("Settings")
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(.regularMaterial)

            if showingSettings {
                settingsPanel
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }

    private var toolbarButtonTitle: String {
        if runner.isGenerating {
            return "Stop"
        }
        if runner.isEngineRunning {
            return "Run"
        }
        return "Start"
    }

    private var toolbarButtonIcon: String {
        if runner.isGenerating {
            return "stop.fill"
        }
        if runner.isEngineRunning {
            return "arrow.up"
        }
        return "play.fill"
    }

    private var settingsPanel: some View {
        VStack(spacing: 14) {
            pathRow("Agent", value: $agentPath) {
                pickingAgent = true
            }

            pathRow("Model", value: $modelPath) {
                pickingModel = true
            }

            sliderRow(
                "Model cache",
                value: $modelCacheGiB,
                range: 1...Double(Defaults.maxCacheGiB),
                step: 1,
                valueText: "\(Int(modelCacheGiB)) GB",
                dangerZoneStart: Double(Defaults.safeCacheGiB)
            )

            sliderRow(
                "Context window",
                value: $contextTokens,
                range: 4096...200000,
                step: 1024,
                valueText: "\(Int(contextTokens).formatted()) tokens"
            )
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 14)
        .frame(maxWidth: 920)
        .frame(maxWidth: .infinity)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    private var conversation: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 22) {
                        if !runner.startupDetails.isEmpty {
                            StartupDetailsBox(lines: runner.startupDetails)
                                .id("startup-details")
                        }

                        if let message = runner.webApprovalMessage {
                            WebApprovalNoticeBox(
                                message: message,
                                onAllow: {
                                    runner.answerWebApproval(allow: true)
                                },
                                onDeny: {
                                    runner.answerWebApproval(allow: false)
                                }
                            )
                                .id("web-approval")
                        }

                        if runner.messages.isEmpty && !runner.isGenerating {
                            emptyState
                                .id("empty")
                        } else {
                            ForEach(runner.messages) { message in
                                MessageBubble(
                                    role: message.role,
                                    text: message.displayText,
                                    isPlaceholder: message.isPlaceholder
                                )
                                .id(message.id)
                            }

                            if let progress = runner.prefillProgress {
                                prefillProgressView(progress)
                                    .id("prefill")
                            }
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 26)
                    .padding(.bottom, 34)
                    .frame(maxWidth: 920)
                    .frame(maxWidth: .infinity)
                }
                .background(chatBackground)
                .onChange(of: runner.latestUserMessageID) { _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: runner.latestAssistantMessageID) { _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: runner.latestCompletedAssistantMessageID) { _ in
                    scrollToBottom(proxy)
                }
                .onChange(of: runner.startupDetails.count) { _ in
                    if runner.messages.isEmpty {
                        scrollToStartupDetails(proxy)
                    }
                }
                .onChange(of: runner.webApprovalMessage) { _ in
                    scrollToWebApproval(proxy)
                }
            }

            Divider()

            composer
        }
    }

    private var chatBackground: Color {
        Color(nsColor: .textBackgroundColor)
            .opacity(NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? 0.72 : 1)
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .frame(width: 68, height: 68)
                Image(systemName: "sparkles")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
            }
            Text("Ask DS4 anything")
                .font(.title3.weight(.semibold))
            Text(runner.isEngineRunning ? "Ready for a new prompt." : "Agent offline.")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: 420)
        .frame(maxWidth: .infinity, minHeight: 320)
    }

    private var composer: some View {
        VStack(spacing: 8) {
            ZStack(alignment: .topLeading) {
                ChatInputEditor(
                    text: $prompt,
                    isFocused: Binding(
                        get: { promptFocused },
                        set: { promptFocused = $0 }
                    ),
                    onSubmit: runCurrentPrompt
                )
                .padding(.horizontal, 4)
                .padding(.vertical, 2)

                if prompt.isEmpty {
                    Text("Message DS4")
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 10)
                        .allowsHitTesting(false)
                }
            }
            .frame(minHeight: 58, maxHeight: 126)

            HStack(spacing: 8) {
                Button {
                    prompt = ""
                    promptFocused = true
                } label: {
                    Label("Clear prompt", systemImage: "xmark")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.plain)
                .help("Clear prompt")
                .disabled(prompt.isEmpty || runner.isGenerating)

                Button {
                    runner.clearChat()
                    promptFocused = true
                } label: {
                    Label("Clear chat", systemImage: "trash")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.plain)
                .help("Clear chat")
                .disabled(runner.isGenerating || runner.messages.isEmpty)

                Spacer()

                Text(composerStatus)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)

                Button {
                    handlePrimaryAction()
                } label: {
                    Image(systemName: toolbarButtonIcon)
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 30, height: 30)
                        .background(
                            Circle()
                                .fill(primaryActionDisabled ? Color.secondary.opacity(0.45) : Color.accentColor)
                        )
                }
                .buttonStyle(.plain)
                .disabled(primaryActionDisabled)
                .keyboardShortcut(.return, modifiers: [.command])
                .help(toolbarButtonTitle)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(nsColor: .controlBackgroundColor))
                .shadow(color: Color.black.opacity(promptFocused ? 0.12 : 0.06), radius: promptFocused ? 16 : 9, x: 0, y: 5)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(promptFocused ? Color.accentColor.opacity(0.65) : Color(nsColor: .separatorColor), lineWidth: promptFocused ? 1.5 : 1)
        )
        .padding(.horizontal, 24)
        .padding(.top, 14)
        .padding(.bottom, 16)
        .frame(maxWidth: 920)
        .frame(maxWidth: .infinity)
        .background(.regularMaterial)
    }

    private func prefillProgressView(_ progress: PrefillProgress) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(progress.label)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(progress.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }

            if let fraction = progress.fraction {
                ProgressView(value: fraction)
            } else {
                ProgressView()
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .frame(maxWidth: 680, alignment: .leading)
    }

    private func pathRow(_ label: String, value: Binding<String>, picker: @escaping () -> Void) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .leading)

            TextField(label, text: value)
                .textFieldStyle(.roundedBorder)
                .font(.system(.caption, design: .monospaced))

            Button {
                picker()
            } label: {
                Label("Choose \(label)", systemImage: "folder")
            }
            .labelStyle(.iconOnly)
            .help("Choose \(label.lowercased())")
        }
    }

    private func sliderRow(
        _ label: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        valueText: String,
        dangerZoneStart: Double? = nil
    ) -> some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .leading)

            if let dangerZoneStart {
                ZoneSlider(value: value, range: range, step: step, dangerZoneStart: dangerZoneStart)
                    .frame(height: 20)
            } else {
                Slider(value: value, in: range, step: step)
            }

            Text(valueText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(dangerZoneStart.map { value.wrappedValue >= $0 } == true ? Color.red : Color.secondary)
                .frame(width: 90, alignment: .trailing)
        }
    }

    private var composerStatus: String {
        if runner.isGenerating {
            return runner.status
        }
        if !runner.isEngineRunning {
            return "Offline"
        }
        return "Ready"
    }

    private var primaryActionDisabled: Bool {
        if runner.isGenerating {
            return false
        }
        if runner.isEngineRunning {
            return prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return false
    }

    private func handlePrimaryAction() {
        if runner.isGenerating {
            runner.interruptGeneration(nextPrompt: prompt)
            prompt = ""
            promptFocused = true
        } else if runner.isEngineRunning {
            runCurrentPrompt()
        } else {
            runner.startEngine(agentPath: agentPath, modelPath: modelPath, cacheGiB: appliedCacheGiB, contextTokens: appliedContextTokens)
        }
    }

    private func runCurrentPrompt() {
        if runner.generate(prompt: prompt) {
            prompt = ""
            promptFocused = true
        }
    }

    private func restartEngineWithCurrentSettings() {
        appliedCacheGiB = Int(modelCacheGiB)
        appliedContextTokens = Int(contextTokens)
        runner.restartEngine(agentPath: agentPath, modelPath: modelPath, cacheGiB: appliedCacheGiB, contextTokens: appliedContextTokens)
    }

    private func applySettingsIfNeeded() {
        guard Int(modelCacheGiB) != appliedCacheGiB || Int(contextTokens) != appliedContextTokens else { return }
        restartEngineWithCurrentSettings()
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.16)) {
                if let id = runner.latestCompletedAssistantMessageID ??
                    runner.latestAssistantMessageID ??
                    runner.latestUserMessageID {
                    proxy.scrollTo(id, anchor: .bottom)
                } else {
                    proxy.scrollTo("empty", anchor: .bottom)
                }
            }
        }
    }

    private func scrollToStartupDetails(_ proxy: ScrollViewProxy) {
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.16)) {
                proxy.scrollTo("startup-details", anchor: .top)
            }
        }
    }

    private func scrollToWebApproval(_ proxy: ScrollViewProxy) {
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.16)) {
                proxy.scrollTo("web-approval", anchor: .center)
            }
        }
    }
}

private struct WebApprovalNoticeBox: View {
    let message: String
    let onAllow: () -> Void
    let onDeny: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.orange)
                .frame(width: 24, height: 24)

            VStack(alignment: .leading, spacing: 6) {
                Text("Web approval required")
                    .font(.subheadline.weight(.semibold))
                Text(message)
                    .font(.body)
                    .textSelection(.enabled)
                Text("Google search needs permission to start a headless browser session.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 8) {
                    Button {
                        onAllow()
                    } label: {
                        Label("Allow", systemImage: "checkmark")
                    }
                    .buttonStyle(.borderedProminent)

                    Button {
                        onDeny()
                    } label: {
                        Label("Deny", systemImage: "xmark")
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.top, 4)
            }

            Spacer()
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.orange.opacity(0.10))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.orange.opacity(0.35), lineWidth: 1)
        )
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct StartupDetailsBox: View {
    let lines: [String]

    private var details: StartupDetails {
        StartupDetails(lines: lines)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "cpu")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(
                        RoundedRectangle(cornerRadius: 9)
                            .fill(Color.accentColor)
                    )
                Text("Agent startup")
                    .font(.headline)
                Spacer()
            }

            LazyVGrid(
                columns: [
                    GridItem(.adaptive(minimum: 190), spacing: 10, alignment: .topLeading)
                ],
                alignment: .leading,
                spacing: 10
            ) {
                ForEach(details.items) { item in
                    StartupInfoTile(item: item)
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.accentColor.opacity(0.09))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.accentColor.opacity(0.32), lineWidth: 1.5)
        )
        .shadow(color: Color.black.opacity(0.08), radius: 12, x: 0, y: 5)
        .frame(maxWidth: 760, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct StartupInfoItem: Identifiable {
    let id: String
    let title: String
    let value: String
    let detail: String?
    let icon: String
}

private struct StartupDetails {
    var items: [StartupInfoItem] = []

    init(lines: [String]) {
        if let parsed = Self.parseMetalDevice(lines) {
            items.append(StartupInfoItem(
                id: "device",
                title: "Device",
                value: parsed.name,
                detail: parsed.ram,
                icon: "cpu"
            ))
        }

        items.append(StartupInfoItem(
            id: "cache",
            title: "Model cache memory",
            value: Self.parseExpertCacheMemory(lines) ?? "None",
            detail: nil,
            icon: "memorychip"
        ))

        if let ctx = Self.parseContextLength(lines) {
            items.append(StartupInfoItem(
                id: "context",
                title: "Context length",
                value: Self.withThousandsSeparator(ctx),
                detail: nil,
                icon: "square.stack.3d.up"
            ))
        }

        let streamingEnabled = lines.contains { $0.contains("SSD streaming mode enabled") }
        items.append(StartupInfoItem(
            id: "mode",
            title: "SSD streaming mode",
            value: streamingEnabled ? "Enabled" : "Disabled",
            detail: streamingEnabled ? "Model streamed from disk" : "Model fully resident in memory",
            icon: streamingEnabled ? "externaldrive" : "bolt.horizontal"
        ))
    }

    private static func parseMetalDevice(_ lines: [String]) -> (name: String, ram: String?)? {
        guard let line = lines.first(where: { $0.hasPrefix("ds4: Metal device ") }) else { return nil }
        let body = String(line.dropFirst("ds4: Metal device ".count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = body.split(separator: ",", maxSplits: 1).map {
            String($0).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return (parts.first ?? body, parts.count > 1 ? parts[1] : nil)
    }

    private static func parseContextLength(_ lines: [String]) -> String? {
        guard let line = lines.first(where: { $0.hasPrefix("ds4-agent: context buffers ") }) else { return nil }
        let prefix = "ds4-agent: context buffers "
        let rest = String(line.dropFirst(prefix.count))
        guard let paren = rest.firstIndex(of: "(") else { return nil }
        let options = rest[paren...]
            .trimmingCharacters(in: CharacterSet(charactersIn: "()"))
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        var values: [String: String] = [:]
        for option in options {
            let pair = option.split(separator: "=", maxSplits: 1).map(String.init)
            if pair.count == 2 {
                values[pair[0]] = pair[1]
            }
        }
        return values["ctx"] ?? "-"
    }

    private static func parseExpertCacheMemory(_ lines: [String]) -> String? {
        if let line = lines.last(where: { $0.contains("SSD streaming cache budget") && $0.contains("GiB") }) {
            return Self.gibValue(from: line)
        }
        if let line = lines.last(where: { $0.contains("cached expert count") && $0.contains("GiB") }) {
            return Self.gibValue(from: line)
        }
        return nil
    }

    private static func gibValue(from line: String) -> String? {
        guard let range = line.range(of: "GiB") else { return nil }
        var numberChars: [Character] = []
        for ch in line[..<range.lowerBound].reversed() {
            if ch.isNumber || ch == "." {
                numberChars.append(ch)
            } else if ch == " " && numberChars.isEmpty {
                continue
            } else {
                break
            }
        }
        guard !numberChars.isEmpty else { return nil }
        return "\(String(numberChars.reversed())) GiB"
    }

    private static func withThousandsSeparator(_ raw: String) -> String {
        guard let value = Int(raw) else { return raw }
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.usesGroupingSeparator = true
        formatter.groupingSeparator = ","
        formatter.groupingSize = 3
        return formatter.string(from: NSNumber(value: value)) ?? raw
    }
}

private struct StartupInfoTile: View {
    let item: StartupInfoItem

    var body: some View {
        HStack(alignment: .top, spacing: 9) {
            Image(systemName: item.icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Color.accentColor)
                .frame(width: 24, height: 24)

            VStack(alignment: .leading, spacing: 3) {
                Text(item.title)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(item.value)
                    .font(.system(.body, design: .rounded).weight(.semibold))
                    .textSelection(.enabled)
                    .lineLimit(2)
                if let detail = item.detail, !detail.isEmpty {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, minHeight: 82, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .textBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
    }
}

/// A slider whose track is permanently split into a normal zone and a red
/// danger zone starting at `dangerZoneStart`, so the risky portion of the
/// range (e.g. cache sizes that leave little memory for other apps) stays
/// visible regardless of where the thumb currently sits.
private struct ZoneSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let dangerZoneStart: Double

    private let thumbDiameter: CGFloat = 14
    private let trackHeight: CGFloat = 4

    private func fraction(of v: Double) -> CGFloat {
        let span = range.upperBound - range.lowerBound
        guard span > 0 else { return 0 }
        return CGFloat((v - range.lowerBound) / span)
    }

    var body: some View {
        GeometryReader { geo in
            let usableWidth = max(0, geo.size.width - thumbDiameter)
            let dangerX = fraction(of: dangerZoneStart) * usableWidth
            let thumbX = fraction(of: value) * usableWidth

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.accentColor.opacity(0.35))
                    .frame(width: dangerX, height: trackHeight)
                    .offset(x: thumbDiameter / 2)

                Capsule()
                    .fill(Color.red.opacity(0.55))
                    .frame(width: max(0, usableWidth - dangerX), height: trackHeight)
                    .offset(x: thumbDiameter / 2 + dangerX)

                Circle()
                    .fill(Color.white)
                    .overlay(Circle().stroke(Color(nsColor: .separatorColor), lineWidth: 0.5))
                    .shadow(color: .black.opacity(0.25), radius: 1, y: 0.5)
                    .frame(width: thumbDiameter, height: thumbDiameter)
                    .offset(x: thumbX)
            }
            .frame(height: max(thumbDiameter, trackHeight))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        guard usableWidth > 0 else { return }
                        let x = min(max(0, drag.location.x - thumbDiameter / 2), usableWidth)
                        let rawValue = range.lowerBound + Double(x / usableWidth) * (range.upperBound - range.lowerBound)
                        let stepped = step > 0 ? (rawValue / step).rounded() * step : rawValue
                        value = min(max(stepped, range.lowerBound), range.upperBound)
                    }
            )
        }
        .frame(height: max(thumbDiameter, trackHeight))
    }
}

private struct StatusPill: View {
    let text: String
    let isActive: Bool
    let isBusy: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(dotColor)
                .frame(width: 7, height: 7)

            Text(text)
                .font(.caption.weight(.medium))
                .lineLimit(1)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            Capsule()
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            Capsule()
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
    }

    private var dotColor: Color {
        if isBusy {
            return .orange
        }
        if isActive {
            return .green
        }
        return .secondary
    }
}

private struct MetricPill: View {
    let icon: String
    let text: String
    let detail: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 11, weight: .semibold))
            Text(text)
                .font(.caption.weight(.medium))
                .monospacedDigit()
                .lineLimit(1)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(
            Capsule()
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            Capsule()
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
        .help(detail)
    }
}

private struct ChatInputEditor: NSViewRepresentable {
    @Binding var text: String
    @Binding var isFocused: Bool
    let onSubmit: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text, isFocused: $isFocused)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.borderType = .noBorder
        scrollView.autohidesScrollers = true

        let textView = SubmitTextView()
        textView.delegate = context.coordinator
        textView.onSubmit = onSubmit
        textView.string = text
        textView.font = .preferredFont(forTextStyle: .body)
        textView.textColor = .labelColor
        textView.drawsBackground = false
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainerInset = NSSize(width: 0, height: 7)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        textView.autoresizingMask = [.width]
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? SubmitTextView else { return }
        textView.onSubmit = onSubmit
        if textView.string != text {
            textView.string = text
        }
        if isFocused && textView.window?.firstResponder !== textView {
            textView.window?.makeFirstResponder(textView)
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        @Binding private var text: String
        @Binding private var isFocused: Bool

        init(text: Binding<String>, isFocused: Binding<Bool>) {
            _text = text
            _isFocused = isFocused
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            text = textView.string
        }

        func textDidBeginEditing(_ notification: Notification) {
            isFocused = true
        }

        func textDidEndEditing(_ notification: Notification) {
            isFocused = false
        }
    }
}

private final class SubmitTextView: NSTextView {
    var onSubmit: (() -> Void)?

    override func keyDown(with event: NSEvent) {
        let isReturn = event.keyCode == 36 || event.keyCode == 76
        let insertsNewline = event.modifierFlags.intersection(.deviceIndependentFlagsMask).contains(.shift)

        if isReturn && !insertsNewline {
            onSubmit?()
            return
        }

        super.keyDown(with: event)
    }
}

private struct MessageBubble: View {
    let role: MessageRole
    let text: String
    var isPlaceholder = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if role == .user {
                Spacer(minLength: 80)
            } else {
                avatar
            }

            messageText
                .frame(maxWidth: role == .user ? 620 : 720, alignment: role == .user ? .trailing : .leading)

            if role == .assistant {
                Spacer(minLength: 80)
            }
        }
        .frame(maxWidth: .infinity, alignment: role == .user ? .trailing : .leading)
    }

    private var messageText: some View {
        Group {
            if role == .assistant && !isPlaceholder {
                AssistantMessageContent(text: text)
            } else {
                Text(text)
                    .font(.body)
                    .foregroundStyle(foregroundStyle)
                    .textSelection(.enabled)
                    .lineSpacing(4)
            }
        }
        .padding(.horizontal, horizontalPadding)
        .padding(.vertical, verticalPadding)
        .background(background)
        .clipShape(RoundedRectangle(cornerRadius: role == .user ? 16 : 10))
    }

    private var avatar: some View {
        ZStack {
            Circle()
                .fill(Color.accentColor.opacity(0.12))
            Image(systemName: "sparkles")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color.accentColor)
        }
        .frame(width: 28, height: 28)
    }

    private var foregroundStyle: Color {
        if role == .user {
            return .white
        }
        if isPlaceholder {
            return Color(nsColor: .secondaryLabelColor)
        }
        return Color(nsColor: .labelColor)
    }

    private var background: Color {
        if role == .user {
            return .accentColor
        }
        if isPlaceholder {
            return Color(nsColor: .controlBackgroundColor)
        }
        return .clear
    }

    private var horizontalPadding: CGFloat {
        role == .user || isPlaceholder ? 14 : 0
    }

    private var verticalPadding: CGFloat {
        role == .user || isPlaceholder ? 11 : 0
    }
}

private enum AssistantBlock: Identifiable {
    case text(id: UUID, value: String)
    case tool(id: UUID, call: ToolCallSummary)

    var id: UUID {
        switch self {
        case .text(let id, _), .tool(let id, _):
            return id
        }
    }
}

private struct ToolCallSummary {
    let group: ToolGroup
    let title: String
    let detail: String?
}

private enum ToolGroup {
    case fileRead
    case fileWrite
    case web
    case shell
    case localSearch
    case other

    var icon: String {
        switch self {
        case .fileRead:
            return "doc.text.magnifyingglass"
        case .fileWrite:
            return "square.and.pencil"
        case .web:
            return "globe"
        case .shell:
            return "terminal"
        case .localSearch:
            return "magnifyingglass"
        case .other:
            return "wrench.and.screwdriver"
        }
    }

    var label: String {
        switch self {
        case .fileRead:
            return "File read"
        case .fileWrite:
            return "File change"
        case .web:
            return "Web"
        case .shell:
            return "Shell"
        case .localSearch:
            return "Search"
        case .other:
            return "Tool"
        }
    }

    var tint: Color {
        switch self {
        case .fileRead:
            return .blue
        case .fileWrite:
            return .green
        case .web:
            return .cyan
        case .shell:
            return .purple
        case .localSearch:
            return .orange
        case .other:
            return .secondary
        }
    }
}

private struct AssistantMessageContent: View {
    let text: String

    private var blocks: [AssistantBlock] {
        AssistantMessageParser.blocks(from: text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(blocks) { block in
                switch block {
                case .text(_, let value):
                    MarkdownContentView(text: value)
                case .tool(_, let call):
                    ToolCallRow(call: call)
                }
            }
        }
    }
}

private enum AssistantMessageParser {
    static func blocks(from text: String) -> [AssistantBlock] {
        var blocks: [AssistantBlock] = []
        var textBuffer: [String] = []

        func flushText() {
            let value = textBuffer.joined(separator: "\n")
            textBuffer.removeAll()
            guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
            blocks.append(.text(id: UUID(), value: value))
        }

        for line in text.components(separatedBy: .newlines) {
            if let call = parseToolLine(line) {
                flushText()
                blocks.append(.tool(id: UUID(), call: call))
            } else if isIncompleteToolLine(line) {
                flushText()
            } else {
                textBuffer.append(line)
            }
        }

        flushText()
        return blocks
    }

    private static func isIncompleteToolLine(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed == "🛠️" || trimmed.hasPrefix("🛠️ ")
    }

    private static func parseToolLine(_ line: String) -> ToolCallSummary? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("🛠️ ") else { return nil }
        let content = String(trimmed.dropFirst("🛠️ ".count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return nil }

        if content.hasPrefix("Reading ") {
            let detail = String(content.dropFirst("Reading ".count))
            return ToolCallSummary(group: .fileRead, title: "Reading file", detail: detail)
        }
        if content.hasPrefix("read ") {
            return ToolCallSummary(group: .fileRead, title: "Reading file", detail: String(content.dropFirst(5)))
        }
        if content.hasPrefix("write ") {
            return ToolCallSummary(group: .fileWrite, title: "Writing file", detail: String(content.dropFirst(6)))
        }
        if content.hasPrefix("edit ") {
            return ToolCallSummary(group: .fileWrite, title: "Editing file", detail: String(content.dropFirst(5)))
        }
        if content.hasPrefix("google ") {
            return ToolCallSummary(group: .web, title: "Searching web", detail: String(content.dropFirst(7)))
        }
        if content.hasPrefix("visit ") {
            return ToolCallSummary(group: .web, title: "Visiting page", detail: String(content.dropFirst(6)))
        }
        if content.hasPrefix("search ") {
            return ToolCallSummary(group: .localSearch, title: "Searching workspace", detail: String(content.dropFirst(7)))
        }
        if content.hasPrefix("$ ") {
            return ToolCallSummary(group: .shell, title: "Running command", detail: String(content.dropFirst(2)))
        }

        return nil
    }
}

/// Renders a block of assistant prose as markdown: fenced code blocks get a
/// monospaced box, headings/lists/blockquotes get their own layout, and
/// everything else is parsed for inline styling (bold, italic, code spans,
/// links) via AttributedString. Deliberately not a full CommonMark
/// implementation -- just the subset that shows up in day-to-day model output.
private struct MarkdownContentView: View {
    let text: String

    private var blocks: [MarkdownBlock] {
        MarkdownParser.parse(text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(blocks) { block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .paragraph(_, let text):
            Text(text)
                .font(.body)
                .foregroundStyle(Color(nsColor: .labelColor))
                .textSelection(.enabled)
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)

        case .heading(_, let level, let text):
            Text(text)
                .font(headingFont(level))
                .foregroundStyle(Color(nsColor: .labelColor))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)

        case .bulletList(_, let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .top, spacing: 8) {
                        Text("•")
                            .font(.body)
                            .foregroundStyle(.secondary)
                        Text(item)
                            .font(.body)
                            .foregroundStyle(Color(nsColor: .labelColor))
                            .textSelection(.enabled)
                            .lineSpacing(4)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

        case .numberedList(_, let items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, entry in
                    HStack(alignment: .top, spacing: 8) {
                        Text("\(entry.number).")
                            .font(.body.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Text(entry.text)
                            .font(.body)
                            .foregroundStyle(Color(nsColor: .labelColor))
                            .textSelection(.enabled)
                            .lineSpacing(4)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

        case .blockquote(_, let text):
            HStack(spacing: 8) {
                Rectangle()
                    .fill(Color.secondary.opacity(0.4))
                    .frame(width: 3)
                Text(text)
                    .font(.body.italic())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            }

        case .codeBlock(_, let language, let code):
            MarkdownCodeBlockView(language: language, code: code)

        case .rule:
            Divider()

        case .table(_, let header, let alignments, let rows):
            MarkdownTableView(header: header, alignments: alignments, rows: rows)
        }
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title2.weight(.semibold)
        case 2: return .title3.weight(.semibold)
        default: return .headline
        }
    }
}

private struct MarkdownCodeBlockView: View {
    let language: String?
    let code: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let language, !language.isEmpty {
                Text(language)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.top, 8)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(Color(nsColor: .labelColor))
                    .textSelection(.enabled)
                    .padding(10)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
    }
}

private struct MarkdownTableView: View {
    let header: [AttributedString]
    let alignments: [HorizontalAlignment]
    let rows: [[AttributedString]]

    private func alignment(for column: Int) -> HorizontalAlignment {
        column < alignments.count ? alignments[column] : .leading
    }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .topLeading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    ForEach(Array(header.enumerated()), id: \.offset) { column, cell in
                        Text(cell)
                            .font(.body.weight(.semibold))
                            .foregroundStyle(Color(nsColor: .labelColor))
                            .textSelection(.enabled)
                            .gridColumnAlignment(alignment(for: column))
                    }
                }

                Divider()
                    .gridCellColumns(max(header.count, 1))

                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            Text(cell)
                                .font(.body)
                                .foregroundStyle(Color(nsColor: .labelColor))
                                .textSelection(.enabled)
                        }
                    }
                }
            }
            .padding(12)
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
    }
}

enum MarkdownBlock: Identifiable {
    case paragraph(id: UUID, text: AttributedString)
    case heading(id: UUID, level: Int, text: AttributedString)
    case bulletList(id: UUID, items: [AttributedString])
    case numberedList(id: UUID, items: [(number: Int, text: AttributedString)])
    case blockquote(id: UUID, text: AttributedString)
    case codeBlock(id: UUID, language: String?, code: String)
    case rule(id: UUID)
    case table(id: UUID, header: [AttributedString], alignments: [HorizontalAlignment], rows: [[AttributedString]])

    var id: UUID {
        switch self {
        case .paragraph(let id, _), .heading(let id, _, _), .bulletList(let id, _),
             .numberedList(let id, _), .blockquote(let id, _), .codeBlock(let id, _, _), .rule(let id),
             .table(let id, _, _, _):
            return id
        }
    }
}

private struct ToolCallRow: View {
    let call: ToolCallSummary

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: call.group.icon)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(call.group.tint)
                .frame(width: 28, height: 28)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(call.group.tint.opacity(0.12))
                )

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(call.group.label)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(call.group.tint)
                    Text(call.title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Color(nsColor: .labelColor))
                }

                if let detail = call.detail, !detail.isEmpty {
                    Text(detail)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
            }

            Spacer(minLength: 0)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
        .frame(maxWidth: 620, alignment: .leading)
    }
}
