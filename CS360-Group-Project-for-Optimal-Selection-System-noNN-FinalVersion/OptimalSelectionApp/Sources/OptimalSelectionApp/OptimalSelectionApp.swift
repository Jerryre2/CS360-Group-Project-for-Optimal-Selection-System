import AppKit
import Combine
import Foundation
import SwiftUI

@main
struct OptimalSelectionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1120, minHeight: 760)
        }
    }
}

struct ContentView: View {
    @StateObject private var state = AppState()

    var body: some View {
        HStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    environmentSection
                    problemSection
                    solverSection
                    exactSection
                    resultSection
                    actionSection
                }
                .padding(22)
            }
            .frame(width: 430)

            Divider()

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Command Output")
                        .font(.title3.weight(.semibold))
                    Spacer()
                    Button("Clear") {
                        state.output = ""
                    }
                    .disabled(state.isRunning)
                }

                TextEditor(text: $state.output)
                    .font(.system(.body, design: .monospaced))
                    .scrollContentBackground(.hidden)
                    .background(Color(nsColor: .textBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.25))
                    )
            }
            .padding(22)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Optimal Selection System")
                .font(.largeTitle.weight(.bold))
            Text("macOS SwiftUI wrapper for the noNN Python solver")
                .foregroundStyle(.secondary)
        }
    }

    private var environmentSection: some View {
        Panel("Environment") {
            LabeledTextField("Python", text: $state.pythonPath)
            LabeledTextField("Project", text: $state.projectPath)
            Picker("Log Level", selection: $state.logLevel) {
                ForEach(["INFO", "DEBUG", "WARNING", "ERROR"], id: \.self) {
                    Text($0).tag($0)
                }
            }
            Button("Open Results Folder") {
                state.openResultsFolder()
            }
        }
    }

    private var problemSection: some View {
        Panel("Problem Parameters") {
            twoColumnFields([
                ("m", $state.m),
                ("n", $state.n),
                ("k", $state.k),
                ("j", $state.j),
                ("s", $state.s),
                ("r", $state.r)
            ])

            LabeledTextField("Samples", text: $state.samples)

            Picker("Coverage", selection: $state.coverageMode) {
                Text("at_least_one").tag("at_least_one")
                Text("at_least_r").tag("at_least_r")
                Text("all_subsets").tag("all_subsets")
            }

            Picker("Aggregation", selection: $state.aggregationMode) {
                Text("distinct_subsets").tag("distinct_subsets")
                Text("single_candidate").tag("single_candidate")
            }
        }
    }

    private var solverSection: some View {
        Panel("Heuristic Solver") {
            twoColumnFields([
                ("Seed", $state.seed),
                ("Restarts", $state.restarts),
                ("Local Steps", $state.localSteps),
                ("SA Iterations", $state.saIterations),
                ("Candidate Sample", $state.candidateSampleSize),
                ("DB Dir", $state.dbDir)
            ])

            Toggle("Save result to DB directory", isOn: $state.saveResult)
            Toggle("Disable exact verification during solve", isOn: $state.disableILP)
        }
    }

    private var exactSection: some View {
        Panel("Exact / Bound Proof") {
            twoColumnFields([
                ("Target Size", $state.targetSize),
                ("Time Limit", $state.exactTimeLimit)
            ])

            Picker("Backend", selection: $state.exactBackend) {
                ForEach(["auto", "gurobi", "scip", "scipy"], id: \.self) {
                    Text($0).tag($0)
                }
            }

            Toggle("Force exact verification for large instances", isOn: $state.forceExact)
        }
    }

    private var resultSection: some View {
        Panel("Saved Results") {
            LabeledTextField("Filename", text: $state.resultFilename)
            LabeledTextField("JSON Path", text: $state.validationFilePath)
        }
    }

    private var actionSection: some View {
        Panel("Actions") {
            VStack(spacing: 10) {
                HStack {
                    CommandButton("Solve", isRunning: state.isRunning) {
                        state.runSolve()
                    }
                    CommandButton("Prove Bound", isRunning: state.isRunning) {
                        state.runProveBound()
                    }
                }

                HStack {
                    CommandButton("Validate All", isRunning: state.isRunning) {
                        state.runValidateResults()
                    }
                    CommandButton("List Results", isRunning: state.isRunning) {
                        state.runListResults()
                    }
                }

                HStack {
                    CommandButton("Show Result", isRunning: state.isRunning) {
                        state.runShowResult()
                    }
                    CommandButton("Validate File", isRunning: state.isRunning) {
                        state.runValidateFile()
                    }
                }

                HStack {
                    CommandButton("Delete Result", isRunning: state.isRunning) {
                        state.runDeleteResult()
                    }
                    CommandButton("Run Demo", isRunning: state.isRunning) {
                        state.runDemo()
                    }
                }
            }

            if state.isRunning {
                ProgressView("Running command...")
                    .padding(.top, 8)
            }
        }
    }

    @ViewBuilder
    private func twoColumnFields(_ rows: [(String, Binding<String>)]) -> some View {
        let pairs = stride(from: 0, to: rows.count, by: 2).map { index in
            (rows[index], index + 1 < rows.count ? rows[index + 1] : nil)
        }

        VStack(spacing: 8) {
            ForEach(Array(pairs.enumerated()), id: \.offset) { _, pair in
                HStack(spacing: 12) {
                    LabeledTextField(pair.0.0, text: pair.0.1)
                    if let second = pair.1 {
                        LabeledTextField(second.0, text: second.1)
                    } else {
                        Spacer()
                    }
                }
            }
        }
    }
}

struct Panel<Content: View>: View {
    let title: String
    let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
            content
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

struct LabeledTextField: View {
    let label: String
    @Binding var text: String

    init(_ label: String, text: Binding<String>) {
        self.label = label
        self._text = text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(label, text: $text)
                .textFieldStyle(.roundedBorder)
        }
    }
}

struct CommandButton: View {
    let title: String
    let isRunning: Bool
    let action: () -> Void

    init(_ title: String, isRunning: Bool, action: @escaping () -> Void) {
        self.title = title
        self.isRunning = isRunning
        self.action = action
    }

    var body: some View {
        Button(title, action: action)
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)
            .frame(maxWidth: .infinity)
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var pythonPath = "/opt/homebrew/anaconda3/bin/python"
    @Published var projectPath = "/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN"
    @Published var logLevel = "INFO"

    @Published var m = "45"
    @Published var n = "15"
    @Published var k = "6"
    @Published var j = "5"
    @Published var s = "4"
    @Published var r = ""
    @Published var samples = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    @Published var coverageMode = "at_least_one"
    @Published var aggregationMode = "distinct_subsets"

    @Published var seed = "42"
    @Published var restarts = "5"
    @Published var localSteps = ""
    @Published var saIterations = ""
    @Published var candidateSampleSize = "48"
    @Published var dbDir = "results_db_v3"
    @Published var saveResult = true
    @Published var disableILP = false

    @Published var targetSize = "296"
    @Published var exactBackend = "auto"
    @Published var exactTimeLimit = "600"
    @Published var forceExact = false

    @Published var resultFilename = ""
    @Published var validationFilePath = ""
    @Published var output = "Ready. Configure parameters and click an action."
    @Published var isRunning = false

    func runSolve() {
        var arguments = baseCommand("solve") + problemArguments()
        appendOption("--seed", seed, to: &arguments)
        appendOption("--restarts", restarts, to: &arguments)
        appendOption("--local-steps", localSteps, to: &arguments)
        appendOption("--sa-iterations", saIterations, to: &arguments)
        appendOption("--candidate-sample-size", candidateSampleSize, to: &arguments)
        appendOption("--db-dir", dbDir, to: &arguments)

        if saveResult {
            arguments.append("--save")
        }
        if disableILP {
            arguments.append("--disable-ilp")
        }
        appendOption("--exact-backend", exactBackend, to: &arguments)
        appendOption("--exact-time-limit", exactTimeLimit, to: &arguments)
        if forceExact {
            arguments.append("--force-exact")
        }

        run(arguments)
    }

    func runProveBound() {
        var arguments = baseCommand("prove-bound") + problemArguments()
        appendOption("--target-size", targetSize, to: &arguments)
        appendOption("--exact-backend", exactBackend, to: &arguments)
        appendOption("--exact-time-limit", exactTimeLimit, to: &arguments)
        run(arguments)
    }

    func runValidateResults() {
        var arguments = baseCommand("validate-results")
        appendOption("--db-dir", dbDir, to: &arguments)
        run(arguments)
    }

    func runListResults() {
        var arguments = baseCommand("list-results")
        appendOption("--db-dir", dbDir, to: &arguments)
        run(arguments)
    }

    func runShowResult() {
        guard !resultFilename.trimmed.isEmpty else {
            output = "Please enter a saved result filename."
            return
        }
        var arguments = baseCommand("show-result")
        arguments.append(resultFilename.trimmed)
        appendOption("--db-dir", dbDir, to: &arguments)
        run(arguments)
    }

    func runValidateFile() {
        guard !validationFilePath.trimmed.isEmpty else {
            output = "Please enter a JSON file path."
            return
        }
        var arguments = baseCommand("validate-file")
        arguments.append(validationFilePath.trimmed)
        run(arguments)
    }

    func runDeleteResult() {
        guard !resultFilename.trimmed.isEmpty else {
            output = "Please enter a saved result filename."
            return
        }
        var arguments = baseCommand("delete-result")
        arguments.append(resultFilename.trimmed)
        appendOption("--db-dir", dbDir, to: &arguments)
        run(arguments)
    }

    func runDemo() {
        var arguments = baseCommand("demo")
        appendOption("--db-dir", dbDir, to: &arguments)
        appendOption("--seed", seed, to: &arguments)
        if saveResult {
            arguments.append("--save")
        }
        run(arguments)
    }

    func openResultsFolder() {
        let path = URL(fileURLWithPath: projectPath)
            .appendingPathComponent(dbDir.trimmed.isEmpty ? "results_db_v3" : dbDir.trimmed)
        NSWorkspace.shared.open(path)
    }

    private func baseCommand(_ subcommand: String) -> [String] {
        ["-m", "optimal_samples_system", "--log-level", logLevel, subcommand]
    }

    private func problemArguments() -> [String] {
        var arguments: [String] = []
        appendOption("--m", m, to: &arguments)
        appendOption("--n", n, to: &arguments)
        appendOption("--k", k, to: &arguments)
        appendOption("--j", j, to: &arguments)
        appendOption("--s", s, to: &arguments)
        appendOption("--samples", samples, to: &arguments)
        appendOption("--coverage-mode", coverageMode, to: &arguments)
        appendOption("--aggregation-mode", aggregationMode, to: &arguments)
        appendOption("--r", r, to: &arguments)
        appendOption("--seed", seed, to: &arguments)
        return arguments
    }

    private func appendOption(_ option: String, _ value: String, to arguments: inout [String]) {
        let cleanValue = value.trimmed
        guard !cleanValue.isEmpty else {
            return
        }
        arguments.append(option)
        arguments.append(cleanValue)
    }

    private func run(_ arguments: [String]) {
        guard !pythonPath.trimmed.isEmpty else {
            output = "Python path is empty."
            return
        }
        guard !projectPath.trimmed.isEmpty else {
            output = "Project path is empty."
            return
        }

        isRunning = true
        output = "Running command...\n\n" + shellPreview(arguments: arguments)

        Task {
            let result = await ProcessRunner.run(
                pythonPath: pythonPath.trimmed,
                projectPath: projectPath.trimmed,
                arguments: arguments
            )
            output = result.formatted
            isRunning = false
        }
    }

    private func shellPreview(arguments: [String]) -> String {
        ([pythonPath.trimmed] + arguments)
            .map { $0.shellEscaped }
            .joined(separator: " ")
    }
}

struct CommandResult {
    let commandLine: String
    let exitCode: Int32
    let stdout: String
    let stderr: String
    let errorMessage: String?

    var formatted: String {
        var sections = [
            "Command:",
            commandLine,
            "",
            "Exit code: \(exitCode)"
        ]

        if let errorMessage {
            sections += ["", "Error:", errorMessage]
        }
        if !stdout.isEmpty {
            sections += ["", "STDOUT:", stdout]
        }
        if !stderr.isEmpty {
            sections += ["", "STDERR:", stderr]
        }
        return sections.joined(separator: "\n")
    }
}

enum ProcessRunner {
    static func run(
        pythonPath: String,
        projectPath: String,
        arguments: [String]
    ) async -> CommandResult {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                let commandLine = ([pythonPath] + arguments)
                    .map { $0.shellEscaped }
                    .joined(separator: " ")

                process.executableURL = URL(fileURLWithPath: pythonPath)
                process.arguments = arguments
                process.currentDirectoryURL = URL(fileURLWithPath: projectPath)
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe

                do {
                    try process.run()
                    process.waitUntilExit()
                    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                    continuation.resume(
                        returning: CommandResult(
                            commandLine: commandLine,
                            exitCode: process.terminationStatus,
                            stdout: String(data: stdoutData, encoding: .utf8) ?? "",
                            stderr: String(data: stderrData, encoding: .utf8) ?? "",
                            errorMessage: nil
                        )
                    )
                } catch {
                    continuation.resume(
                        returning: CommandResult(
                            commandLine: commandLine,
                            exitCode: -1,
                            stdout: "",
                            stderr: "",
                            errorMessage: error.localizedDescription
                        )
                    )
                }
            }
        }
    }
}

extension String {
    var trimmed: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var shellEscaped: String {
        if isEmpty {
            return "''"
        }
        if rangeOfCharacter(from: CharacterSet.whitespacesAndNewlines.union(.init(charactersIn: "'\"\\$`"))) == nil {
            return self
        }
        return "'" + replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
