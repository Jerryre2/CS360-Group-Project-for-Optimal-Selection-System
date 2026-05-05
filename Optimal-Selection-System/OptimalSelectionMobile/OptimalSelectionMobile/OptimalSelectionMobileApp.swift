import Foundation
import SwiftUI

@main
struct OptimalSelectionMobileApp: App {
    var body: some Scene {
        WindowGroup {
            MobileRootView()
        }
    }
}

struct MobileRootView: View {
    @StateObject private var store = MobileAppStore()

    var body: some View {
        TabView {
            SolveView(store: store)
                .tabItem {
                    Label("Solve", systemImage: "bolt.fill")
                }

            ProofView(store: store)
                .tabItem {
                    Label("Proof", systemImage: "checkmark.seal.fill")
                }

            ValidateView(store: store)
                .tabItem {
                    Label("Validate", systemImage: "checklist")
                }

            ResultsView(store: store)
                .tabItem {
                    Label("Results", systemImage: "tray.full.fill")
                }
        }
        .tint(.orange)
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.95, green: 0.97, blue: 1.0),
                    Color(red: 1.0, green: 0.97, blue: 0.93)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        )
    }
}

@MainActor
final class MobileAppStore: ObservableObject {
    @Published var serverURL = "http://127.0.0.1:8000"

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
    @Published var candidateSampleSize = "48"
    @Published var localSteps = ""
    @Published var saIterations = ""
    @Published var useILP = true
    @Published var saveResult = true
    @Published var dbDir = "results_db_v3"
    @Published var exactBackend = "auto"
    @Published var solveExactTimeLimit = "60"
    @Published var proofExactTimeLimit = "300"
    @Published var forceExact = false
    @Published var adaptiveNeighborhoods = true
    @Published var reducedExactPolish = true
    @Published var reducedExactTimeLimit = "8"
    @Published var reducedExactCoreCap = "256"

    @Published var targetSize = "296"
    @Published var validationFilePath = ""
    @Published var resultFilename = ""

    @Published var isBusy = false
    @Published var statusLine = "Ready"
    @Published var logs = "Start the Python bridge, then tap Solve or Proof."
    @Published var latestResult: SolverResultPayload?
    @Published var latestProof: ProofPayload?
    @Published var validationOutput = ""
    @Published var validationReports: [ValidationReportPayload] = []
    @Published var resultItems: [ResultListItem] = []
    @Published var selectedResult: SolverResultPayload?
    @Published var latestCertification: CertificationResponse?
    @Published var demoExamples: [DemoExample] = []

    private var client: MobileAPIClient {
        MobileAPIClient(baseURL: serverURL)
    }

    func loadDemoExamples() async {
        await runTask("Loading presets") { [self] in
            let response = try await self.client.fetchDemos()
            self.demoExamples = response.examples
            if let first = response.examples.first {
                self.applyPreset(first)
            }
        }
    }

    func applyPreset(_ example: DemoExample) {
        let problem = example.problem
        m = String(problem.m)
        n = String(problem.n)
        k = String(problem.k)
        j = String(problem.j)
        s = String(problem.s)
        samples = problem.samples.map(String.init).joined(separator: ",")
        coverageMode = problem.coverageMode
        aggregationMode = problem.aggregationMode
        seed = problem.seed.map(String.init) ?? seed
        r = problem.requiredR.map(String.init) ?? ""
        statusLine = "Loaded preset: \(example.title)"
    }

    func solve() async {
        await runTask("Running solver") { [self] in
            let response = try await self.client.solve(
                problem: self.buildProblemRequest(),
                solver: self.buildSolverRequest()
            )
            self.latestResult = response.result
            self.selectedResult = response.result
            self.latestCertification = nil
            self.logs = response.logs
            self.statusLine = "Solved: \(response.result.numGroups) groups"
        }
    }

    func proveBound() async {
        await runTask("Running exact proof") { [self] in
            let response = try await self.client.proveBound(
                problem: self.buildProblemRequest(),
                targetSize: Int(self.targetSize.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0,
                exactBackend: self.exactBackend,
                exactTimeLimit: Int(self.proofExactTimeLimit) ?? 300
            )
            self.latestProof = response.proof
            self.logs = response.logs
            self.statusLine = "Proof: \(response.proof.status)"
        }
    }

    func validateAll() async {
        await runTask("Validating saved results") { [self] in
            let response = try await self.client.validateResults(dbDir: self.dbDir)
            self.validationReports = response.reports
            self.validationOutput = response.formatted.joined(separator: "\n\n")
            self.statusLine = "Validated \(response.reports.count) result files"
        }
    }

    func auditAll() async {
        await runTask("Auditing saved results") { [self] in
            let response = try await self.client.auditResults(dbDir: self.dbDir)
            self.validationOutput = response.formatted.joined(separator: "\n\n")
            self.statusLine = "Audited \(response.formatted.count) result files"
        }
    }

    func validateFile() async {
        guard !validationFilePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusLine = "Enter a JSON file path first"
            return
        }
        await runTask("Validating file") { [self] in
            let response = try await self.client.validateFile(path: self.validationFilePath)
            self.validationReports = [response.report]
            self.validationOutput = response.formatted
            self.statusLine = response.report.isValid ? "File is valid" : "File is invalid"
        }
    }

    func auditFile() async {
        guard !validationFilePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusLine = "Enter a JSON file path first"
            return
        }
        await runTask("Auditing file") { [self] in
            let response = try await self.client.auditFile(path: self.validationFilePath)
            self.validationOutput = response.formatted
            self.statusLine = "Audit completed"
        }
    }

    func certifySavedResult() async {
        guard !resultFilename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusLine = "Enter or select a result filename first"
            return
        }
        await runTask("Certifying saved result") { [self] in
            let response = try await self.client.certifyResult(
                filename: self.resultFilename,
                dbDir: self.dbDir,
                exactBackend: self.exactBackend,
                exactTimeLimit: Int(self.proofExactTimeLimit) ?? 300
            )
            self.latestCertification = response
            self.validationOutput = response.summaryText + "\n\n" + response.formatted
            self.statusLine = "Certification: \(response.report.status)"
        }
    }

    func refreshResults() async {
        await runTask("Loading saved results") { [self] in
            let response = try await self.client.listResults(dbDir: self.dbDir)
            self.resultItems = response.items
            self.statusLine = "Loaded \(response.items.count) saved results"
        }
    }

    func loadResult(_ item: ResultListItem) async {
        await runTask("Loading result detail") { [self] in
            let response = try await self.client.showResult(filename: item.filename, dbDir: self.dbDir)
            self.selectedResult = response.result
            self.resultFilename = item.filename
            self.latestCertification = nil
            self.statusLine = "Loaded \(item.filename)"
        }
    }

    func deleteSelectedResult() async {
        guard !resultFilename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            statusLine = "Select or enter a filename first"
            return
        }
        await runTask("Deleting result") { [self] in
            _ = try await self.client.deleteResult(filename: self.resultFilename, dbDir: self.dbDir)
            self.selectedResult = nil
            self.resultFilename = ""
            self.latestCertification = nil
            await self.refreshResults()
            self.statusLine = "Result deleted"
        }
    }

    private func buildProblemRequest() -> ProblemRequest {
        ProblemRequest(
            m: Int(m) ?? 45,
            n: Int(n) ?? 15,
            k: Int(k) ?? 6,
            j: Int(j) ?? 5,
            s: Int(s) ?? 4,
            samples: samples
                .split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) },
            coverageMode: coverageMode,
            aggregationMode: aggregationMode,
            requiredR: Int(r),
            seed: Int(seed)
        )
    }

    private func buildSolverRequest() -> SolverRequest {
        SolverRequest(
            nRestarts: Int(restarts) ?? 5,
            useILP: useILP,
            exactBackend: exactBackend,
            exactTimeLimit: Int(solveExactTimeLimit) ?? 60,
            forceExact: forceExact,
            maxLocalSteps: Int(localSteps),
            maxSAIterations: Int(saIterations),
            candidateSampleSize: Int(candidateSampleSize) ?? 48,
            adaptiveNeighborhoods: adaptiveNeighborhoods,
            reducedExactPolish: reducedExactPolish,
            reducedExactTimeLimit: Int(reducedExactTimeLimit) ?? 8,
            reducedExactCoreCap: Int(reducedExactCoreCap) ?? 256,
            saveResult: saveResult,
            dbDir: dbDir
        )
    }

    private func runTask(_ label: String, work: @escaping () async throws -> Void) async {
        isBusy = true
        statusLine = label
        do {
            try await work()
        } catch {
            statusLine = "Error"
            logs = "\(label) failed:\n\(error.localizedDescription)"
        }
        isBusy = false
    }
}

struct SolveView: View {
    @ObservedObject var store: MobileAppStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    HeroCard(
                        title: "iPhone Solver Console",
                        subtitle: "Run the existing Python solver from a phone-first interface in Xcode."
                    )

                    section("Connection") {
                        LabeledField("API Base URL", text: $store.serverURL)
                        Text("Use `http://127.0.0.1:8000` for Simulator. For a physical iPhone, replace this with your Mac's LAN IP.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    section("Preset") {
                        if store.demoExamples.isEmpty {
                            Button("Load Demo Presets") {
                                Task { await store.loadDemoExamples() }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(store.isBusy)
                        } else {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 10) {
                                    ForEach(store.demoExamples) { demo in
                                        Button(demo.title) {
                                            store.applyPreset(demo)
                                        }
                                        .buttonStyle(.bordered)
                                    }
                                }
                            }
                        }
                    }

                    problemSection
                    solverSection
                    resultSection
                    logsSection
                }
                .padding(16)
            }
            .navigationTitle("Optimal Selection")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if store.isBusy {
                        ProgressView()
                    }
                }
            }
        }
    }

    private var problemSection: some View {
        section("Problem") {
            NumericGrid(fields: [
                ("m", $store.m),
                ("n", $store.n),
                ("k", $store.k),
                ("j", $store.j),
                ("s", $store.s),
                ("r", $store.r)
            ])
            LabeledField("Samples", text: $store.samples)

            Picker("Coverage", selection: $store.coverageMode) {
                Text("at_least_one").tag("at_least_one")
                Text("at_least_r").tag("at_least_r")
                Text("all_subsets").tag("all_subsets")
            }
            .pickerStyle(.segmented)

            Picker("Aggregation", selection: $store.aggregationMode) {
                Text("distinct_subsets").tag("distinct_subsets")
                Text("single_candidate").tag("single_candidate")
            }
            .pickerStyle(.segmented)

            LabeledField("Seed", text: $store.seed)
        }
    }

    private var solverSection: some View {
        section("Solver") {
            NumericGrid(fields: [
                ("Restarts", $store.restarts),
                ("Candidate Sample", $store.candidateSampleSize),
                ("Local Steps", $store.localSteps),
                ("SA Iterations", $store.saIterations),
                ("Solve Exact Time", $store.solveExactTimeLimit),
            ])
            NumericGrid(fields: [
                ("Reduced Exact Time", $store.reducedExactTimeLimit),
                ("Reduced Core Cap", $store.reducedExactCoreCap),
            ])
            LabeledField("DB Dir", text: $store.dbDir)

            Picker("Exact Backend", selection: $store.exactBackend) {
                Text("auto").tag("auto")
                Text("gurobi").tag("gurobi")
                Text("scip").tag("scip")
                Text("scipy").tag("scipy")
            }
            .pickerStyle(.segmented)

            Toggle("Use exact verification during solve", isOn: $store.useILP)
            Toggle("Save result JSON", isOn: $store.saveResult)
            Toggle("Force exact on larger instances", isOn: $store.forceExact)
            Toggle("Adaptive neighborhoods", isOn: $store.adaptiveNeighborhoods)
            Toggle("Reduced-core exact polish", isOn: $store.reducedExactPolish)

            Button {
                Task { await store.solve() }
            } label: {
                Label("Run Solve", systemImage: "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(store.isBusy)
        }
    }

    private var resultSection: some View {
        section("Latest Result") {
            if let result = store.latestResult {
                ResultSummaryCard(result: result)
            } else {
                EmptyState(text: "No result yet. Run the solver to populate this panel.")
            }
        }
    }

    private var logsSection: some View {
        section("Logs") {
            Text(store.statusLine)
                .font(.headline)
            ScrollView {
                Text(store.logs)
                    .font(.system(.footnote, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .frame(minHeight: 180)
            .background(Color.black.opacity(0.92))
            .foregroundStyle(.green)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.title3.weight(.semibold))
            content()
        }
        .padding(16)
        .background(.white.opacity(0.88))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .shadow(color: .black.opacity(0.05), radius: 12, y: 6)
    }
}

struct ProofView: View {
    @ObservedObject var store: MobileAppStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    HeroCard(
                        title: "Bound Proof",
                        subtitle: "Use Gurobi, SCIP, or SciPy to test whether a smaller feasible family exists."
                    )

                    formSection
                    proofSection
                    logsSection
                }
                .padding(16)
            }
            .navigationTitle("Proof")
        }
    }

    private var formSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Proof Settings")
                .font(.title3.weight(.semibold))
            LabeledField("API Base URL", text: $store.serverURL)
            NumericGrid(fields: [
                ("Target Size", $store.targetSize),
                ("Proof Exact Time", $store.proofExactTimeLimit),
                ("Seed", $store.seed),
            ])
            LabeledField("DB Dir", text: $store.dbDir)
            Picker("Exact Backend", selection: $store.exactBackend) {
                Text("auto").tag("auto")
                Text("gurobi").tag("gurobi")
                Text("scip").tag("scip")
                Text("scipy").tag("scipy")
            }
            .pickerStyle(.segmented)

            Button {
                Task { await store.proveBound() }
            } label: {
                Label("Run Bound Proof", systemImage: "checkmark.shield.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(store.isBusy)
        }
        .padding(16)
        .background(.white.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 18))
    }

    private var proofSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Latest Proof")
                .font(.title3.weight(.semibold))
            if let proof = store.latestProof {
                VStack(alignment: .leading, spacing: 8) {
                    KeyValueLine(label: "Status", value: proof.status)
                    KeyValueLine(label: "Method", value: proof.method ?? "unknown")
                    KeyValueLine(label: "Message", value: proof.message)
                    if let solution = proof.solution {
                        KeyValueLine(label: "Feasible Size", value: "\(solution.count)")
                    }
                    KeyValueLine(label: "Elapsed", value: String(format: "%.2fs", proof.elapsedSeconds))
                }
            } else {
                EmptyState(text: "No proof result yet.")
            }
        }
        .padding(16)
        .background(.white.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 18))
    }

    private var logsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Proof Logs")
                .font(.title3.weight(.semibold))
            ScrollView {
                Text(store.logs)
                    .font(.system(.footnote, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .frame(minHeight: 220)
            .background(Color.black.opacity(0.92))
            .foregroundStyle(.yellow)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .padding(16)
        .background(.white.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 18))
    }
}

struct ValidateView: View {
    @ObservedObject var store: MobileAppStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    HeroCard(
                        title: "Validation Center",
                        subtitle: "Audit saved JSON outputs or validate a specific result file from the mobile interface."
                    )

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Validation Actions")
                            .font(.title3.weight(.semibold))
                        LabeledField("API Base URL", text: $store.serverURL)
                        LabeledField("DB Directory", text: $store.dbDir)
                        LabeledField("Result Filename", text: $store.resultFilename)
                        LabeledField("File Path", text: $store.validationFilePath)

                        HStack {
                            Button("Validate Saved Results") {
                                Task { await store.validateAll() }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(store.isBusy)
                            Button("Audit Saved Results") {
                                Task { await store.auditAll() }
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.isBusy)
                        }

                        HStack {
                            Button("Validate File") {
                                Task { await store.validateFile() }
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.isBusy)

                            Button("Audit File") {
                                Task { await store.auditFile() }
                            }
                            .buttonStyle(.bordered)
                            .disabled(store.isBusy)
                        }

                        Button("Certify Saved Result") {
                            Task { await store.certifySavedResult() }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(store.isBusy)

                        if let certification = store.latestCertification {
                            CertificationSummaryCard(summary: certification.summary)
                        }
                    }
                    .padding(16)
                    .background(.white.opacity(0.9))
                    .clipShape(RoundedRectangle(cornerRadius: 18))

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Validation Output")
                            .font(.title3.weight(.semibold))
                        if store.validationOutput.isEmpty {
                            EmptyState(text: "No validation output yet.")
                        } else {
                            ScrollView {
                                Text(store.validationOutput)
                                    .font(.system(.footnote, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(12)
                            }
                            .frame(minHeight: 260)
                            .background(Color(red: 0.1, green: 0.11, blue: 0.15))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    .padding(16)
                    .background(.white.opacity(0.9))
                    .clipShape(RoundedRectangle(cornerRadius: 18))
                }
                .padding(16)
            }
            .navigationTitle("Validate")
        }
    }
}

struct ResultsView: View {
    @ObservedObject var store: MobileAppStore

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                topPanel
                List {
                    ForEach(store.resultItems) { item in
                        Button {
                            Task { await store.loadResult(item) }
                        } label: {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(alignment: .top) {
                                    Text(item.filename)
                                        .font(.headline)
                                    Spacer(minLength: 8)
                                    ValidationStatusBadge(status: item.validationStatus ?? "unverified")
                                }
                                Text("size \(item.numGroups ?? -1) • \(item.coverageMode ?? "-") / \(item.aggregationMode ?? "-")")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 6)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .listStyle(.plain)

                if let result = store.selectedResult {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            ResultSummaryCard(result: result)
                            if let certification = store.latestCertification {
                                CertificationSummaryCard(summary: certification.summary)
                            }
                        }
                        .padding(16)
                    }
                    .frame(maxHeight: 320)
                    .background(.ultraThinMaterial)
                }
            }
            .navigationTitle("Results")
            .task {
                if store.resultItems.isEmpty {
                    await store.refreshResults()
                }
            }
        }
    }

    private var topPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            LabeledField("API Base URL", text: $store.serverURL)
            HStack {
                LabeledField("DB Directory", text: $store.dbDir)
                LabeledField("Filename", text: $store.resultFilename)
            }
            HStack {
                Button("Refresh Results") {
                    Task { await store.refreshResults() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(store.isBusy)

                Button("Certify Selected") {
                    Task { await store.certifySavedResult() }
                }
                .buttonStyle(.bordered)
                .disabled(store.isBusy || store.resultFilename.isEmpty)

                Button("Delete Selected") {
                    Task { await store.deleteSelectedResult() }
                }
                .buttonStyle(.bordered)
                .disabled(store.isBusy)
            }
        }
        .padding(16)
        .background(.white.opacity(0.92))
    }
}

struct HeroCard: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.system(size: 30, weight: .bold, design: .rounded))
            Text(subtitle)
                .font(.body)
                .foregroundStyle(.white.opacity(0.9))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.12, green: 0.34, blue: 0.98),
                    Color(red: 1.0, green: 0.44, blue: 0.16)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .foregroundStyle(.white)
        .clipShape(RoundedRectangle(cornerRadius: 24))
        .shadow(color: .orange.opacity(0.22), radius: 20, y: 12)
    }
}

struct NumericGrid: View {
    let fields: [(String, Binding<String>)]

    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            ForEach(Array(fields.enumerated()), id: \.offset) { _, field in
                LabeledField(field.0, text: field.1)
            }
        }
    }
}

struct LabeledField: View {
    let label: String
    @Binding var text: String

    init(_ label: String, text: Binding<String>) {
        self.label = label
        self._text = text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            TextField(label, text: $text)
                .textFieldStyle(.roundedBorder)
        }
    }
}

struct EmptyState: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct KeyValueLine: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label.uppercased())
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.body)
        }
    }
}

struct ResultSummaryCard: View {
    let result: SolverResultPayload

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                metric(title: "Best Size", value: "\(result.numGroups)")
                Spacer()
                metric(title: "Exact", value: result.exactSize.map(String.init) ?? "—")
                Spacer()
                metric(title: "Time", value: String(format: "%.2fs", result.elapsedSeconds))
            }

            Divider()

            HStack(alignment: .top, spacing: 10) {
                ValidationStatusBadge(status: result.validation?.displayStatus ?? "unverified")
                VStack(alignment: .leading, spacing: 4) {
                    Text("coverage: \(result.coverageMode)")
                    Text("aggregation: \(result.aggregationMode)")
                    if let validation = result.validation {
                        Text("primary \(validation.primaryValid ? "valid" : "invalid") • independent \(validation.independentValid ? "valid" : "invalid") • agree \(validation.methodsAgree ? "yes" : "no")")
                        Text("unsatisfied \(validation.unsatisfiedTargets) • deficit \(validation.deficitUnits)")
                    }
                }
                .font(.footnote)
                .foregroundStyle(.secondary)
            }

            Divider()

            Text("Selected Groups")
                .font(.headline)

            ForEach(Array(result.groups.prefix(12).enumerated()), id: \.offset) { index, group in
                Text("\(index + 1). \(group.map(String.init).joined(separator: ", "))")
                    .font(.system(.footnote, design: .monospaced))
            }

            if result.groups.count > 12 {
                Text("… \(result.groups.count - 12) more groups")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(Color.white)
        )
        .shadow(color: .black.opacity(0.05), radius: 12, y: 6)
    }

    private func metric(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title2.weight(.semibold))
        }
    }
}

struct ValidationStatusBadge: View {
    let status: String

    private var palette: (Color, Color) {
        switch status {
        case "validated":
            return (Color.green.opacity(0.16), Color.green)
        case "invalid":
            return (Color.red.opacity(0.14), Color.red)
        case "mismatch":
            return (Color.orange.opacity(0.16), Color.orange)
        default:
            return (Color.gray.opacity(0.14), Color.gray)
        }
    }

    var body: some View {
        let colors = palette
        Text(status.uppercased())
            .font(.caption.weight(.bold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(colors.0)
            .foregroundStyle(colors.1)
            .clipShape(Capsule())
    }
}

struct CertificationSummaryCard: View {
    let summary: CertificationSummaryPayload

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Optimality")
                    .font(.headline)
                Spacer()
                ValidationStatusBadge(status: statusToken)
            }

            Text(summary.statusLabel)
                .font(.title3.weight(.semibold))

            Text(summary.feasibilityLabel)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack {
                metric(title: "Incumbent", value: "\(summary.incumbentSize)")
                Spacer()
                metric(title: "Lower Bound", value: summary.certifiedLowerBound.map(String.init) ?? "—")
                Spacer()
                metric(title: "Gap", value: summary.gapUpperBound.map(String.init) ?? "—")
            }

            if let method = summary.method {
                Text("method: \(method)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(Color.white)
        )
        .shadow(color: .black.opacity(0.05), radius: 12, y: 6)
    }

    private var statusToken: String {
        switch summary.statusCode {
        case "certified_optimal":
            return "validated"
        case "not_optimal":
            return "invalid"
        case "unresolved":
            return "mismatch"
        default:
            return "unverified"
        }
    }

    private func metric(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased())
                .font(.caption.weight(.bold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.title3.weight(.semibold))
        }
    }
}

struct ProblemRequest: Codable {
    let m: Int
    let n: Int
    let k: Int
    let j: Int
    let s: Int
    let samples: [Int]
    let coverageMode: String
    let aggregationMode: String
    let requiredR: Int?
    let seed: Int?

    enum CodingKeys: String, CodingKey {
        case m, n, k, j, s, samples, seed
        case coverageMode = "coverage_mode"
        case aggregationMode = "aggregation_mode"
        case requiredR = "required_r"
    }
}

struct SolverRequest: Codable {
    let nRestarts: Int
    let useILP: Bool
    let exactBackend: String
    let exactTimeLimit: Int
    let forceExact: Bool
    let maxLocalSteps: Int?
    let maxSAIterations: Int?
    let candidateSampleSize: Int
    let adaptiveNeighborhoods: Bool
    let reducedExactPolish: Bool
    let reducedExactTimeLimit: Int
    let reducedExactCoreCap: Int
    let saveResult: Bool
    let dbDir: String

    enum CodingKeys: String, CodingKey {
        case nRestarts = "n_restarts"
        case useILP = "use_ilp"
        case exactBackend = "exact_backend"
        case exactTimeLimit = "exact_time_limit"
        case forceExact = "force_exact"
        case maxLocalSteps = "max_local_steps"
        case maxSAIterations = "max_sa_iterations"
        case candidateSampleSize = "candidate_sample_size"
        case adaptiveNeighborhoods = "adaptive_neighborhoods"
        case reducedExactPolish = "reduced_exact_polish"
        case reducedExactTimeLimit = "reduced_exact_time_limit"
        case reducedExactCoreCap = "reduced_exact_core_cap"
        case saveResult = "save_result"
        case dbDir = "db_dir"
    }
}

struct SolveRequestEnvelope: Codable {
    let problem: ProblemRequest
    let solver: SolverRequest
}

struct SolveResponse: Decodable {
    let result: SolverResultPayload
    let logs: String
}

struct SolverResultPayload: Codable {
    let solutionIndices: [Int]
    let groups: [[Int]]
    let numGroups: Int
    let exactSize: Int?
    let exactMethod: String?
    let samples: [Int]
    let elapsedSeconds: Double
    let seed: Int
    let numTargets: Int
    let numCandidates: Int
    let aggregationMode: String
    let coverageMode: String
    let validation: ResultValidationPayload?

    enum CodingKeys: String, CodingKey {
        case solutionIndices = "solution_indices"
        case groups
        case numGroups = "num_groups"
        case exactSize = "exact_size"
        case exactMethod = "exact_method"
        case samples
        case elapsedSeconds = "elapsed_seconds"
        case seed
        case numTargets = "num_targets"
        case numCandidates = "num_candidates"
        case aggregationMode = "aggregation_mode"
        case coverageMode = "coverage_mode"
        case validation
    }
}

struct ResultValidationPayload: Codable {
    let primaryValid: Bool
    let independentValid: Bool
    let methodsAgree: Bool
    let unsatisfiedTargets: Int
    let deficitUnits: Int

    enum CodingKeys: String, CodingKey {
        case primaryValid = "primary_valid"
        case independentValid = "independent_valid"
        case methodsAgree = "methods_agree"
        case unsatisfiedTargets = "unsatisfied_targets"
        case deficitUnits = "deficit_units"
    }

    var displayStatus: String {
        if !methodsAgree {
            return "mismatch"
        }
        if primaryValid && independentValid {
            return "validated"
        }
        return "invalid"
    }
}

struct ProveBoundRequestEnvelope: Codable {
    let problem: ProblemRequest
    let targetSize: Int
    let exactBackend: String
    let exactTimeLimit: Int

    enum CodingKeys: String, CodingKey {
        case problem
        case targetSize = "target_size"
        case exactBackend = "exact_backend"
        case exactTimeLimit = "exact_time_limit"
    }
}

struct ProofResponse: Decodable {
    let proof: ProofPayload
    let logs: String
}

struct ProofPayload: Codable {
    let status: String
    let solution: [Int]?
    let method: String?
    let message: String
    let elapsedSeconds: Double

    enum CodingKeys: String, CodingKey {
        case status, solution, method, message
        case elapsedSeconds = "elapsed_seconds"
    }
}

struct ValidationFileRequest: Codable {
    let path: String
}

struct ValidationResultsRequest: Codable {
    let dbDir: String

    enum CodingKeys: String, CodingKey {
        case dbDir = "db_dir"
    }
}

struct ResultFilenameRequest: Codable {
    let filename: String
    let dbDir: String

    enum CodingKeys: String, CodingKey {
        case filename
        case dbDir = "db_dir"
    }
}

struct CertifyResultRequest: Codable {
    let filename: String
    let dbDir: String
    let exactBackend: String
    let exactTimeLimit: Int

    enum CodingKeys: String, CodingKey {
        case filename
        case dbDir = "db_dir"
        case exactBackend = "exact_backend"
        case exactTimeLimit = "exact_time_limit"
    }
}

struct ValidationResponse: Decodable {
    let report: ValidationReportPayload
    let formatted: String
}

struct ValidationAllResponse: Decodable {
    let reports: [ValidationReportPayload]
    let formatted: [String]
}

struct AuditResponse: Decodable {
    let formatted: String
}

struct AuditAllResponse: Decodable {
    let formatted: [String]
}

struct CertificationResponse: Decodable {
    let report: CertificationPayload
    let summary: CertificationSummaryPayload
    let summaryText: String
    let formatted: String

    enum CodingKeys: String, CodingKey {
        case report, summary, formatted
        case summaryText = "summary_text"
    }
}

struct CertificationPayload: Codable {
    let status: String
    let method: String?
    let message: String
    let incumbentSize: Int
    let proofTarget: Int
    let certifiedOptimal: Bool
    let certifiedLowerBound: Int?

    enum CodingKeys: String, CodingKey {
        case status, method, message
        case incumbentSize = "incumbent_size"
        case proofTarget = "proof_target"
        case certifiedOptimal = "certified_optimal"
        case certifiedLowerBound = "certified_lower_bound"
    }
}

struct CertificationSummaryPayload: Codable {
    let source: String
    let statusCode: String
    let statusLabel: String
    let incumbentSize: Int
    let feasibilityLabel: String
    let certifiedOptimal: Bool
    let certifiedLowerBound: Int?
    let gapUpperBound: Int?
    let betterSolutionSize: Int?
    let method: String?

    enum CodingKeys: String, CodingKey {
        case source, method
        case statusCode = "status_code"
        case statusLabel = "status_label"
        case incumbentSize = "incumbent_size"
        case feasibilityLabel = "feasibility_label"
        case certifiedOptimal = "certified_optimal"
        case certifiedLowerBound = "certified_lower_bound"
        case gapUpperBound = "gap_upper_bound"
        case betterSolutionSize = "better_solution_size"
    }
}

struct ValidationReportPayload: Codable, Identifiable {
    var id: String { source }
    let source: String
    let isValid: Bool
    let numGroups: Int
    let declaredNumGroups: Int?
    let unsatisfiedTargets: Int
    let deficitUnits: Int
    let issues: [ValidationIssuePayload]
    let uncoveredExamples: [String]

    enum CodingKeys: String, CodingKey {
        case source
        case isValid = "is_valid"
        case numGroups = "num_groups"
        case declaredNumGroups = "declared_num_groups"
        case unsatisfiedTargets = "unsatisfied_targets"
        case deficitUnits = "deficit_units"
        case issues
        case uncoveredExamples = "uncovered_examples"
    }
}

struct ValidationIssuePayload: Codable, Identifiable {
    var id: String { severity + message }
    let severity: String
    let message: String
}

struct ResultListResponse: Decodable {
    let items: [ResultListItem]
}

struct ResultListItem: Codable, Identifiable {
    var id: String { filename }
    let filename: String
    let numGroups: Int?
    let coverageMode: String?
    let aggregationMode: String?
    let timestamp: String?
    let exactSize: Int?
    let validationStatus: String?

    enum CodingKeys: String, CodingKey {
        case filename
        case numGroups = "num_groups"
        case coverageMode = "coverage_mode"
        case aggregationMode = "aggregation_mode"
        case timestamp
        case exactSize = "exact_size"
        case validationStatus = "validation_status"
    }
}

struct ResultShowResponse: Decodable {
    let result: SolverResultPayload
}

struct DeleteResultResponse: Decodable {
    let deleted: String
}

struct DemoResponse: Decodable {
    let examples: [DemoExample]
}

struct DemoExample: Codable, Identifiable {
    var id: String { title }
    let title: String
    let problem: ProblemRequest
}

struct APIErrorResponse: Decodable, Error {
    let error: String
}

struct EmptyResponse: Decodable {}

struct MobileAPIClient {
    let baseURL: String

    private var session: URLSession {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 600
        configuration.timeoutIntervalForResource = 3600
        configuration.waitsForConnectivity = true
        return URLSession(configuration: configuration)
    }

    private var decoder: JSONDecoder {
        JSONDecoder()
    }

    func solve(problem: ProblemRequest, solver: SolverRequest) async throws -> SolveResponse {
        try await post("/solve", body: SolveRequestEnvelope(problem: problem, solver: solver))
    }

    func proveBound(
        problem: ProblemRequest,
        targetSize: Int,
        exactBackend: String,
        exactTimeLimit: Int
    ) async throws -> ProofResponse {
        try await post(
            "/prove-bound",
            body: ProveBoundRequestEnvelope(
                problem: problem,
                targetSize: targetSize,
                exactBackend: exactBackend,
                exactTimeLimit: exactTimeLimit
            )
        )
    }

    func validateFile(path: String) async throws -> ValidationResponse {
        try await post("/validate-file", body: ValidationFileRequest(path: path))
    }

    func validateResults(dbDir: String) async throws -> ValidationAllResponse {
        try await post("/validate-results", body: ValidationResultsRequest(dbDir: dbDir))
    }

    func auditFile(path: String) async throws -> AuditResponse {
        try await post("/audit-file", body: ValidationFileRequest(path: path))
    }

    func auditResults(dbDir: String) async throws -> AuditAllResponse {
        try await post("/audit-results", body: ValidationResultsRequest(dbDir: dbDir))
    }

    func certifyResult(
        filename: String,
        dbDir: String,
        exactBackend: String,
        exactTimeLimit: Int
    ) async throws -> CertificationResponse {
        try await post(
            "/certify-result",
            body: CertifyResultRequest(
                filename: filename,
                dbDir: dbDir,
                exactBackend: exactBackend,
                exactTimeLimit: exactTimeLimit
            )
        )
    }

    func listResults(dbDir: String) async throws -> ResultListResponse {
        try await get("/results?db_dir=\(dbDir.urlQueryEscaped)")
    }

    func showResult(filename: String, dbDir: String) async throws -> ResultShowResponse {
        try await get("/results/\(filename.urlPathEscaped)?db_dir=\(dbDir.urlQueryEscaped)")
    }

    func deleteResult(filename: String, dbDir: String) async throws -> DeleteResultResponse {
        try await delete("/results/\(filename.urlPathEscaped)?db_dir=\(dbDir.urlQueryEscaped)")
    }

    func fetchDemos() async throws -> DemoResponse {
        try await post("/demo", body: EmptyRequest())
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        try await request(path: path, method: "GET", body: nil)
    }

    private func post<T: Decodable, Body: Encodable>(_ path: String, body: Body) async throws -> T {
        let encoded = try JSONEncoder().encode(body)
        return try await request(path: path, method: "POST", body: encoded)
    }

    private func delete<T: Decodable>(_ path: String) async throws -> T {
        try await request(path: path, method: "DELETE", body: nil)
    }

    private func request<T: Decodable>(path: String, method: String, body: Data?) async throws -> T {
        #if !targetEnvironment(simulator)
        if baseURL.contains("127.0.0.1") || baseURL.contains("localhost") {
            throw NSError(
                domain: "MobileAPI",
                code: -1000,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "This iPhone is using \(baseURL). On a physical device, 127.0.0.1 or localhost points to the phone itself, not your Mac. Replace API Base URL with your Mac's LAN IP, such as http://192.168.x.x:8000."
                ]
            )
        }
        #endif

        guard let url = URL(string: baseURL + path) else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 600
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            if let apiError = try? decoder.decode(APIErrorResponse.self, from: data) {
                throw NSError(domain: "MobileAPI", code: http.statusCode, userInfo: [
                    NSLocalizedDescriptionKey: apiError.error
                ])
            }
            throw NSError(domain: "MobileAPI", code: http.statusCode, userInfo: [
                NSLocalizedDescriptionKey: HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            ])
        }
        return try decoder.decode(T.self, from: data)
    }
}

struct EmptyRequest: Codable {}

extension String {
    var urlQueryEscaped: String {
        addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? self
    }

    var urlPathEscaped: String {
        addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? self
    }
}
