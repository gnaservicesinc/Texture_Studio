import XCTest
@testable import TextureStudio

@MainActor final class WorkbenchClassifierDatasetTests: XCTestCase {
    func testRefreshUsesIndependentSourceAndDestinationWithoutPreparingTrainingData() async throws {
        let suite = "org.ipde.classifier-ui.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        var calls: [[String]] = []
        let store = WorkbenchStore(preferences: defaults, workerOverride: { arguments, _ in
            calls.append(arguments)
            return """
            {"dataset_path":"/tmp/classifier","manifest_path":"/tmp/manifest.json","review_path":"/tmp/review.tsv","training_path":"/tmp/classifier-createml","training_ready":true,"total":12,"added":2,"preserved":10,"reclassified":4,"needs_review":3,"classes":[{"id":"wood","name":"Wood","count":12}],"warnings":[]}
            """
        })
        store.classifierSourceFolderPath = "/tmp/original materials"
        store.classifierDatasetPath = "/tmp/classifier"
        store.refreshClassifierDataset()
        for _ in 0..<100 where store.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls, [["refresh-classifier-dataset", "--source-folder", "/tmp/original materials", "--destination", "/tmp/classifier", "--size", "2048"]])
        XCTAssertEqual(store.classifierDatasetResult?.total, 12)
        XCTAssertEqual(store.classifierDatasetResult?.needsReview, 3)
        XCTAssertEqual(store.classifierDatasetResult?.trainingPath, "/tmp/classifier-createml")
        XCTAssertEqual(store.classifierDatasetResult?.trainingReady, true)
        XCTAssertTrue(store.classifierDatasetResult?.summary.contains("4 reclassified") == true)
        XCTAssertNil(store.dataset)
        let reopened = WorkbenchStore(preferences: defaults)
        XCTAssertEqual(reopened.classifierSourceFolderPath, "/tmp/original materials")
        XCTAssertEqual(reopened.classifierDatasetPath, "/tmp/classifier")
    }

    func testOpenDatasetUsesItsGridDespiteAStaleSourceFolderUntilExplicitlyOverridden() async throws {
        let suite = "org.ipde.classifier-source.\(UUID().uuidString)", defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set("/tmp/stale development corpus", forKey: "classifierSourceFolder")
        var calls: [[String]] = []
        let store = WorkbenchStore(preferences: defaults, workerOverride: { arguments, _ in
            calls.append(arguments)
            return """
            {"dataset_path":"/tmp/classifier","manifest_path":"/tmp/manifest.json","review_path":"/tmp/review.html","training_path":"/tmp/classifier-createml","training_ready":true,"training_count":9,"minimum_native_size":2048,"eligible_count":12,"undersized_count":4,"unreadable_count":1,"total":15,"added":2,"preserved":10,"needs_review":3,"classes":[],"warnings":[]}
            """
        })
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
        {"dataset_path":"/tmp/open material dataset","index_sha256":"index","materials":[],"training_size":2048}
        """)
        store.classifierDatasetPath = "/tmp/classifier"
        store.refreshClassifierDataset()
        for _ in 0..<100 where store.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.first, ["refresh-classifier-dataset", "--dataset", "/tmp/open material dataset", "--destination", "/tmp/classifier", "--size", "2048"])
        XCTAssertTrue(store.classifierDatasetResult?.summary.contains("9 Create ML images") == true)
        XCTAssertTrue(store.classifierDatasetResult?.summary.contains("4 smaller originals preserved") == true)
        store.classifierUseSourceFolder = true
        store.classifierSourceMinimumSize = 4096
        store.refreshClassifierDataset()
        for _ in 0..<100 where store.isBusy { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(calls.last, ["refresh-classifier-dataset", "--source-folder", "/tmp/stale development corpus", "--destination", "/tmp/classifier", "--size", "4096"])
        let reopened = WorkbenchStore(preferences: defaults)
        XCTAssertTrue(reopened.classifierUseSourceFolder)
        XCTAssertEqual(reopened.classifierSourceMinimumSize, 4096)
    }

    func testMissingSourceOpensSettingsBeforeRunningWorker() {
        let suite = "org.ipde.classifier-ui.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults, workerOverride: { _, _ in
            XCTFail("No worker should start without a source")
            return ""
        })
        store.classifierSourceFolderPath = ""
        store.refreshClassifierDataset()
        XCTAssertTrue(store.showClassifierDatasetSettings)
        XCTAssertFalse(store.isBusy)
        XCTAssertNil(store.error)
    }
}
