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
        XCTAssertEqual(calls, [["refresh-classifier-dataset", "--source-folder", "/tmp/original materials", "--destination", "/tmp/classifier"]])
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
