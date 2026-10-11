import XCTest
@testable import TextureStudio

final class WorkbenchTrainingProgressTests: XCTestCase {
    func testOnDemandCropAndDepthProPreparationKeepTheActiveUpdateVisible() {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "training_started", "requested_updates": 365, "updates_per_map": 1])
        progress.consume(["event": "update_started", "completed_updates": 0, "current_update": 1,
            "sample_id": "stone", "sample_position": 1, "sample_total": 365])
        progress.consume(["event": "crop_preparation", "operation": "Caching DepthPro input for crop",
            "completed": 0, "total": 1, "sample_id": "stone", "sample_position": 1, "sample_total": 1,
            "epoch": 1, "workflow_phase": 3])
        XCTAssertEqual(progress.state, .training)
        XCTAssertEqual(progress.phaseIndex, 3)
        XCTAssertEqual(progress.currentUpdateSummary, "Running step 1 of 365")
        XCTAssertEqual(progress.operationLabel, "Caching DepthPro input for crop")
        XCTAssertEqual(progress.operationDetail, "0 / 1 cached crops refreshed")
        XCTAssertEqual(progress.sampleID, "stone")
        XCTAssertEqual(progress.sampleSummary, "Training map 1 / 1")
        progress.consume(["event": "depthpro_preparation", "phase": "training", "workflow_phase": 3,
            "operation": "Caching DepthPro input for training map", "completed": 0, "total": 1,
            "sample_id": "stone", "sample_position": 1])
        XCTAssertEqual(progress.state, .training)
        XCTAssertEqual(progress.phaseIndex, 3)
        XCTAssertEqual(progress.currentUpdateSummary, "Running step 1 of 365")
        XCTAssertEqual(progress.operationLabel, "Caching DepthPro input for training map")
        progress.consume(["event": "depthpro_preparation", "phase": "validation", "workflow_phase": 4,
            "completed": 1, "total": 1, "sample_id": "check"])
        XCTAssertEqual(progress.state, .validation)
        XCTAssertEqual(progress.phaseIndex, 4)
    }

    @MainActor func testDepthProDownloadShowsByteProgressAndPersistsReadyPathBeforeTrainingStarts() async throws {
        let suite = "depthpro-download-progress-\(UUID())", defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training DepthPro Height Refiner", training: true) {
            store.recordTrainingProgress("{\"event\":\"depthpro_model_setup\",\"operation\":\"Downloading DepthPro model\",\"bytes_completed\":100000000,\"bytes_total\":1902515337}\n")
            XCTAssertEqual(store.activity, "Downloading DepthPro model · 100 MB / 1.9 GB")
            XCTAssertEqual(store.trainingProgress?.state, .modelSetup)
            XCTAssertEqual(store.trainingProgress?.completedUpdates, 0)
            XCTAssertNil(store.trainingProgress?.stageSummary)
            XCTAssertFalse(store.hasTrainingStarted)
            let path = NativeDepthProContract.managedModelURL.path
            let event: [String: Any] = ["event": "depthpro_model_setup", "operation": "DepthPro model ready", "ready": true, "model_path": path]
            store.recordTrainingProgress(String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self) + "\n")
            XCTAssertEqual(store.depthProModelPath, path)
            XCTAssertFalse(store.hasTrainingStarted)
            store.recordTrainingProgress("{\"event\":\"training_started\",\"requested_updates\":1}\n")
            XCTAssertTrue(store.hasTrainingStarted)
            XCTAssertEqual(store.trainingProgress?.operationDetail, "")
        }
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(WorkbenchStore(preferences: defaults).depthProModelPath, NativeDepthProContract.managedModelURL.path)
    }

    func testCropPreparationUsesSetupThenTrainingWithoutResettingOptimizerPosition() {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "training_setup", "requested_updates": 12, "updates_per_map": 4,
            "initial_step": 100])
        progress.consume(["event": "crop_preparation", "operation": "Refreshing cached crops",
            "completed": 1, "total": 3, "epoch": 1])
        XCTAssertEqual(progress.state, .modelSetup)
        XCTAssertEqual(progress.phaseIndex, 2)
        XCTAssertEqual(progress.operationDetail, "1 / 3 cached crops refreshed")
        XCTAssertEqual(progress.stageSummary, "1 / 3 cached crops refreshed")
        XCTAssertEqual(progress.updateSummary, "Total steps: 0 / 12")
        progress.consume(["event": "training_started", "requested_updates": 12])
        XCTAssertEqual(progress.operationDetail, "")
        XCTAssertNil(progress.stageSummary)

        progress.consume(["event": "update", "completed_updates": 3, "current_update": 3,
            "checkpoint_step": 103, "epoch": 1, "total": 0.2, "update_duration_seconds": 0.5])
        progress.consume(["event": "crop_preparation", "operation": "Refreshing cached crops",
            "completed": 2, "total": 3, "epoch": 2])
        XCTAssertEqual(progress.state, .training)
        XCTAssertEqual(progress.phaseIndex, 3)
        XCTAssertEqual(progress.operationLabel, "Refreshing cached crops")
        XCTAssertEqual(progress.operationDetail, "2 / 3 cached crops refreshed")
        XCTAssertEqual(progress.completedUpdates, 3)
        XCTAssertEqual(progress.currentUpdate, 3)
        XCTAssertEqual(progress.totalUpdates, 12)
        XCTAssertEqual(progress.initialStep, 100)
        XCTAssertEqual(progress.checkpointStep, 103)
        XCTAssertEqual(progress.updateSummary, "Total steps: 3 / 12")
        XCTAssertEqual(progress.epochSummary, "Epoch 2 / 4")
        XCTAssertEqual(progress.fractionCompleted, 0.25)
        XCTAssertEqual(progress.lastLoss, 0.2)
        XCTAssertEqual(progress.lastUpdateSeconds, 0.5)
        XCTAssertNil(progress.sampleSummary)
        progress.consume(["event": "update", "completed_updates": 4, "current_update": 4])
        XCTAssertEqual(progress.operationDetail, "")
        XCTAssertNil(progress.stageSummary)
        XCTAssertEqual(progress.updateSummary, "Total steps: 4 / 12")
    }

    @MainActor func testStoreShowsCropRefreshAndPreservesVisibleTrainingCounters() async throws {
        let suite = "crop-progress-\(UUID())", defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training", training: true) {
            store.recordTrainingProgress("{\"event\":\"crop_preparation\",\"operation\":\"Refreshing cached crops\",\"completed\":0,\"total\":4,\"epoch\":1}\n")
            XCTAssertEqual(store.activity, "Refreshing cached crops · 0 / 4 cached crops refreshed")
            XCTAssertEqual(store.trainingProgress?.state, .modelSetup)
            XCTAssertFalse(store.hasTrainingStarted)
            store.recordTrainingProgress("{\"event\":\"training_started\",\"requested_updates\":8,\"updates_per_map\":2}\n")
            store.recordTrainingProgress("{\"event\":\"update\",\"completed_updates\":4,\"current_update\":4,\"epoch\":1}\n")
            store.recordTrainingProgress("{\"event\":\"crop_preparation\",\"operation\":\"Refreshing cached crops\",\"completed\":2,\"total\":4,\"epoch\":2}\n")
            XCTAssertEqual(store.activity, "Refreshing cached crops · 2 / 4 cached crops refreshed")
            XCTAssertTrue(store.hasTrainingStarted)
            XCTAssertEqual(store.trainingProgress?.state, .training)
            XCTAssertEqual(store.trainingProgress?.updateSummary, "Total steps: 4 / 8")
            XCTAssertEqual(store.trainingProgress?.epochSummary, "Epoch 2 / 2")
            store.recordTrainingProgress("{\"event\":\"update_started\",\"completed_updates\":4,\"current_update\":5,\"epoch\":2}\n")
            XCTAssertEqual(store.trainingProgress?.operationDetail, "")
            XCTAssertNil(store.trainingProgress?.stageSummary)
            XCTAssertTrue(store.activity.contains("Running step 5 of 8"))
            XCTAssertFalse(store.activity.contains("cached crops"))
        }
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
    }

    func testDepthProPreparationShowsInputProgressBeforeTrainingAndClearsAtFirstUpdate() {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "training_setup", "operation": "Preparing model execution graph", "completed": 2, "total": 3])
        progress.consume(["event": "depthpro_preparation", "completed": 1, "total": 5, "sample_id": "stone",
            "sample_position": 2, "requested_updates": 10, "completed_updates": 0])
        XCTAssertEqual(progress.state, .modelSetup)
        XCTAssertEqual(progress.phaseIndex, 2)
        XCTAssertEqual(progress.operationLabel, "Caching DepthPro inputs")
        XCTAssertEqual(progress.operationDetail, "1 / 5 inputs processed")
        XCTAssertEqual(progress.stageSummary, "1 / 5 inputs processed")
        XCTAssertEqual(progress.sampleSummary, "Input map 2 / 5")
        XCTAssertEqual(progress.sampleID, "stone")
        XCTAssertEqual(progress.completedUpdates, 0)
        progress.consume(["event": "depthpro_preparation", "completed": 5, "total": 5, "sample_id": "wood", "sample_position": 5])
        XCTAssertEqual(progress.stageCompleted, 5)
        progress.consume(["event": "training_started", "requested_updates": 10])
        XCTAssertEqual(progress.state, .training)
        XCTAssertEqual(progress.phaseIndex, 3)
        XCTAssertEqual(progress.operationDetail, "")
        XCTAssertNil(progress.sampleID)
        XCTAssertNil(progress.sampleSummary)
        XCTAssertNil(progress.stageSummary)
    }

    @MainActor func testStoreShowsDepthProPreparationActivityBeforeTrainingStarts() async throws {
        let suite = "depthpro-progress-\(UUID())", defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training DepthPro Height Refiner", training: true) {
            store.recordTrainingProgress("{\"event\":\"depthpro_preparation\",\"completed\":2,\"total\":4,\"sample_id\":\"stone\",\"sample_position\":3}\n")
            XCTAssertEqual(store.activity, "Caching DepthPro inputs · 2 / 4 inputs processed · stone")
            XCTAssertEqual(store.trainingProgress?.state, .modelSetup)
            XCTAssertFalse(store.hasTrainingStarted)
            store.recordTrainingProgress("{\"event\":\"training_started\",\"requested_updates\":8}\n")
            XCTAssertTrue(store.hasTrainingStarted)
            XCTAssertEqual(store.trainingProgress?.state, .training)
            XCTAssertNil(store.trainingProgress?.sampleID)
        }
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
    }

    func testSampleSkipsRemainVisibleThroughUnavailableValidationAndSaving() {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "sample_skipped", "sample_id": "bad-map", "error": "Unreadable image",
            "skipped_sample_count": 1, "completed_updates": 4])
        XCTAssertEqual(progress.skippedSampleCount, 1)
        XCTAssertEqual(progress.completedUpdates, 4)
        XCTAssertEqual(progress.operationDetail, "Unreadable image")
        progress.consume(["event": "validation", "status": "unavailable", "mae": NSNull()])
        XCTAssertNil(progress.validationError)
        XCTAssertTrue(progress.operationLabel.contains("saving can continue"))
        progress.consume(["event": "checkpoint_saved"])
        progress.consume(["event": "training_completed", "status": "completed"])
        XCTAssertEqual(progress.state, .completed)
        XCTAssertEqual(progress.skippedSampleCount, 1)
    }

    func testUpdateTimingAndAccumulationAreIndependentOfOptimizerStepCount() {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "training_started", "gradient_accumulation_steps": 4, "learning_rate": 0.00001])
        progress.consume(["event": "update_started", "completed_updates": 2, "current_update": 3])
        progress.consume(["event": "accumulation_sample", "accumulation_step": 2, "gradient_accumulation_steps": 4, "sample_id": "stone"])
        XCTAssertEqual(progress.completedUpdates, 2)
        XCTAssertEqual(progress.currentUpdate, 3)
        XCTAssertEqual(progress.accumulationStep, 2)
        XCTAssertEqual(progress.gradientAccumulationSteps, 4)
        XCTAssertEqual(progress.sampleID, "stone")
        progress.consume(["event": "update", "completed_updates": 3, "current_update": 3,
            "learning_rate": 0.000002, "update_duration_seconds": 75.4])
        XCTAssertEqual(progress.completedUpdates, 3)
        XCTAssertEqual(progress.accumulationStep, 0)
        XCTAssertEqual(progress.lastUpdateSeconds, 75.4)
        XCTAssertEqual(progress.learningRate, 0.000002)
        progress.consume(["event": "update", "update_duration_seconds": Double.nan, "learning_rate": Double.infinity])
        XCTAssertEqual(progress.lastUpdateSeconds, 75.4)
        XCTAssertEqual(progress.learningRate, 0.000002)
    }

    func testSetupTrainingValidationAndSavingKeepSeparateCounters() throws {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "preparation_progress", "completed": 2, "total": 6])
        XCTAssertEqual(progress.phaseIndex, 1)
        XCTAssertEqual(progress.stageCompleted, 2)
        progress.consume(["event": "training_setup", "operation": "Loading material model", "completed": 1, "total": 3,
            "requested_updates": 12, "updates_per_map": 4])
        XCTAssertEqual(progress.phaseIndex, 2)
        XCTAssertEqual(progress.totalUpdates, 12)
        XCTAssertEqual(progress.totalEpochs, 4)
        progress.consume(["event": "training_started", "requested_updates": 12, "updates_per_map": 4, "initial_step": 100])
        XCTAssertNil(progress.currentUpdateSummary)
        XCTAssertNil(progress.epochSummary)
        progress.consume(["event": "update_started", "completed_updates": 4, "current_update": 5, "checkpoint_step": 104,
            "epoch": 2, "sample_position": 2, "sample_total": 3, "sample_id": "soil"])
        XCTAssertEqual(progress.currentUpdateSummary, "Running step 5 of 12")
        XCTAssertEqual(progress.epochSummary, "Epoch 2 / 4")
        progress.consume(["event": "operation_progress", "phase": "training", "operation": "Backward pass", "completed": 3, "total": 8])
        XCTAssertEqual(progress.operationLabel, "Backward pass")
        XCTAssertEqual(progress.sampleID, "soil")
        XCTAssertEqual(progress.stageCompleted, 3)
        XCTAssertEqual(progress.stageTotal, 8)
        progress.consume(["event": "update", "completed_updates": 5, "current_update": 5, "checkpoint_step": 105, "total": 0.2])
        progress.consume(["event": "validation_started", "context": "checkpoint", "scope": "full", "sample_count": 2])
        XCTAssertNil(progress.sampleID)
        XCTAssertEqual(progress.samplePosition, 0)
        progress.consume(["event": "validation_sample", "context": "checkpoint", "sample_position": 2, "sample_total": 2, "sample_id": "rock"])
        progress.consume(["event": "operation_progress", "phase": "validation", "operation": "Forward pass", "completed": 7, "total": 10])
        XCTAssertEqual(progress.state, .validation)
        XCTAssertEqual(progress.completedUpdates, 5)
        XCTAssertEqual(progress.totalUpdates, 12)
        XCTAssertEqual(progress.epoch, 2)
        XCTAssertEqual(progress.sampleID, "rock")
        XCTAssertEqual(progress.sampleSummary, "Validation crop 2 / 2")
        XCTAssertEqual(progress.phaseIndex, 3)
        progress.consume(["event": "validation_started", "context": "final", "sample_count": 2, "workflow_phase": 4])
        XCTAssertEqual(progress.phaseIndex, 4)
        XCTAssertNil(progress.sampleID)
        progress.consume(["event": "checkpoint_started"])
        XCTAssertEqual(progress.state, .checkpoint)
        XCTAssertEqual(progress.stageTotal, 0)
        progress.consume(["event": "export_started", "elapsed_training_seconds": 34.5])
        XCTAssertEqual(progress.state, .export)
        XCTAssertEqual(progress.completedUpdates, 5)
        XCTAssertNil(progress.sampleSummary)
        XCTAssertEqual(progress.elapsedSeconds, 34.5)
        progress.consume(["event": "training_completed", "status": "stopped"])
        XCTAssertEqual(progress.state, .stopped)
        XCTAssertEqual(progress.checkpointStep, 105)
        XCTAssertNil(progress.currentUpdateSummary)
        XCTAssertEqual(progress.fractionCompleted, 5.0 / 12.0, accuracy: 0.00001)
    }

    func testAbortAndFailureDoNotClaimAnUnfinishedStepIsStillRunning() {
        for finalState in [WorkbenchTrainingProgress.State.aborted, .failed] {
            var progress = WorkbenchTrainingProgress()
            progress.consume(["event": "update_started", "completed_updates": 1, "current_update": 2, "requested_updates": 5])
            XCTAssertNotNil(progress.currentUpdateSummary)
            progress.finish(state: finalState)
            XCTAssertEqual(progress.completedUpdates, 1)
            XCTAssertNil(progress.currentUpdateSummary)
        }
    }

    @MainActor func testStoreRetainsProgressDuringGracefulStopAndClearsItForAnotherOperation() async throws {
        let suite = "training-progress-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training", training: true) {
            store.recordTrainingProgress("{\"event\":\"training_started\",\"requested_updates\":5,\"updates_per_map\":5}\n")
            store.recordTrainingProgress("{\"event\":\"update_started\",\"completed_updates\":1,\"current_update\":2}\n")
            store.stopAndSave()
            store.recordTrainingProgress("{\"event\":\"operation_progress\",\"phase\":\"training\",\"operation\":\"Backward pass\",\"completed\":3,\"total\":8}\n")
            XCTAssertEqual(store.trainingProgress?.operationLabel, "Backward pass")
            XCTAssertEqual(store.trainingProgress?.stageCompleted, 3)
            store.recordTrainingProgress("{\"event\":\"validation\",\"sample_count\":0,\"pool_count\":0,\"mae\":null}\n")
            XCTAssertEqual(store.validationSummary, "")
            store.recordTrainingProgress("{\"event\":\"training_completed\",\"status\":\"stopped\",\"completed_updates\":2,\"current_update\":2}\n")
        }
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(store.trainingProgress?.state, .stopped)
        XCTAssertEqual(store.trainingProgress?.completedUpdates, 2)
        store.operation("Open another dataset") { }
        XCTAssertNil(store.trainingProgress)
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
    }
}
